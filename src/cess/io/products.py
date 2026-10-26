"""Versioned product layout and read-only views; no scientific recalculation.

``structure_product`` is an in-place final step for a new atomic writer, not a
transaction on an existing scientific file. ``convert_product`` supplies that
transaction for existing schema-2.5 products and never changes its source.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import tempfile
import uuid

import h5py
import numpy as np

from cess._version import __version__

SCHEMA_VERSION = '3.0'
LAYOUT_VERSION = 'cess.structured-product.v1'
ROOT_GROUPS = ('catalog', 'spectra', 'photometry', 'zero_order', 'lines', 'provenance')
BANDS = ('GU', 'GV', 'GI')
CATALOG_KEYS = ('ID', 'input_row_indices', 'parent_row_indices', 'data_mask', 'input_quality_flags',
                'parameters_desi', 'parameters_grism', 'parameters_phot',
                'zero_order_phot', 'parameters_elg', 'spectral_extraction', 'mge_morphology')
_OLD_SCHEMAS = ('', 'legacy', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 'extracted-mock-spectra-1.1')


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _line_name(name):
    return name.startswith(('intri_el_', 'detect_el_', 'el_diagnostics', 'line_measurements'))


def _copy_attrs(source, target):
    for name in source.attrs:
        attr = source.attrs.get_id(name)
        target.attrs.create(name, source.attrs[name], dtype=attr.dtype, shape=attr.shape)


def _object_id(obj):
    return h5py.h5o.get_info(obj.id).addr


def _references(dtype):
    return (h5py.check_dtype(ref=dtype) is not None or
            any(_references(dtype.fields[name][0]) for name in dtype.names or ()) or
            dtype.subdtype is not None and _references(dtype.subdtype[0]))


def _check_attrs(obj):
    if any(_references(obj.attrs.get_id(name).dtype) for name in obj.attrs):
        raise ValueError('Reference-valued attributes are unsupported: ' + obj.name)


def _check_members(group, seen=None):
    """Fail explicitly on storage requiring external files or reference remapping."""
    seen = set() if seen is None else seen
    if _object_id(group) in seen:
        return
    seen.add(_object_id(group))
    _check_attrs(group)
    for key in group:
        if not isinstance(group.get(key, getlink=True), h5py.HardLink):
            raise ValueError('External/soft links are unsupported: ' + group.name + '/' + key)
        item = group[key]
        _check_attrs(item)
        if isinstance(item, h5py.Group):
            _check_members(item, seen)
        elif item.is_virtual or item.external:
            raise ValueError('External/VDS dataset storage is unsupported: ' + item.name)
        elif _references(item.dtype):
            raise ValueError('Reference-valued datasets are unsupported: ' + item.name)


def _kind_mode(handle):
    if ('parameters_elg' in handle or
            _text(handle.attrs.get('pipeline_stage', '')) == 'emission_line_postprocess'):
        return 'emission_line_postprocess', 'el-detect'
    paired = any(band + '/flux_ujy_with_noise_wce' in handle for band in BANDS)
    return 'simulation', 'paired-wce' if paired else 'standard'


def _plan(handle, unknown):
    if unknown not in ('preserve', 'error'):
        raise ValueError('unknown must be preserve or error')
    if _text(handle.attrs.get('schema_version', '')) != '2.5':
        raise ValueError('Layout conversion requires an explicit schema-2.5 input')
    if 'ID' not in handle or not isinstance(handle['ID'], h5py.Dataset) or handle['ID'].ndim != 1:
        raise ValueError('A one-dimensional ID dataset is required')
    _check_members(handle)
    kind, mode = _kind_mode(handle)
    moves, bands = [], []
    for key in handle:
        if key in CATALOG_KEYS:
            moves.append((key, 'catalog/' + key))
        elif key in ('photometry', 'zero_order'):
            if not isinstance(handle[key], h5py.Group):
                raise ValueError('Expected group: ' + key)
            moves.append((key, key))
        elif key in BANDS:
            if not isinstance(handle[key], h5py.Group):
                raise ValueError('Expected spectral/line band group: ' + key)
            targets = set()
            for child in handle[key]:
                category = 'lines' if _line_name(child) else 'spectra'
                if category == 'spectra' and kind == 'emission_line_postprocess':
                    category = 'provenance/extensions/bands'
                targets.add(category)
                moves.append((key + '/' + child, category + '/' + key + '/' + child))
            if not targets:
                targets.add('lines' if kind == 'emission_line_postprocess' else 'spectra')
            bands.append((key, targets))
        elif key[:3] in ('gv_', 'gi_') and _line_name(key[3:]):
            moves.append((key, 'lines/' + key[:2].upper() + '/' + key[3:]))
        elif unknown == 'error':
            raise ValueError('Unknown root item: /' + key)
        else:
            moves.append((key, 'provenance/extensions/root/' + key))
    destinations = [target for _, target in moves]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Conflicting legacy items map to the same structured path')
    return moves, bands, kind, mode


def _attribute_text(obj, name):
    value = obj.attrs.get(name)
    if not isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise ValueError('Missing/nontext layout attribute: ' + obj.name + '@' + name)
    return _text(value)


def _mapping(handle):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate original path in layout mapping: ' + key)
            result[key] = value
        return result
    try:
        result = json.loads(_attribute_text(handle['provenance/conversion'], 'mapping_json'),
                            object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid structured layout mapping JSON') from error
    if not isinstance(result, dict):
        raise ValueError('Structured layout mapping must be an object')
    return result


def _destination(original, kind):
    """The lossless adapter's declared route, independent of stored mapping."""
    parts = original.split('/')
    if len(parts) == 1:
        key = parts[0]
        if key in CATALOG_KEYS:
            return 'catalog/' + key
        if key in ('photometry', 'zero_order'):
            return key
        if key in BANDS:
            raise ValueError('Band mapping must name its child: ' + original)
        if key[:3] in ('gv_', 'gi_') and _line_name(key[3:]):
            return 'lines/' + key[:2].upper() + '/' + key[3:]
        return 'provenance/extensions/root/' + key
    if len(parts) == 2 and parts[0] in BANDS:
        category = 'lines' if _line_name(parts[1]) else (
            'provenance/extensions/bands' if kind == 'emission_line_postprocess' else 'spectra')
        return category + '/' + original
    raise ValueError('Invalid original layout path: ' + original)


def _same_attrs(first, second):
    if set(first.attrs) != set(second.attrs):
        return False
    for key in first.attrs:
        if first.attrs.get_id(key).dtype != second.attrs.get_id(key).dtype:
            return False
        try:
            np.testing.assert_array_equal(first.attrs[key], second.attrs[key])
        except AssertionError:
            return False
    return True


def validate_structure(handle):
    """Validate the schema-3 layout envelope and lossless path mapping.

    Checks group/header/status consistency, original-root provenance and all
    mapped storage paths without reading dataset arrays. Numerical/schema-2.5
    science contracts remain the manifest verifier's responsibility. Unknown
    extensions are valid when declared by the same original->stored mapping;
    undeclared objects and ambiguous parallel legacy paths are rejected.
    Return plain metadata suitable for manifest decisions, never a repaired file.
    """
    if set(handle) != set(ROOT_GROUPS):
        raise ValueError('Structured product must contain exactly the six root groups; ambiguous legacy paths are forbidden')
    _check_members(handle)
    if any(not isinstance(handle[name], h5py.Group) for name in ROOT_GROUPS):
        raise ValueError('Every structured root member must be a group')
    if (_attribute_text(handle, 'schema_version') != SCHEMA_VERSION or
            _attribute_text(handle, 'layout_version') != LAYOUT_VERSION):
        raise ValueError('Invalid or unknown structured product layout')
    software = _attribute_text(handle, 'software_version')
    kind, mode = _attribute_text(handle, 'product_kind'), _attribute_text(handle, 'mode')
    if (not software or kind not in ('simulation', 'emission_line_postprocess') or
            mode not in (('standard', 'paired-wce') if kind == 'simulation' else ('el-detect',))):
        raise ValueError('Inconsistent structured product kind/mode/software')
    provenance = handle['provenance']
    if _attribute_text(provenance, 'schema_version') != '2.5':
        raise ValueError('Original provenance schema must remain 2.5')
    if 'pipeline_stage' in provenance.attrs and _attribute_text(provenance, 'pipeline_stage') != kind:
        raise ValueError('Product kind disagrees with original pipeline stage')
    if 'conversion' not in provenance or not isinstance(provenance['conversion'], h5py.Group):
        raise ValueError('Missing structured layout conversion metadata')
    conversion = provenance['conversion']
    operation = _attribute_text(conversion, 'operation')
    if (operation not in ('native_layout_adapter', 'legacy_file_conversion') or len(conversion) or
            _attribute_text(conversion, 'source_schema_version') != '2.5' or
            _attribute_text(conversion, 'layout_version') != LAYOUT_VERSION or
            _attribute_text(conversion, 'tool_version') != software):
        raise ValueError('Inconsistent structured conversion metadata')
    if operation == 'legacy_file_conversion':
        source_sha = _attribute_text(conversion, 'source_sha256')
        source_path = _attribute_text(conversion, 'source_path')
        chunk = conversion.attrs.get('chunk_rows')
        if (len(source_sha) != 64 or any(char not in '0123456789abcdef' for char in source_sha) or
                not source_path or not isinstance(chunk, (int, np.integer)) or
                isinstance(chunk, (bool, np.bool_)) or chunk < 1):
            raise ValueError('Invalid legacy conversion source identity/chunk metadata')
    elif any(name in conversion.attrs for name in ('source_sha256', 'source_path', 'chunk_rows')):
        raise ValueError('Native layout adapter must not invent file-conversion identity')
    mapping = _mapping(handle)
    targets = set()
    for original, stored in mapping.items():
        for path in (original, stored):
            if (not isinstance(path, str) or not path or path.startswith('/') or
                    any(part in ('', '.', '..') for part in path.split('/'))):
                raise ValueError('Noncanonical structured layout mapping path')
        if stored != _destination(original, kind):
            raise ValueError('Layout mapping route disagrees with its declared original: ' + original)
        if stored in targets:
            raise ValueError('Conflicting originals map to the same structured path: ' + stored)
        if stored not in handle:
            raise ValueError('Missing mapped structured object: ' + stored)
        targets.add(stored)
    if mapping.get('ID') != 'catalog/ID':
        raise ValueError('Structured mapping must retain the original catalogue ID')
    ids = handle['catalog/ID']
    if not isinstance(ids, h5py.Dataset) or ids.ndim != 1:
        raise ValueError('Structured catalogue ID must be a one-dimensional dataset')
    el = 'parameters_elg' in handle['catalog'] or _text(provenance.attrs.get('pipeline_stage', '')) == 'emission_line_postprocess'
    paired = any('spectra/' + band + '/flux_ujy_with_noise_wce' in handle for band in BANDS)
    if el != (kind == 'emission_line_postprocess') or (kind == 'simulation' and paired != (mode == 'paired-wce')):
        raise ValueError('Stored product data disagrees with kind/mode')
    if kind == 'emission_line_postprocess' and len(handle['spectra']):
        raise ValueError('EL-only layout must not claim a spectral product')
    for name in ROOT_GROUPS:
        group = handle[name]
        expected = ('present' if len(group) or name == 'provenance' else
                    'not_applicable' if kind == 'emission_line_postprocess' or
                    name == 'zero_order' and mode == 'paired-wce' else 'disabled')
        if _attribute_text(group, 'cess_layout_status') != expected:
            raise ValueError('Structured group status/content mismatch: ' + name)
    for band in BANDS:
        parts = [handle[parent + '/' + band] for parent in ('spectra', 'lines', 'provenance/extensions/bands')
                 if parent + '/' + band in handle]
        if any(not isinstance(part, h5py.Group) for part in parts):
            raise ValueError('Structured band must be a group: ' + band)
        if any(not _same_attrs(parts[0], part) for part in parts[1:]):
            raise ValueError('Split band attributes disagree: ' + band)
    scaffolds = set(ROOT_GROUPS) | {'provenance/conversion', 'provenance/extensions',
        'provenance/extensions/root', 'provenance/extensions/bands'}
    scaffolds.update(parent + '/' + band for parent in ('spectra', 'lines', 'provenance/extensions/bands') for band in BANDS)
    def covered(group, prefix=''):
        # h5py.visititems deduplicates hard links; inspect every stored namespace
        # entry so an undeclared alias cannot hide behind an already-seen object.
        for key in group:
            name, value = prefix + key, group[key]
            if name in targets:
                continue  # A mapped group owns its complete preserved subtree.
            if name not in scaffolds:
                raise ValueError('Undeclared object outside structured mapping: ' + name)
            if not isinstance(value, h5py.Group):
                raise ValueError('Structured namespace must be a group: ' + name)
            covered(value, name + '/')
    covered(handle)
    return {'schema_version': SCHEMA_VERSION, 'layout_version': LAYOUT_VERSION,
            'product_kind': kind, 'mode': mode, 'rows': len(ids), 'mapping': mapping,
            'source_schema_version': '2.5', 'operation': operation}


def structure_product(handle, unknown='preserve'):
    """Move a newly completed schema-2.5 HDF5 handle into schema 3.0 in place.

    Call last, inside the caller's atomic writer. Array objects are moved, not
    copied. Original root attrs remain byte/type-preserving provenance attrs.
    No source-file SHA is invented for this native layout-adapter operation.
    """
    if handle.mode == 'r':
        raise ValueError('structure_product needs a writable, newly created product')
    if (_text(handle.attrs.get('schema_version', '')) == SCHEMA_VERSION and
            _text(handle.attrs.get('layout_version', '')) == LAYOUT_VERSION):
        validate_structure(handle)
        return handle
    moves, bands, kind, mode = _plan(handle, unknown)
    staging = '__cess_structure_' + uuid.uuid4().hex
    target = handle.create_group(staging)
    for name in ROOT_GROUPS:
        if name not in ('photometry', 'zero_order') or name not in handle:
            target.create_group(name)
    _copy_attrs(handle, target['provenance'])
    for band, categories in bands:
        for category in categories:
            _copy_attrs(handle[band], target.require_group(category + '/' + band))
    for original, destination in moves:
        parent = destination.rsplit('/', 1)[0] if '/' in destination else ''
        if parent:
            target.require_group(parent)
        handle.move(original, staging + '/' + destination)
    for band, _ in bands:
        del handle[band]
    for name in ROOT_GROUPS:
        group = target[name]
        if 'cess_layout_status' not in group.attrs:
            group.attrs['cess_layout_status'] = ('present' if len(group) or name == 'provenance' else
                'not_applicable' if kind == 'emission_line_postprocess' or name == 'zero_order' and mode == 'paired-wce'
                else 'disabled')
    conversion = target['provenance'].create_group('conversion')
    conversion.attrs['operation'] = 'native_layout_adapter'
    conversion.attrs['source_schema_version'] = '2.5'
    conversion.attrs['tool_version'] = __version__
    conversion.attrs['layout_version'] = LAYOUT_VERSION
    conversion.attrs['mapping_json'] = json.dumps(dict(moves), sort_keys=True)
    for name in list(handle.attrs):
        del handle.attrs[name]
    handle.attrs['schema_version'] = SCHEMA_VERSION
    handle.attrs['layout_version'] = LAYOUT_VERSION
    handle.attrs['software_version'] = __version__
    handle.attrs['product_kind'] = kind
    handle.attrs['mode'] = mode
    for name in ROOT_GROUPS:
        handle.move(staging + '/' + name, name)
    del handle[staging]
    validate_structure(handle)
    return handle


def _copy_tree(source, target, chunk_rows, memo):
    _copy_attrs(source, target)
    memo[_object_id(source)] = target.name
    for name in source:
        item = source[name]
        identity = _object_id(item)
        if identity in memo:
            target[name] = target.file[memo[identity]]
            continue
        if isinstance(item, h5py.Group):
            _copy_tree(item, target.create_group(name), chunk_rows, memo)
            continue
        # Preserve the original dataspace/creation properties exactly, including
        # valid zero-size datasets whose auto chunks exceed their fixed extent.
        # Reconstructing them through create_dataset's high-level chunk check
        # would incorrectly reject an existing valid HDF5 layout.
        copied = h5py.Dataset(h5py.h5d.create(target.id, name.encode('utf-8'),
            item.id.get_type(), item.id.get_space(), dcpl=item.id.get_create_plist()))
        memo[identity] = copied.name
        _copy_attrs(item, copied)
        vlen = h5py.check_dtype(vlen=item.dtype)
        if item.shape is None:
            continue
        if item.ndim == 0:
            copied[()] = item[()]
        elif isinstance(vlen, np.dtype):
            # Numeric ragged rows avoid old h5py's singleton slice broadcasting.
            for index in np.ndindex(item.shape):
                copied[index] = item[index]
        else:
            for first in range(0, item.shape[0], chunk_rows):
                copied[first:first + chunk_rows] = item[first:first + chunk_rows]


def convert_product(source, destination, chunk_rows=256, unknown='preserve'):
    """Copy legacy 2.5 once in bounded row blocks, structure, atomically publish.

    The source is never opened writable. Any validation/copy/source-drift error
    removes only the owned partial file. Existing targets, including dangling
    symlinks and publication-race winners, are never replaced.
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(str(destination)):
        raise FileExistsError(str(destination))
    if source == destination.resolve():
        raise ValueError('Source and destination must differ')
    before = _sha256(source)
    temporary = None
    try:
        with h5py.File(str(source), 'r') as original:
            _plan(original, unknown)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.',
                suffix='.partial', dir=str(destination.parent))
            os.close(descriptor)
            with h5py.File(temporary, 'w') as product:
                _copy_tree(original, product, int(chunk_rows), {})
                structure_product(product, unknown=unknown)
                conversion = product['provenance/conversion']
                conversion.attrs['operation'] = 'legacy_file_conversion'
                conversion.attrs['source_path'] = str(source)
                conversion.attrs['source_sha256'] = before
                conversion.attrs['chunk_rows'] = int(chunk_rows)
                validate_structure(product)
                if _sha256(source) != before:
                    raise ValueError('Source changed during conversion')
                product.flush()
        os.link(temporary, str(destination))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination.resolve()


class _Namespace(Mapping):
    """Read-only group-like alias whose dataset values remain actual h5py objects."""
    def __init__(self, handle, entries, name='/', attrs=None):
        self.file = handle
        self.filename = handle.filename
        self.name = name
        self._entries = entries
        self.attrs = handle.attrs if attrs is None else attrs

    def __getitem__(self, key):
        key = str(key).strip('/')
        if key in ('', '.'):
            return self
        first, _, rest = key.partition('/')
        value = self._entries[first]
        return value[rest] if rest else value

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def visititems(self, function):
        """Visit logical legacy paths, once per actual HDF5 object; no array reads."""
        seen = set()
        def visit(group, prefix):
            for key in sorted(group):
                value = group[key]
                identity = _object_id(value) if isinstance(value, (h5py.Group, h5py.Dataset)) else id(value)
                if identity in seen:
                    continue
                seen.add(identity)
                name = prefix + key
                result = function(name, value)
                if result is not None:
                    return result
                if isinstance(value, (h5py.Group, _Namespace)):
                    result = visit(value, name + '/')
                    if result is not None:
                        return result
        return visit(self, '')


class ProductView:
    """Unified read access; close through open_product's context manager."""
    def __init__(self, handle):
        self.handle = handle
        self.filename = handle.filename
        self.schema_version = _text(handle.attrs.get('schema_version', ''))
        self.structured = self.schema_version == SCHEMA_VERSION
        if self.structured:
            self._layout = validate_structure(handle)
        elif self.schema_version not in _OLD_SCHEMAS:
            raise ValueError('Unsupported product schema: ' + self.schema_version)
        if 'ID' not in self.catalog:
            raise ValueError('Product has no catalogue ID')

    @property
    def catalog(self):
        return self.handle['catalog'] if self.structured else _Namespace(self.handle,
            {key: self.handle[key] for key in CATALOG_KEYS if key in self.handle})

    @property
    def provenance(self):
        return self.handle['provenance'] if self.structured else _Namespace(self.handle, {}, attrs=self.handle.attrs)

    @property
    def attrs(self):
        return self.handle.attrs

    def band(self, band):
        return self.handle['spectra/' + band] if self.structured else self.handle[band]

    def line_group(self, band):
        if self.structured:
            return self.handle['lines/' + band]
        entries = {}
        if band in self.handle:
            entries.update({key: self.handle[band][key] for key in self.handle[band] if _line_name(key)})
        prefix = band.lower() + '_'
        entries.update({key[len(prefix):]: self.handle[key] for key in self.handle
                        if key.startswith(prefix) and _line_name(key[len(prefix):])})
        return _Namespace(self.handle, entries, '/' + band,
                          self.handle[band].attrs if band in self.handle else self.handle.attrs)

    def parameter_table(self, name='parameters_desi', required=()):
        from .parameters import read_parameter_table
        return read_parameter_table(self.catalog[name], ids=self.catalog['ID'], required=required)

    def read_row(self, index, bands=BANDS, fields=('flux_ujy', 'flux_ujy_with_noise', 'ferr', 'snr')):
        count = len(self.catalog['ID'])
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < count:
            raise IndexError('Source row must lie in [0,N)')
        row = {key: self.catalog[key][index] for key in
               ('ID', 'input_row_indices', 'parent_row_indices', 'data_mask',
                'input_quality_flags') if key in self.catalog}
        row['spectra'] = {}
        for band in bands:
            try:
                group = self.band(band)
            except KeyError:
                continue
            row['spectra'][band] = {'wave': group['wave'][:]} if 'wave' in group else {}
            for key in fields:
                if key in group:
                    row['spectra'][band][key] = group[key][index]
        return row

    def legacy_view(self):
        if not self.structured:
            return self.handle
        entries = {key: self.catalog[key] for key in self.catalog}
        for root in ('photometry', 'zero_order'):
            if root in self._layout['mapping']:
                entries[root] = self.handle[root]
        for band in BANDS:
            parts = [self.handle[name + '/' + band] for name in ('spectra', 'lines', 'provenance/extensions/bands')
                     if name + '/' + band in self.handle]
            if parts:
                merged = {}
                for group in parts:
                    for key in group:
                        if key in merged:
                            raise ValueError('Ambiguous split-band item: ' + band + '/' + key)
                        merged[key] = group[key]
                entries[band] = _Namespace(self.handle, merged, '/' + band, parts[0].attrs)
        for original, stored in self._layout['mapping'].items():
            if '/' not in original:
                value = self.handle[stored]
                if original in entries and _object_id(entries[original]) != _object_id(value):
                    raise ValueError('Ambiguous original root item: ' + original)
                entries[original] = value
        # EL-only historically exposed these sparse leaves at the root too.
        if self._layout['product_kind'] == 'emission_line_postprocess':
            for band in ('GV', 'GI'):
                if band in entries:
                    for name in entries[band]:
                        if name.startswith(('intri_el_', 'detect_el_')):
                            entries[band.lower() + '_' + name] = entries[band][name]
        return _Namespace(self.handle, entries, attrs=self.provenance.attrs)


@contextmanager
def open_product(path):
    """Open either a legacy or structured product read-only, never convert it."""
    with h5py.File(str(path), 'r') as handle:
        yield ProductView(handle)


def read_product_summary(path):
    """Metadata-only row/band/variant summary; no full spectral array reads."""
    with open_product(path) as product:
        bands = {}
        for band in BANDS:
            try:
                group = product.band(band)
            except KeyError:
                continue
            bands[band] = dict(pixels=len(group['wave']) if 'wave' in group else None,
                fields=list(group), paired='flux_ujy_with_noise_wce' in group,
                zero_order_proxy='flux_elec_0th' in group)
        return dict(schema_version=product.schema_version,
            layout_version=LAYOUT_VERSION if product.structured else 'legacy',
            rows=len(product.catalog['ID']), bands=bands,
            line_bands=[band for band in BANDS if len(product.line_group(band))]
                if not product.structured else list(product.handle['lines']),
            product_kind=_text(product.attrs.get('product_kind', product.provenance.attrs.get('pipeline_stage', 'unknown'))))
