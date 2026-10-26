"""Exact collection oracle from the SHA-bound dev6 runners, without their engines."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from cess.io.legacy_writer import pad_spectral_rows
from cess.pipeline import collection as collection_module
from cess.pipeline.collection import SpectralResults
from cess.pipeline.results import pad_results


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'tests/baselines'
SNAPSHOT_SHA = 'b6c38ac10a62cb46ec4d8fcabf6e9cad187638a38e0d9ea485967387afc511c0'
SOURCE_ARCHIVE_SHA = '8b6a244d73eae63b15e44ea0a5fb34a161689c3fa8b8c13f80fc0d243f1303e0'
SOURCE_SHAS = {
    'standard': '5a4b4ef2038d0008e8d9b48ce6a50163c9cb71da88f17a155f30aec42e5381a8',
    'paired': '84e36d79ed5abb95a05b150194c10414994e30c31f7f5944b1780c5f1f4158ad',
}
BANDS = ('GU', 'GV', 'GI')
BASE_FIELDS = ('elec_per_column', 'fnu_with_noise', 'efnu', 'rms_in_e',
               'snr', 'snr_mean', 'spec_mask', 'snr_mask')
WCE_FIELDS = ('flux_wce', 'elec_per_column_wce', 'fnu_with_noise_wce', 'efnu_wce', 'snr_wce')


def assigned_name(node):
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)):
        return node.targets[0].id


def block(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[])),
                   str(BASELINE / 'collection_dev6.tar.gz'), 'exec')


def select_blocks(nodes, predicate):
    """Preserve archived flag guards; discard only unrelated statements."""
    selected = []
    for node in nodes:
        if isinstance(node, ast.If):
            body = select_blocks(node.body, predicate)
            orelse = select_blocks(node.orelse, predicate)
            if body or orelse:
                replacement = deepcopy(node)
                replacement.body = body or [ast.Pass()]
                replacement.orelse = orelse
                selected.append(replacement)
        elif predicate(node):
            selected.append(node)
    return selected


@pytest.fixture(scope='module', params=['standard', 'paired'])
def reference(request):
    snapshot = BASELINE / 'collection_dev6.tar.gz'
    metadata = json.loads((BASELINE / 'PROVENANCE.json').read_text())['collection_dev6']
    assert metadata['source_archive'] == 'docs/refactor_collection_20261024/baseline_dev6_package.tar.gz'
    assert metadata['source_archive_sha256'] == SOURCE_ARCHIVE_SHA
    assert metadata['source_archive_members'] == 105
    assert metadata['snapshot'] == snapshot.name
    assert metadata['snapshot_sha256'] == SNAPSHOT_SHA
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == SNAPSHOT_SHA
    expected_members = {'src/cess/pipeline/' + name + '.py': sha
                        for name, sha in SOURCE_SHAS.items()}
    assert metadata['snapshot_members'] == expected_members
    with tarfile.open(snapshot) as archive:
        members = archive.getmembers()
        assert len(members) == 2 and {item.name for item in members} == set(expected_members)
        for member in members:
            assert member.isfile()
            source = archive.extractfile(member).read()
            assert hashlib.sha256(source).hexdigest() == expected_members[member.name]
            if member.name.endswith('/' + request.param + '.py'):
                tree = ast.parse(source.decode('utf-8'))
    function, = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'simulate_file']
    scope, = [node for node in function.body if isinstance(node, ast.With)]
    row_loop, = [node for node in scope.body if isinstance(node, ast.For)
                 and any(isinstance(child, ast.If) and isinstance(child.test, ast.Subscript)
                         and isinstance(child.test.value, ast.Name)
                         and child.test.value.id == 'mask' for child in node.body)]
    before = scope.body[:scope.body.index(row_loop)]
    after = scope.body[scope.body.index(row_loop) + 1:]
    names = {assigned_name(node) for statement in before for node in ast.walk(statement)
             if assigned_name(node) and assigned_name(node).startswith(('gu_', 'gv_', 'gi_',
                                                                        'gu0_', 'gv0_', 'gi0_'))
             and isinstance(node.value, ast.List) and not node.value.elts}
    assert len(names) == (42 if request.param == 'paired' else 54)
    initialization = select_blocks(before, lambda node: assigned_name(node) in names)
    decision, = [node for node in row_loop.body if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Subscript)
                  and isinstance(node.test.value, ast.Name) and node.test.value.id == 'mask']

    def is_append(node):
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'append'
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id in names)

    valid = select_blocks(decision.body, is_append)
    invalid = select_blocks(decision.orelse[0].body, is_append)
    pad_index = next(index for index, node in enumerate(after)
                     if assigned_name(node) == 'padded_data_dict')
    conversions = [node for node in after[:pad_index]
                   if assigned_name(node) == 'data_dict' or any(
                       isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                       and isinstance(call.func.value, ast.Name) and call.func.value.id == 'np'
                       and call.func.attr == 'array' and call.args
                       and isinstance(call.args[0], ast.Name) and call.args[0].id in names
                       for call in ast.walk(node))]
    offset_block = after[pad_index + 1]
    assert isinstance(offset_block, ast.If)
    offset_names = [assigned_name(node) for node in offset_block.body]
    assert all(name in names and name.endswith('_wave_off') for name in offset_names)
    stats = [assigned_name(node) for node in conversions
             if assigned_name(node) and assigned_name(node).endswith(('_rms_in_e', '_snr_mean'))]
    assert len(stats) == (6 if request.param == 'paired' else 12)
    return dict(paired=request.param == 'paired', initialize=block(initialization),
                valid=block(valid), invalid=block(invalid), spectra=block(conversions),
                padding=block([after[pad_index]]), offsets=block([offset_block]),
                stats=stats, offset_names=offset_names)


def make_waves():
    # Input mapping order must not set the legacy GU/GV/GI output order.
    return {'GI': np.arange(3.), 'GV': np.arange(5.), 'GU': np.arange(7.)}


def payloads(waves, paired, row, dtype, *, short=False, special=False, numeric_masks=False):
    result = {}
    for band_index, band in enumerate(BANDS):
        for variant_index, variant in enumerate(('', '_wce') if paired else ('', '_0th')):
            fields = WCE_FIELDS if variant == '_wce' else BASE_FIELDS
            values = []
            width = len(waves[band]) - (2 if short else 0)
            for field_index, field in enumerate(fields):
                base = 100 * row + 20 * band_index + 10 * variant_index + field_index + 1
                if field in ('rms_in_e', 'snr_mean'):
                    value = dtype(base / 7.)
                else:
                    value = (np.arange(width, dtype=dtype) + base).astype(dtype)
                    if field.endswith('_mask') and not numeric_masks:
                        value = (value.astype(int) % 2 == 0)
                    elif special:
                        value[0] = np.nan
                        if width > 1:
                            value[-1] = np.inf if field_index % 2 else -np.inf
                    if 'fnu' in field or field == 'flux_wce':
                        value = value * dtype(1.e-29)
                values.append(value)
            offset = dtype(base / 13.)
            result[band, variant] = (tuple(values), offset)
    return result


def old_single_values(payload):
    namespace = {}
    for (band, variant), (values, offset) in payload.items():
        prefix = band.lower() + ('0' if variant == '_0th' else '')
        fields = WCE_FIELDS if variant == '_wce' else BASE_FIELDS
        for field, value in zip(fields, values):
            old_field = 'fnu_error' if field == 'efnu' else field
            namespace[prefix + '_' + old_field + '_single'] = value
        if variant != '_wce':
            namespace[prefix + '_wave_off_single'] = offset
    return namespace


def oracle_namespace(reference, waves, count, wave_error, *, numpy=np, pad=pad_spectral_rows):
    namespace = dict(np=numpy, f={'ID': np.arange(count)}, pad_results=pad_results,
                     pad_spectral_rows=pad,
                     context=SimpleNamespace(options={
                         'apply_wavelength_calibration_error': wave_error}))
    namespace.update({band.lower() + 'wave': value for band, value in waves.items()})
    exec(reference['initialize'], namespace)
    return namespace


def append_both(reference, namespace, actual, payload):
    namespace.update(old_single_values(payload))
    exec(reference['valid'], namespace)
    for (band, variant), (values, offset) in payload.items():
        actual.append(band, variant, values, wave_offset=offset)


def assert_mapping_exact(actual, expected):
    assert list(actual) == list(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], strict=True, err_msg=key)


@pytest.mark.parametrize('wave_error', [False, True])
@pytest.mark.parametrize('dtype,valid,short,special,numeric_masks', [
    (np.float64, (), False, False, False),
    (np.float32, (True,), False, False, False),
    (np.float64, (False,), False, False, False),
    (np.float64, (True, False, True), False, True, False),
    (np.float32, (True, True), True, False, False),
    (np.float64, (True,), False, True, True),
])
def test_spectra_statistics_offsets_and_padding_match_dev6(
        reference, wave_error, dtype, valid, short, special, numeric_masks):
    waves, count = make_waves(), len(valid)
    namespace = oracle_namespace(reference, waves, count, wave_error)
    actual = SpectralResults(waves, paired=reference['paired'], wave_error=wave_error)
    for row, is_valid in enumerate(valid):
        if is_valid:
            append_both(reference, namespace, actual,
                        payloads(waves, reference['paired'], row, dtype,
                                 short=short, special=special, numeric_masks=numeric_masks))
        else:
            exec(reference['invalid'], namespace)
            actual.append_invalid()
    exec(reference['spectra'], namespace)
    data, stats = actual.spectra(count)
    assert_mapping_exact(data, namespace['data_dict'])
    assert_mapping_exact(stats, {key: namespace[key] for key in reference['stats']})
    assert all(value.shape == (count, 1) for value in stats.values())
    # Invoke the exact old pad_results call, then the explicit new call with the same hook.
    old_calls, new_calls = [], []

    def recording_pad(calls, source):
        def supplied(rows, source_count, width, label):
            assert rows is source[label]
            calls.append((source_count, width, label))
            return pad_spectral_rows(rows, source_count, width, label)
        return supplied

    namespace['pad_spectral_rows'] = recording_pad(old_calls, namespace['data_dict'])
    exec(reference['padding'], namespace)
    padded = pad_results(data, count, {band.lower(): wave for band, wave in waves.items()},
                         recording_pad(new_calls, data))
    assert new_calls == old_calls
    assert [call[2] for call in new_calls] == list(data)
    assert_mapping_exact(padded, namespace['padded_data_dict'])
    for key, value in padded.items():
        assert value.shape == (count, len(waves[key[7:9].upper()]))
        assert value.dtype == np.dtype(bool if key.endswith(('_spec_mask', '_snr_mask')) else float)
    exec(reference['offsets'], namespace)
    offsets = actual.offsets(count)
    expected_offsets = {key: namespace[key] for key in reference['offset_names']} if wave_error else {}
    assert_mapping_exact(offsets, expected_offsets)
    assert all(value.shape == (count, 1) for value in offsets.values())
    if reference['paired']:
        assert not any(key.endswith(('_spec_mask_wce', '_snr_mask_wce')) for key in data)
        assert not any('_wce' in key for key in stats)


class ArrayRecorder:
    """Record np.array inputs without altering dtype inference or the input objects."""
    def __init__(self, labels, events):
        self.labels, self.events = labels, events

    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, values, *args, **kwargs):
        if isinstance(values, list) and values:
            self.events.append(('array', self.labels[id(values[0])]))
        return np.array(values, *args, **kwargs)


@pytest.mark.parametrize('wave_error', [False, True])
def test_array_conversion_order_and_offsets_follow_padding(reference, wave_error, monkeypatch):
    waves = make_waves()
    payload = payloads(waves, reference['paired'], 0, np.float64)
    singles = old_single_values(payload)
    labels = {id(value): key for key, value in singles.items()}
    old_events, new_events = [], []

    def hook(events):
        def supplied(rows, source_count, width, label):
            events.append(('pad', label))
            return pad_spectral_rows(rows, source_count, width, label)
        return supplied

    namespace = oracle_namespace(reference, waves, 1, wave_error,
                                 numpy=ArrayRecorder(labels, old_events), pad=hook(old_events))
    namespace.update(singles)
    exec(reference['valid'], namespace)
    exec(reference['spectra'], namespace)
    exec(reference['padding'], namespace)
    exec(reference['offsets'], namespace)
    monkeypatch.setattr(collection_module, 'np', ArrayRecorder(labels, new_events))
    actual = SpectralResults(waves, paired=reference['paired'], wave_error=wave_error)
    for (band, variant), (values, offset) in payload.items():
        actual.append(band, variant, values, wave_offset=offset)
    data, _ = actual.spectra(1)
    pad_results(data, 1, {band.lower(): wave for band, wave in waves.items()}, hook(new_events))
    actual.offsets(1)
    assert new_events == old_events
    if wave_error:
        last_pad = max(index for index, event in enumerate(new_events) if event[0] == 'pad')
        assert all(event[0] == 'array' and event[1].endswith('_wave_off_single')
                   for event in new_events[last_pad + 1:])


@pytest.mark.parametrize('paired', [False, True])
def test_collection_leaves_random_streams_untouched(paired):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    waves = make_waves()
    result = SpectralResults(waves, paired=paired, wave_error=True)
    for (band, variant), (values, offset) in payloads(waves, paired, 0, np.float32).items():
        result.append(band, variant, values, wave_offset=offset)
    result.append_invalid()
    data, _ = result.spectra(2)
    pad_results(data, 2, {band.lower(): wave for band, wave in waves.items()}, pad_spectral_rows)
    result.offsets(2)
    for before, after in zip(numpy_state, np.random.get_state()):
        np.testing.assert_array_equal(before, after)
    assert python_state == random.getstate()
