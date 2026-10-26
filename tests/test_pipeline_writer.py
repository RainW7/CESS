"""Compare grism columns and base spectra against SHA-bound dev5 runner ASTs."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tarfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from cess.pipeline.common import write_base_spectra
from cess.pipeline.results import grism_parameters


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'tests/baselines'
SNAPSHOT_SHA = '256c75359166cb1fc6462895a4beb7e51365fec3e6a71aa8ad85388539914f1a'
SOURCE_ARCHIVE_SHA = 'b39d57c589c21237e16e2a5ceda98dd74d6b5470b59ec53a5f199b4ae4ba330a'
SOURCE_SHAS = {
    'standard': '94d6dcdd8e1e76ca87cdf5593d89f172a878fd598c2bbecd25484542609635b0',
    'paired': '6ce0a8fa4e4caa7827ed2fef1ffbd3f123325a185162c480ebc22ab00c1025a6',
}
BANDS = ('GU', 'GV', 'GI')


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def block(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[])),
                   str(BASELINE / 'writer_dev5.tar.gz'), 'exec')


@pytest.fixture(scope='module', params=['standard', 'paired'])
def reference(request):
    snapshot = BASELINE / 'writer_dev5.tar.gz'
    metadata = json.loads((BASELINE / 'PROVENANCE.json').read_text())['writer_dev5']
    assert metadata['source_archive'] == 'docs/refactor_results_20261023/checkpoint_20261023_173457.tar.gz'
    assert metadata['source_archive_sha256'] == SOURCE_ARCHIVE_SHA
    assert metadata['source_archive_members'] == 304
    assert metadata['snapshot'] == snapshot.name
    assert metadata['snapshot_sha256'] == SNAPSHOT_SHA
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == SNAPSHOT_SHA
    assert metadata['snapshot_members'] == {
        'src/cess/pipeline/' + name + '.py': sha for name, sha in SOURCE_SHAS.items()}
    with tarfile.open(snapshot) as archive:
        members = archive.getmembers()
        assert len(members) == 2 and {item.name for item in members} == {
            'src/cess/pipeline/' + name + '.py' for name in SOURCE_SHAS}
        for name, expected_sha in SOURCE_SHAS.items():
            path = 'src/cess/pipeline/' + name + '.py'
            member, = [item for item in archive.getmembers() if item.name == path]
            assert member.isfile()
            source = archive.extractfile(member).read()
            assert hashlib.sha256(source).hexdigest() == expected_sha
            assert metadata['snapshot_members'][path] == expected_sha
            if name == request.param:
                tree = ast.parse(source.decode('utf-8'))
    table = min((node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and any(assigned(child, 'parameters_grism') for child in node.body)),
                key=lambda node: node.lineno)
    names = min((node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and any(assigned(child, 'dataset99') for child in node.body)),
                key=lambda node: node.lineno)
    atomic, = [node for node in ast.walk(tree) if isinstance(node, ast.With)
               and any(isinstance(item.context_expr, ast.Call)
                       and isinstance(item.context_expr.func, ast.Name)
                       and item.context_expr.func.id == 'atomic_hdf5' for item in node.items)]
    first = next(i for i, node in enumerate(atomic.body) if assigned(node, 'group_gu'))
    last = next(i for i in range(first, len(atomic.body)) if isinstance(atomic.body[i], ast.If))
    return dict(kind=request.param, table=block([table]), names=block([names]),
                writer=block(atomic.body[first:last]))


class AttributesOnlyHandle:
    def create_dataset(self, name, **kwargs):
        self.dataset = SimpleNamespace(attrs={})
        return self.dataset


@pytest.mark.parametrize('el_detect,wave_error', [(False, False), (False, True),
                                               (True, False), (True, True)])
@pytest.mark.parametrize('count,dtype', [(0, np.float64), (1, np.float32),
                                       (3, np.float64), (2, np.int16)])
def test_grism_values_names_dtype_and_axes_match_dev5(reference, el_detect, wave_error, count, dtype):
    columns = {}
    for index, name in enumerate(prefix + '_' + field
            for field in ('rms_in_e', 'snr_mean', 'wave_off')
            for prefix in ('gu', 'gv', 'gi', 'gu0', 'gv0', 'gi0')):
        value = (np.arange(count) + 10 * index).astype(dtype).reshape(count, 1)
        if count == 3:
            value[0, 0], value[2, 0] = np.nan, np.inf
        columns[name] = value
    flags = (np.arange(count, dtype=float) % 3, np.arange(count, dtype=float)[::-1] % 3)
    namespace = dict(columns, np=np, f={'ID': np.arange(count)},
        context=SimpleNamespace(options=dict(apply_el_detect=el_detect,
                                            apply_wavelength_calibration_error=wave_error)),
        gv_el_flag=flags[0], gi_el_flag=flags[1], file=AttributesOnlyHandle())
    exec(reference['table'], namespace)
    exec(reference['names'], namespace)
    group = lambda field, zero=False: tuple(columns[band + ('0' if zero else '') + '_' + field]
                                             for band in ('gu', 'gv', 'gi'))
    standard = reference['kind'] == 'standard'
    actual, names = grism_parameters(count, group('rms_in_e'), group('snr_mean'),
        wave_offsets=group('wave_off') if wave_error else None,
        el_flags=flags if el_detect else None,
        zero_rms=group('rms_in_e', True) if standard else None,
        zero_mean_snr=group('snr_mean', True) if standard else None,
        zero_wave_offsets=group('wave_off', True) if standard and wave_error else None)
    np.testing.assert_array_equal(actual, namespace['parameters_grism'], strict=True)
    assert names == namespace['file'].dataset.attrs['name']
    expected_width = ((14 if el_detect else 6) + (6 if el_detect else 3) * wave_error
                      if standard else 6 + 2 * el_detect + 3 * wave_error)
    assert actual.shape == (count, expected_width)
    assert len(names) == expected_width


class RecordingGroup:
    def __init__(self, group, calls):
        self.group, self.calls = group, calls

    def create_group(self, name):
        group = self.group.create_group(name)
        self.calls.append(('group', group.name))
        return RecordingGroup(group, self.calls)

    def create_dataset(self, name, **kwargs):
        dataset = self.group.create_dataset(name, **kwargs)
        self.calls.append(('dataset', dataset.name, tuple(kwargs), id(kwargs['data'])))
        return dataset


def spectral_inputs(count, dtype):
    # Deliberately reverse mapping order: legacy output order is always GU/GV/GI.
    waves = {band: np.arange(width, dtype=float) for band, width in zip(BANDS[::-1], (5, 3, 2))}
    incident, padded = {}, {}
    for index, (band, wave) in enumerate(waves.items()):
        values = np.arange(count * len(wave), dtype=dtype).reshape(count, len(wave)) + index
        if count > 1:
            values[0, 0], values[-1, -1] = np.nan, np.inf
        incident[band] = values
        for prefix in (band.lower(), band.lower() + '0'):
            for field in ('elec_per_column', 'ujy_with_noise', 'ferr', 'snr', 'spec_mask', 'snr_mask'):
                value = values > 0 if field.endswith('_mask') else values.copy()
                padded['padded_' + prefix + '_' + field] = value
                if prefix == band.lower():
                    padded['padded_' + prefix + '_' + field + '_wce'] = value.copy()
        padded['padded_' + band.lower() + '_ujy_wce'] = values.copy() + 0.5
    return waves, incident, padded


@pytest.mark.parametrize('count,dtype', [(0, np.float64), (1, np.float32), (3, np.float64)])
def test_base_writer_calls_datasets_and_variant_boundaries_match_dev5(reference, tmp_path, count, dtype):
    waves, incident, padded = spectral_inputs(count, dtype)
    before = {name: value.copy() for name, value in padded.items()}
    expected_calls, actual_calls = [], []
    with h5py.File(tmp_path / 'old.hdf5', 'w') as old, h5py.File(tmp_path / 'new.hdf5', 'w') as new:
        namespace = dict(file=RecordingGroup(old, expected_calls), padded_data_dict=padded)
        for band in BANDS:
            namespace[band.lower() + 'wave'] = waves[band]
            namespace[band.lower() + '_ujy'] = incident[band]
        exec(reference['writer'], namespace)
        groups = write_base_spectra(RecordingGroup(new, actual_calls), waves, incident, padded,
                                    paired=reference['kind'] == 'paired')
        assert tuple(groups) == BANDS
        assert actual_calls == expected_calls
        for band in BANDS:
            assert set(new[band]) == set(old[band])
            for name in old[band]:
                expected, actual = old[band][name], new[band][name]
                assert actual.dtype.metadata == expected.dtype.metadata
                np.testing.assert_array_equal(actual[:], expected[:], strict=True)
                assert dict(actual.attrs) == dict(expected.attrs)
            if reference['kind'] == 'paired':
                assert 'flux_ujy_wce' in new[band]
                assert 'spec_mask_wce' not in new[band] and 'snr_mask_wce' not in new[band]
            else:
                assert 'flux_ujy_0th' not in new[band]
                assert new[band]['spec_mask_0th'].dtype == np.dtype(bool)
    for name, value in before.items():
        np.testing.assert_array_equal(padded[name], value, strict=True)


def test_new_formatters_leave_random_streams_untouched(tmp_path):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    columns = tuple(np.ones((1, 1)) for _ in BANDS)
    grism_parameters(1, columns, columns, el_flags=(np.array([1.]), np.array([2.])))
    waves, incident, padded = spectral_inputs(1, np.float64)
    with h5py.File(tmp_path / 'random.hdf5', 'w') as handle:
        write_base_spectra(handle, waves, incident, padded, paired=False)
    for before, after in zip(numpy_state, np.random.get_state()):
        np.testing.assert_array_equal(before, after)
    assert python_state == random.getstate()
