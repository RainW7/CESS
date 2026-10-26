"""Compare pure result formatting with SHA-bound archived dev4 runner blocks."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import pytest
from cess.io.science_writer import pad_spectral_rows
from cess.pipeline.results import pad_results, emission_flags, desi_parameters

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'tests/baselines'
ARCHIVE = BASELINE / 'results_dev4.tar.gz'
SNAPSHOT_SHA = '00f41dd34c41b0075272d0e0f781bc031f753e49bf737d10944db34711c0342c'
SOURCE_ARCHIVE_SHA = 'd5c892fc5db04f451e41726c0cda04300919fbd55dc1c6edeaade357217deac2'
SOURCE_SHAS = {
    'src/cess/pipeline/standard.py': '21eeb84f88f290a45d673d458fe2036e5ae25e93542c878b94c678dc5c9795c4',
    'src/cess/pipeline/paired.py': 'd1954915fcfb0cd10da2d5c633ea8a00fadebffe53c7c40bbf2f7e17da60afb3',
    'src/cess/io/legacy_writer.py': '8b6788d234071eee01573b109ef7ceaa805104680b84b703b1a9ec745207f526',
}


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def block(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[])),
                   str(ARCHIVE), 'exec')


@pytest.fixture(scope='module', params=['standard', 'paired'])
def reference(request):
    metadata = json.loads((BASELINE / 'PROVENANCE.json').read_text())['results_dev4']
    assert metadata['source_archive'] == 'docs/migration_py312_20261023/checkpoint_20261023_162449.tar.gz'
    assert metadata['source_archive_sha256'] == SOURCE_ARCHIVE_SHA
    assert metadata['source_archive_members'] == 307
    assert metadata['snapshot'] == ARCHIVE.name
    assert metadata['snapshot_sha256'] == SNAPSHOT_SHA
    assert metadata['snapshot_members'] == SOURCE_SHAS
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == SNAPSHOT_SHA
    with tarfile.open(ARCHIVE) as archive:
        members = archive.getmembers()
        assert len(members) == 3 and {item.name for item in members} == set(SOURCE_SHAS)
        sources = {}
        for member in members:
            assert member.isfile()
            source = archive.extractfile(member).read()
            assert hashlib.sha256(source).hexdigest() == SOURCE_SHAS[member.name]
            sources[member.name] = source

        def tree(name):
            return ast.parse(sources[name].decode('utf-8'))
        runner = tree('src/cess/pipeline/' + request.param + '.py')
        writer = tree('src/cess/io/legacy_writer.py')
    original_pad, = [node for node in writer.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'pad_spectral_rows']
    pad_namespace = {'np': np}
    exec(block([original_pad]), pad_namespace)
    padding_loop, = [node for node in ast.walk(runner) if isinstance(node, ast.For)
        and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'pad_spectral_rows' for call in ast.walk(node))]
    el_parent, = [node for node in ast.walk(runner) if isinstance(node, ast.If)
                  and any(assigned(child, 'gv_el_flag') for child in node.body)]
    first = next(i for i, node in enumerate(el_parent.body) if assigned(node, 'gv_el_flag'))
    last = max(i for i, node in enumerate(el_parent.body) if assigned(node, 'gi_el_flag'))
    stack, = [node for node in ast.walk(runner) if assigned(node, 'parameters_desi_original')]
    cast, = [node for node in ast.walk(runner) if isinstance(node, ast.Try)
             and any(assigned(child, 'parameters_desi') for child in node.body)]
    names, = [ast.literal_eval(node.value) for node in ast.walk(runner)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name) and target.value.value.id == 'dataset2'
            and target.value.attr == 'attrs' for target in node.targets)]
    return dict(padding=block([padding_loop]), flags=block(el_parent.body[first:last+1]),
                table=block([stack, cast]), names=names, pad=pad_namespace['pad_spectral_rows'])


@pytest.mark.parametrize('rows', [[], [np.nan], [np.array([0., 1., -1., np.nan, np.inf, -np.inf])],
                                [np.array([1.]), np.nan, np.array([])]])
def test_padding_and_masks_match_literal_archive_blocks(reference, rows):
    count = len(rows)
    waves = dict(gu=np.arange(6), gv=np.arange(7), gi=np.arange(8))
    data = {key: deepcopy(rows) for key in ('gu_ferr', 'gu0_spec_mask', 'gv_snr_mask',
                                            'gi_ujy_with_noise_wce')}
    namespace = dict(np=np, data_dict=data, padded_data_dict={},
        f={'ID': np.arange(count)}, guwave=waves['gu'], gvwave=waves['gv'], giwave=waves['gi'],
        pad_spectral_rows=reference['pad'])
    exec(reference['padding'], namespace)
    actual = pad_results(data, count, waves, pad_spectral_rows)
    assert list(actual) == list(namespace['padded_data_dict'])
    for name, expected in namespace['padded_data_dict'].items():
        np.testing.assert_array_equal(actual[name], expected, strict=True)


@pytest.mark.parametrize('count,truth,matched', [
    (0, [], []), (1, [0], [0]), (1, [0], []), (1, [], [0]),
    (4, [1, 3, 3], [0, 1, 8]), (4, [-1, 0, 9], [0, 1, 9])])
def test_flags_match_literal_archive_for_empty_single_and_unmatched_rows(reference, count, truth, matched):
    intrinsic = np.asarray(truth, dtype=np.int64)
    detected = np.asarray(matched, dtype=np.int64)
    namespace = dict(np=np, f={'ID': np.arange(count)},
        gv_noisy_lines=[None]*count, gi_noisy_lines=[None]*count,
        gv_intri_el={'id': intrinsic}, gv_detect_el={'id': detected},
        gi_intri_el={'id': detected}, gi_detect_el={'id': intrinsic})
    exec(reference['flags'], namespace)
    np.testing.assert_array_equal(emission_flags(count, intrinsic, detected),
                                  namespace['gv_el_flag'], strict=True)
    np.testing.assert_array_equal(emission_flags(count, detected, intrinsic),
                                  namespace['gi_el_flag'], strict=True)


@pytest.mark.parametrize('count,objects', [(0, False), (1, False), (3, True)])
def test_eleven_columns_and_names_match_archive_hstack_then_cast(reference, count, objects):
    columns = [np.arange(count, dtype=np.int64) + index*7 for index in range(11)]
    if objects:
        columns = [value.astype(str).astype(object) for value in columns]
        columns[6][1] = np.nan
        columns[10][2] = np.inf
    for index in range(6, 10):
        columns[index] = columns[index].reshape(count, 1)
    fields = ('raall', 'decall', 'zall', 'maggall', 'magrall', 'magzall',
              'ntotal', 'retotal', 'patotal', 'artotal', 'm_starall')
    namespace = dict(zip(fields, columns))
    namespace.update(np=np, f={'ID': np.arange(count)})
    exec(reference['table'], namespace)
    values, names = desi_parameters(count, *columns)
    np.testing.assert_array_equal(values, namespace['parameters_desi'], strict=True)
    assert values.shape == (count, 11) and values.dtype == np.dtype('float64')
    assert names == reference['names']


def test_padding_uses_the_supplied_callable_in_key_order():
    calls = []
    def supplied(rows, count, width, name):
        calls.append((rows, count, width, name))
        return np.array([[1., np.nan]])
    first, second = object(), object()
    actual = pad_results({'gi_spec_mask': first, 'gu_flux': second}, 1,
                         {'gi': [1, 2], 'gu': [3, 4]}, supplied)
    assert calls == [(first, 1, 2, 'gi_spec_mask'), (second, 1, 2, 'gu_flux')]
    np.testing.assert_array_equal(actual['padded_gi_spec_mask'], [[True, False]])
    np.testing.assert_array_equal(actual['padded_gu_flux'], [[1., np.nan]])
