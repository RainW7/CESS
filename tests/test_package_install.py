"""Source-package contracts plus an offline wheel/temporary-target install.

The supported Python 3.12 environment must actually build and install the wheel
with the declared modern build toolchain; unsupported interpreters skip only
these wheel cases. No dependency installation or network access occurs: the
only installed artifact is our wheel in a temporary target.
"""
import configparser
from email import message_from_bytes
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import zipfile
import pytest
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from cess import __version__ as PACKAGE_VERSION

ENTRYPOINTS = {
    'cess': 'cess.cli.main:main',
    'cess-profile-library': 'cess.morphology.profile_generator:main',
    'cess-mge-library': 'cess.morphology.generator:main',
    'cess-extract': 'cess.io.sampling:cli_main',
    'cess-viewer': 'cess.viewer:main',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def process(arguments, cwd, pythonpath):
    env = dict(os.environ, PYTHONPATH=str(pythonpath), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.pop('CESS_CONFIG', None)
    result = subprocess.run(arguments, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_light_package_imports_do_not_load_resources_or_science(tmp_path):
    code = '''
import builtins
import importlib.abc
import pathlib
import sys
class BlockScience(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'numpy','scipy','h5py','astropy','galsim','pandas','matplotlib','photutils','specutils','tqdm'}:
   raise AssertionError('eager science import: '+fullname)
sys.meta_path.insert(0,BlockScience())
def forbidden(*args,**kwargs):raise AssertionError('resource/source content read at import')
builtins.open=forbidden
pathlib.Path.open=forbidden
import cess
import cess.__main__
import cess.resources
import cess.provenance
import cess.config.loader
import cess.morphology.profile_generator
import cess.morphology.generator
assert cess.__version__==EXPECTED_VERSION
assert not any(name in sys.modules for name in ['numpy','scipy','galsim','h5py'])
print('light imports passed')
'''
    code = code.replace('EXPECTED_VERSION', repr(PACKAGE_VERSION))
    assert 'light imports passed' in process([sys.executable, '-s', '-B', '-c', code], tmp_path, SRC)


def test_exact_packaged_default_resource_closure():
    from cess.resources import DATA_ROOT, CALIBRATION_PATHS, calibration_files, default_config_path
    assert len(calibration_files()) == len(set(CALIBRATION_PATHS)) == 17
    expected = set(CALIBRATION_PATHS) | {'emulator_parameters.json'}
    assert {str(path.relative_to(DATA_ROOT)) for path in DATA_ROOT.rglob('*') if path.is_file()} == expected
    inventory = json.loads((ROOT / 'docs/RESOURCE_MANIFEST.json').read_text())['files']
    assert set(inventory) == expected
    for name in sorted(expected):
        assert digest(DATA_ROOT / name) == inventory[name]['sha256']
        assert (DATA_ROOT / name).stat().st_size == inventory[name]['bytes']


def test_direct_requirement_profiles_match_package_metadata():
    from packaging.requirements import Requirement
    config = configparser.ConfigParser()
    config.read(ROOT / 'setup.cfg')
    assert config['options']['python_requires'] == '>=3.12,<3.13'
    def requirements(lines):
        return {(item.name.lower(), str(item.marker) if item.marker else ''): str(item.specifier) for item in
                (Requirement(line.strip()) for line in lines if line.strip()
                 and not line.lstrip().startswith(('#', '-r ')))}
    expected = requirements(config['options']['install_requires'].splitlines())
    assert requirements((ROOT / 'requirements/core-py312.txt').read_text().splitlines()) == expected
    names = {name for name, marker in expected}
    assert 'photutils' not in names and 'matplotlib' not in names and 'tqdm' in names
    for extra in ('tools', 'plot', 'viewer', 'render'):
        assert requirements((ROOT / ('requirements/' + extra + '-py312.txt')).read_text().splitlines()) == \
               requirements(config['options.extras_require'][extra].splitlines())
    assert requirements((ROOT / 'requirements/test-py312.txt').read_text().splitlines()) == {('pytest', ''): '==9.1.1'}
    assert set(config['options.entry_points']['console_scripts'].strip().splitlines()) == {
        name + ' = ' + value for name, value in ENTRYPOINTS.items()}


def test_gravity_profile_matches_linux_package_requirements():
    from packaging.requirements import Requirement
    config = configparser.ConfigParser()
    config.read(ROOT / 'setup.cfg')
    declared = [Requirement(line.strip()) for line in
                config['options']['install_requires'].splitlines() if line.strip()]
    selected = [item for item in declared if item.marker is None or
                item.marker.evaluate({'sys_platform': 'linux', 'extra': ''})]
    profile = [Requirement(line.strip()) for line in
               (ROOT / 'requirements/gravity-glibc217-py312.txt').read_text().splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
    assert len(selected) == len({item.name for item in selected}) == 6
    assert {(item.name, str(item.specifier)) for item in selected} == {
        (item.name, str(item.specifier)) for item in profile}


@pytest.mark.parametrize('name', ['', '.', '../private.fits', '/tmp/private.fits', 'a/../../private', 'a\\b'])
def test_resource_path_rejects_nonrelative_names(name):
    from cess.resources import resource_path
    with pytest.raises(ValueError):
        resource_path(name)


def test_default_config_and_bare_mapping_resolve_packaged_resources(monkeypatch, tmp_path):
    from cess.config.loader import load_config, validate_config, DEFAULT_CONFIG_PATH
    from cess.resources import DATA_ROOT, calibration_files
    monkeypatch.delenv('CESS_CONFIG', raising=False)
    monkeypatch.chdir(tmp_path)
    actual = load_config()
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text())
    bare = validate_config(raw)
    expected = set(calibration_files())
    assert DEFAULT_CONFIG_PATH == DATA_ROOT / 'emulator_parameters.json'
    assert {Path(value) for key, value in actual.items() if key.endswith(('_path', '_pkl'))
            and Path(value).is_file() and key != '_config_path'} == expected
    assert bare['gutp_path'] == actual['gutp_path']
    assert actual['_config_dir'] == str(DATA_ROOT)
    assert actual['_config_path'] == str(DEFAULT_CONFIG_PATH)
    assert bare['_config_path'] is None
    for name in ('input_path', 'output_path', 'el_input_path', 'el_output_path'):
        assert actual[name] == str(tmp_path / raw[name])
        assert load_config(DEFAULT_CONFIG_PATH)[name] == actual[name]
        assert not Path(actual[name]).exists()


def test_explicit_and_environment_configs_keep_their_own_relative_base(tmp_path, monkeypatch):
    from cess.config.loader import DEFAULT_CONFIG_PATH, load_config
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text())
    raw['gems_path'] = 'custom/gems.fits'
    folder = tmp_path / 'user'
    folder.mkdir()
    path = folder / 'parameters.json'
    path.write_text(json.dumps(raw))
    monkeypatch.setenv('CESS_CONFIG', str(path))
    assert load_config()['gems_path'] == str(folder / 'custom/gems.fits')
    for name in ('input_path', 'output_path', 'el_input_path', 'el_output_path'):
        assert load_config()[name] == str(folder / raw[name])
    assert load_config(DEFAULT_CONFIG_PATH)['gems_path'] != str(folder / 'custom/gems.fits')


def test_recursive_provenance_is_complete_and_fresh(tmp_path, monkeypatch):
    from cess import provenance
    package = tmp_path / 'cess'
    (package / 'nested').mkdir(parents=True)
    first, second = package / '__init__.py', package / 'nested/module.py'
    first.write_text('A=1\n')
    second.write_text('B=1\n')
    (package / 'private.fits').write_bytes(b'not a Python source')
    monkeypatch.setattr(provenance, 'PACKAGE_ROOT', package)
    assert provenance.source_files() == tuple(sorted((first, second)))
    before = provenance.code_sha256()
    second.write_text('B=2\n')
    assert provenance.code_sha256()[str(second)] != before[str(second)]
    assert provenance.code_sha256()[str(first)] == before[str(first)]


def test_actual_source_closure_includes_nested_modules_and_initializers():
    from cess.provenance import PACKAGE_ROOT, source_files
    found = set(source_files())
    assert found == {path.resolve() for path in PACKAGE_ROOT.rglob('*.py') if path.is_file()}
    assert {PACKAGE_ROOT / '__init__.py', PACKAGE_ROOT / '__main__.py',
            PACKAGE_ROOT / 'config/loader.py', PACKAGE_ROOT / 'pipeline/support.py'} <= found
    assert all(path.is_relative_to(PACKAGE_ROOT) for path in found)




def test_generator_rejects_foreign_cached_canonical_module(monkeypatch, tmp_path):
    from cess.morphology.profile_generator import _project_module
    fake = types.ModuleType('cess.morphology.reference')
    fake.__file__ = str(tmp_path / 'reference.py')
    monkeypatch.setitem(sys.modules, 'cess.morphology.reference', fake)
    with pytest.raises(RuntimeError, match='Expected package module'):
        _project_module('cess.morphology.reference')


def test_generator_plans_and_source_binding_use_real_package_files():
    from cess.config.loader import DEFAULT_CONFIG_PATH
    from cess.morphology import generator, profile_generator
    axes = dict(n=[.5], re_pixel=[4.], pa_deg=[0.], axis_ratio=[1.])
    old_plan = profile_generator.plan_generation(DEFAULT_CONFIG_PATH, axes)
    new_plan = generator.plan_mge_library(DEFAULT_CONFIG_PATH, axes, [-1., 0., 1.])
    assert old_plan['schema'] == 'cess.generated_profile_library'
    assert new_plan['schema'] == 'cess.positive_marginal_library'
    for plan in (old_plan, new_plan):
        assert plan['generator_revision'] == '20261023-v0.9.0.dev1-R1'
        assert plan['software_version'] == PACKAGE_VERSION
        assert 'relative to the cess package' in plan['generator_source_layout']
    for module in (generator, profile_generator):
        assert all((module.PROJECT / name).is_file() for name in module._SOURCES)
        assert 'config/loader.py' in module._SOURCES
        assert all(not name.startswith('cess_') for name in module._SOURCES)
    assert generator._source_hashes() == {name: digest(generator.PROJECT / name) for name in generator._SOURCES}


def test_tiny_mge_library_publishes_canonical_source_snapshot(tmp_path):
    from cess.config.loader import DEFAULT_CONFIG_PATH
    from cess.morphology.generator import generate_mge_library, _source_hashes
    from cess.morphology.bundle import verify_mge_bundle
    path = tmp_path / 'positive'
    result = generate_mge_library(DEFAULT_CONFIG_PATH,
        dict(n=[.5], re_pixel=[4.], pa_deg=[0.], axis_ratio=[1.]), [-1., 0., 1.], path)
    assert result['source_sha256'] == _source_hashes()
    assert result['status'] == 'complete'
    assert verify_mge_bundle(path)['schema'] == 'cess.positive_marginal_library'


@pytest.mark.parametrize('module', ['cess', 'cess.morphology.profile_generator', 'cess.morphology.generator'])
def test_module_cli_help_from_foreign_directory(tmp_path, module):
    assert 'usage:' in process([sys.executable, '-s', '-B', '-m', module, '--help'], tmp_path, SRC)


@pytest.fixture(scope='module')
def built_wheel(tmp_path_factory):
    if sys.version_info[:2] != (3, 12):
        pytest.skip('The modern package supports Python 3.12 only; leave the legacy environment unchanged')
    from importlib.metadata import version
    import tomllib
    from packaging.requirements import Requirement
    for declared in tomllib.loads((ROOT / 'pyproject.toml').read_text())['build-system']['requires']:
        requirement = Requirement(declared)
        assert version(requirement.name) in requirement.specifier, (
            'Install the declared modern build toolchain in this Python 3.12 test environment: ' + declared)
    folder = tmp_path_factory.mktemp('cess-package-build')
    checkout = folder / 'source'
    checkout.mkdir()
    shutil.copytree(str(SRC), str(checkout / 'src'), ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'setup.cfg', 'README.md', 'LICENSE', 'RESOURCE_NOTICE.md'):
        shutil.copy2(str(ROOT / name), str(checkout / name))
    wheels = folder / 'wheels'
    wheels.mkdir()
    process([sys.executable, '-s', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check',
             'wheel', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir',
             '--wheel-dir', str(wheels), str(checkout)], folder, checkout / 'src')
    wheel, = wheels.glob('*.whl')
    return folder, wheel


def test_wheel_contains_all_default_resources_and_entrypoints(built_wheel):
    from cess.resources import DATA_ROOT, CALIBRATION_PATHS
    _, wheel = built_wheel
    with zipfile.ZipFile(str(wheel)) as archive:
        entries = set(archive.namelist())
        resources = {'cess/data/' + name for name in CALIBRATION_PATHS} | {'cess/data/emulator_parameters.json'}
        assert {name for name in entries if name.startswith('cess/data/')} == resources
        for name in sorted(resources):
            digestor = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digestor.update(block)
            assert digestor.hexdigest() == digest(DATA_ROOT / name[len('cess/data/'):])
        metadata_name, = [name for name in entries if name.endswith('.dist-info/METADATA')]
        from packaging.requirements import Requirement
        metadata = message_from_bytes(archive.read(metadata_name))
        assert metadata['Version'] == PACKAGE_VERSION
        assert metadata['Requires-Python'] == '<3.13,>=3.12'
        assert metadata['License-Expression'] == 'MIT'
        assert set(metadata.get_all('License-File')) == {'LICENSE', 'RESOURCE_NOTICE.md'}
        for filename in ('LICENSE', 'RESOURCE_NOTICE.md'):
            member, = [name for name in entries if name.endswith('.dist-info/licenses/' + filename)]
            assert archive.read(member) == (ROOT / filename).read_bytes()
        requirements = [Requirement(value) for value in metadata.get_all('Requires-Dist')]
        for platform in ('linux', 'darwin', 'win32'):
            selected = [item for item in requirements if item.marker is None or
                        item.marker.evaluate({'sys_platform': platform, 'extra': ''})]
            expected = {'astropy': '==8.0.2', 'specutils': '==2.4.0', 'tqdm': '==4.70.1'}
            expected.update(dict(zip(('h5py', 'numpy', 'scipy'),
                ('==3.14.0', '==2.2.6', '==1.15.3') if platform == 'linux' else
                ('==3.16.0', '==2.5.4', '==1.18.1'))))
            assert len(selected) == len(expected)  # exactly one pin per core package
            assert {item.name.lower(): str(item.specifier) for item in selected} == expected
        for extra, package, constraint in [('render', 'galsim', '==2.8.5'),
                ('tools', 'pandas', '==3.1.0'), ('plot', 'matplotlib', '==3.11.2'),
                ('viewer', 'matplotlib', '==3.11.2'),
                ('test', 'pytest', '==9.1.1')]:
            assert any(item.name.lower() == package and str(item.specifier) == constraint
                       and item.marker is not None and item.marker.evaluate({'extra': extra})
                       for item in requirements)
        assert all(item.name.lower() != 'photutils' for item in requirements)
        entrypoints_name, = [name for name in entries if name.endswith('.dist-info/entry_points.txt')]
        config = configparser.ConfigParser()
        config.read_string(archive.read(entrypoints_name).decode('utf-8'))
        assert dict(config['console_scripts']) == ENTRYPOINTS


def test_offline_wheel_target_install_and_cli(built_wheel):
    folder, wheel = built_wheel
    target = folder / 'installed'
    process([sys.executable, '-s', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check',
             'install', '--no-index', '--no-deps', '--no-compile', '--target', str(target), str(wheel)],
            folder, folder)
    code = '''
from pathlib import Path
import sys
import cess
from cess.config.loader import load_config
from cess.provenance import source_files
assert cess.__version__==EXPECTED_VERSION
target=Path(sys.argv[1]).resolve()
assert Path(cess.__file__).resolve().parent.parent==target
config=load_config()
assert Path(config['gems_path']).is_file()
assert all(target in path.parents for path in source_files())
print('installed package data and source closure passed')
'''
    code = code.replace('EXPECTED_VERSION', repr(PACKAGE_VERSION))
    assert 'installed package data' in process([sys.executable, '-s', '-B', '-c', code, str(target)], folder, target)
    assert 'usage:' in process([sys.executable, '-s', '-B', '-m', 'cess', '--help'], folder, target)
    for name in ENTRYPOINTS:
        assert (target / 'bin' / name).is_file()
        assert 'usage:' in process([str(target / 'bin' / name), '--help'], folder, target)
