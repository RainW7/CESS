"""No-clobber publication is native and atomic, not check-then-rename."""
import ctypes
import errno
import os
from pathlib import Path
import sys

import pytest

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import cess_profile_publish as publisher


def _staging(tmp_path):
    staging = tmp_path / 'owned-staging'
    staging.mkdir()
    for name, content in (('widthlib.pkl', b'width'), ('heightlib.pkl', b'height'),
                          ('manifest.json', b'{"complete":true}')):
        (staging / name).write_bytes(content)
    return staging


@pytest.mark.skipif(sys.platform != 'darwin', reason='real macOS renamex_np test')
def test_native_mac_publication_moves_complete_directory_once(tmp_path):
    staging = _staging(tmp_path)
    output = tmp_path / 'new-library'
    inode = staging.stat().st_ino
    assert publisher.publish_directory(staging, output) is None
    assert not staging.exists()
    assert output.is_dir() and not output.is_symlink()
    assert output.stat().st_ino == inode
    assert {path.name: path.read_bytes() for path in output.iterdir()} == {
        'widthlib.pkl': b'width', 'heightlib.pkl': b'height',
        'manifest.json': b'{"complete":true}'}
    another = _staging(tmp_path)
    with pytest.raises(FileExistsError):
        publisher.publish_directory(another, output)
    assert another.is_dir()  # Cleanup belongs to the caller.
    assert output.stat().st_ino == inode


@pytest.mark.parametrize('kind', ['file', 'empty-directory', 'populated-directory',
                                  'symlink', 'dangling-symlink'])
def test_every_existing_output_is_preserved_without_native_call(tmp_path, monkeypatch, kind):
    staging = _staging(tmp_path)
    output = tmp_path / 'new-library'
    target = tmp_path / 'target'
    if kind == 'file':
        output.write_bytes(b'original file')
    elif kind in ('empty-directory', 'populated-directory'):
        output.mkdir()
        if kind == 'populated-directory':
            (output / 'important').write_bytes(b'original directory')
    else:
        if kind == 'symlink':
            target.mkdir()
            (target / 'important').write_bytes(b'original target')
        output.symlink_to(target, target_is_directory=True)
    before = output.lstat()
    monkeypatch.setattr(publisher, '_rename_no_replace',
                        lambda *args: pytest.fail('existing paths must fail before native call'))
    with pytest.raises(FileExistsError):
        publisher.publish_directory(staging, output)
    assert output.lstat().st_ino == before.st_ino
    assert staging.is_dir()
    if kind == 'file':
        assert output.read_bytes() == b'original file'
    elif kind == 'populated-directory':
        assert (output / 'important').read_bytes() == b'original directory'
    elif kind in ('symlink', 'dangling-symlink'):
        assert output.is_symlink() and Path(os.readlink(str(output))) == target
        if kind == 'symlink':
            assert (target / 'important').read_bytes() == b'original target'
        else:
            assert not target.exists()


@pytest.mark.skipif(sys.platform != 'darwin', reason='real macOS collision arbitration')
@pytest.mark.parametrize('kind', ['empty-directory', 'file', 'symlink', 'dangling-symlink'])
def test_native_mac_rejects_destination_created_after_preflight(tmp_path, monkeypatch, kind):
    staging = _staging(tmp_path)
    output = tmp_path / 'new-library'
    unrelated = tmp_path / 'unrelated'
    unrelated.write_bytes(b'keep')
    native_rename = publisher._rename_no_replace
    competitor_inode = []

    def race(source, destination):
        assert not os.path.lexists(str(destination))
        if kind == 'empty-directory':
            destination.mkdir()
        elif kind == 'file':
            destination.write_bytes(b'competitor')
        else:
            target = tmp_path / 'competitor-target'
            if kind == 'symlink':
                target.mkdir()
            destination.symlink_to(target, target_is_directory=True)
        competitor_inode.append(destination.lstat().st_ino)
        native_rename(source, destination)

    monkeypatch.setattr(publisher, '_rename_no_replace', race)
    with pytest.raises(OSError) as failure:
        publisher.publish_directory(staging, output)
    assert failure.value.errno in (errno.EEXIST, errno.ENOTEMPTY)
    assert output.lstat().st_ino == competitor_inode[0]
    assert staging.is_dir()
    assert unrelated.read_bytes() == b'keep'
    if kind == 'file':
        assert output.read_bytes() == b'competitor'
    elif kind in ('symlink', 'dangling-symlink'):
        assert output.is_symlink()


@pytest.mark.parametrize('role', ['staging', 'output'])
def test_parent_traversal_is_rejected_before_touching_paths(tmp_path, role):
    staging = _staging(tmp_path)
    output = tmp_path / 'new-library'
    bad = tmp_path / 'not-needed' / '..' / ('owned-staging' if role == 'staging' else 'new-library')
    with pytest.raises(ValueError, match='parent traversal'):
        publisher.publish_directory(bad if role == 'staging' else staging,
                                    bad if role == 'output' else output)
    assert staging.is_dir() and not output.exists()


@pytest.mark.parametrize('bad', ['', '.', '..', '/', '/private/tmp/.'])
def test_dangerous_final_names_are_rejected(tmp_path, bad):
    staging = _staging(tmp_path)
    with pytest.raises(ValueError):
        publisher.publish_directory(staging, bad)
    assert staging.is_dir()


def test_same_path_non_siblings_missing_parent_and_missing_source_reject(tmp_path):
    staging = _staging(tmp_path)
    with pytest.raises(ValueError, match='different'):
        publisher.publish_directory(staging, staging)
    other = tmp_path / 'different-parent'
    other.mkdir()
    with pytest.raises(ValueError, match='siblings'):
        publisher.publish_directory(staging, other / 'output')
    with pytest.raises(FileNotFoundError):
        publisher.publish_directory(staging, tmp_path / 'missing-parent' / 'output')
    with pytest.raises(FileNotFoundError):
        publisher.publish_directory(tmp_path / 'missing-source', tmp_path / 'output')
    assert staging.is_dir()


@pytest.mark.parametrize('kind', ['file', 'symlink', 'dangling-symlink'])
def test_staging_must_be_a_real_directory(tmp_path, kind):
    source = tmp_path / 'source'
    if kind == 'file':
        source.write_bytes(b'keep')
        expected = NotADirectoryError
    else:
        target = tmp_path / 'target'
        if kind == 'symlink':
            target.mkdir()
        source.symlink_to(target, target_is_directory=True)
        expected = ValueError
    with pytest.raises(expected):
        publisher.publish_directory(source, tmp_path / 'output')
    assert os.path.lexists(str(source))
    assert not (tmp_path / 'output').exists()


class FakeFunction:
    def __init__(self, result=0, error=0):
        self.calls = []
        self.result = result
        self.error = error

    def __call__(self, *args):
        self.calls.append(args)
        ctypes.set_errno(self.error)
        return self.result


def test_mocked_linux_uses_libc_renameat2_exact_flags_and_types(tmp_path, monkeypatch):
    staging = _staging(tmp_path)
    output = tmp_path / 'new-library'
    function = FakeFunction()
    library = type('Library', (), {})()
    library.renameat2 = function
    loads = []

    def load(name, use_errno):
        loads.append((name, use_errno))
        return library

    monkeypatch.setattr(publisher, '_PLATFORM', 'linux')
    monkeypatch.setattr(publisher.ctypes, 'CDLL', load)
    publisher.publish_directory(staging, output)
    assert loads == [(None, True)]
    assert function.calls == [(-100, os.fsencode(staging), -100, os.fsencode(output), 1)]
    assert function.argtypes == [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.c_char_p, ctypes.c_uint]
    assert function.restype is ctypes.c_int


@pytest.mark.parametrize('platform', ['darwin', 'linux'])
@pytest.mark.parametrize('error', [errno.EEXIST, errno.ENOTEMPTY, errno.ENOSYS,
                                  errno.EINVAL, errno.ENOTSUP, errno.EACCES, errno.EXDEV])
def test_native_errors_never_fall_back_or_clean_staging(tmp_path, monkeypatch, platform, error):
    staging = _staging(tmp_path)
    output = tmp_path / 'new-library'
    function = FakeFunction(result=-1, error=error)
    library = type('Library', (), {})()
    setattr(library, 'renamex_np' if platform == 'darwin' else 'renameat2', function)
    monkeypatch.setattr(publisher, '_PLATFORM', platform)
    monkeypatch.setattr(publisher.ctypes, 'CDLL', lambda *args, **kwargs: library)
    monkeypatch.setattr(publisher.os, 'rename', lambda *args: pytest.fail('unsafe rename fallback'))
    monkeypatch.setattr(publisher.os, 'replace', lambda *args: pytest.fail('unsafe replace fallback'))
    with pytest.raises(OSError) as failure:
        publisher.publish_directory(staging, output)
    assert failure.value.errno == error
    assert len(function.calls) == 1
    assert staging.is_dir() and not output.exists()
    assert (staging / 'manifest.json').read_bytes() == b'{"complete":true}'


@pytest.mark.parametrize('platform', ['darwin', 'linux', 'unsupported-system'])
def test_missing_native_support_fails_closed(tmp_path, monkeypatch, platform):
    staging = _staging(tmp_path)
    output = tmp_path / 'new-library'
    monkeypatch.setattr(publisher, '_PLATFORM', platform)
    monkeypatch.setattr(publisher.ctypes, 'CDLL', lambda *args, **kwargs: object())
    with pytest.raises(OSError) as failure:
        publisher.publish_directory(staging, output)
    assert failure.value.errno == errno.ENOTSUP
    assert staging.is_dir() and not output.exists()
