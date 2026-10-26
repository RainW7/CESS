"""Compatibility entry for run_gravity_zouhu; implementation: cess.pipeline.paired."""
import sys as _sys
from pathlib import Path as _Path
_source = _Path(__file__).resolve().parent.parent / 'src'
if _source.is_dir() and str(_source) not in _sys.path:
    _sys.path.insert(0, str(_source))
from cess.compat import expose as _expose
_as_script = __name__ == '__main__'
_module = _expose(globals(), 'cess.pipeline.paired')
if _as_script:
    from cess.pipeline.support import legacy_main
    _sys.exit(legacy_main(_module.task, _module.configure, _module.hdf5filenames))
