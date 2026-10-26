"""Legacy entry point aliases; scientific implementations live in the package.

Normal imports return the canonical module itself, including its mutable
compatibility state and monkeypatch bindings. File-spec execution receives a
read facade; supported simulation APIs live on the canonical module.
"""
import importlib
import sys


def expose(namespace, target):
    """Make an old module name refer to the same canonical module object."""
    previous_name = namespace['__name__']
    module = importlib.import_module(target)
    namespace.update(module.__dict__)
    if previous_name != '__main__':
        sys.modules[previous_name] = module
    return module
