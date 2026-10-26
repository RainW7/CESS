"""Optional desktop HDF5 viewer. Argument parsing does not import GUI libraries."""
import argparse
import os
from pathlib import Path
import sys


def parser():
    from cess._version import __version__
    result = argparse.ArgumentParser(description='CESS read-only HDF5 browser and spectrum viewer')
    result.add_argument('file', nargs='?', type=Path,
                        help='HDF5 file; omit to start an empty window')
    result.add_argument('--lang', choices=('en', 'zh'),
                        default=os.environ.get('CESS_H5_VIEWER_LANG', 'en'),
                        help='Interface language (default: en; also CESS_H5_VIEWER_LANG)')
    result.add_argument('--version', action='version', version='CESS viewer ' + __version__)
    return result


def main(argv=None):
    arguments_parser = parser()
    args = arguments_parser.parse_args(argv)
    if args.lang not in ('en', 'zh'):
        arguments_parser.error('CESS_H5_VIEWER_LANG must be en or zh')
    path = args.file.expanduser().absolute() if args.file is not None else None
    if path is not None and not path.is_file():
        arguments_parser.error('HDF5 file does not exist: ' + str(path))
    try:
        from .app import launch
    except ImportError as error:
        print('CESS viewer: optional GUI dependencies are unavailable: ' + str(error) +
              '\nInstall CESS with the [viewer] extra and ensure Tcl/Tk is available '
              '(see docs/VIEWER.md).', file=sys.stderr)
        return 2
    import tkinter
    try:
        return launch(str(path) if path is not None else None, args.lang)
    except tkinter.TclError as error:
        print('CESS viewer: cannot start the desktop interface: ' + str(error) +
              '\nCheck Tcl/Tk and the graphical display; use batch CESS commands on '
              'a headless server.', file=sys.stderr)
        return 2
