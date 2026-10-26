#!/usr/bin/env python3
"""Repository entry point for the optional CESS HDF5 viewer.

Install the viewer extra first: python -m pip install '.[viewer]'.
Installed users can run cess-viewer instead.
"""
from pathlib import Path
import sys

# Prefer this checkout when running its compatibility script directly.
_source = Path(__file__).resolve().parent / 'src'
if _source.is_dir():
    sys.path.insert(0, str(_source))

from cess.viewer import main

if __name__ == '__main__':
    raise SystemExit(main())
