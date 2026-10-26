#!/usr/bin/env python
"""Compatibility CLI for explicit, no-clobber profile-library generation."""
from pathlib import Path
import sys
_source = Path(__file__).resolve().parents[2] / 'src'
if _source.is_dir() and str(_source) not in sys.path:
    sys.path.insert(0, str(_source))

def main(argv=None):
    from cess.morphology.legacy_generator import main as generate
    return generate(argv)

if __name__ == '__main__':
    sys.exit(main())
