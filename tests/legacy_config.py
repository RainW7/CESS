"""Explicit D4 instrument settings for tests of historical numerical contracts.

The JSON is a byte-preserved configuration snapshot. It is not the current
scientific default, and new-profile tests must keep using load_config().
Calibration paths are resolved against the installed/current package data, so
this fixture does not depend on a developer's pre-release working directory.
"""
import json
from pathlib import Path


def legacy_config_mapping():
    path = Path(__file__).resolve().parent / 'baselines/config_20261024_D4.json'
    return json.loads(path.read_text())


def load_legacy_config():
    from cess.config.loader import DEFAULT_CONFIG_PATH, validate_config
    return validate_config(legacy_config_mapping(), base_dir=DEFAULT_CONFIG_PATH.parent)
