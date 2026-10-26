import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
sys.path.insert(0, str(CORE))
from cess_config import load_config, validate_config


@pytest.mark.parametrize('key,value', [
    ('photometry_aperture_fraction', 0),
    ('photometry_aperture_fraction', 1.1),
    ('photometry_aperture_radius_re', -1),
    ('photometry_detection_snr', float('nan')),
    ('photometry_systematic_mag', -0.01),
    ('zero_order_exposures', {'GU': 2, 'GV': 0, 'GI': 2}),
    ('zero_order_exposures', {'GU': 2, 'GV': 2}),
    ('photometry_exposures', {'nuv': 4, 'u': 2, 'g': 2, 'r': True, 'i': 2, 'z': 2, 'y': 4}),
])
def test_photometry_options_fail_before_source_processing(key, value):
    config = load_config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)
