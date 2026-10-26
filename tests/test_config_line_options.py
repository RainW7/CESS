"""Explicit, independent pixel-window configuration for line diagnostics."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'))
from cess_config import load_config


def test_line_window_defaults_and_copy_are_independent():
    config = load_config()
    design = config.pop('line_measurement')
    assert load_config(config)['line_measurement'] == design
    other = load_config(config)
    other['line_measurement']['line_width_pixels']['GV'] = 11
    assert load_config(config)['line_measurement']['line_width_pixels']['GV'] == 40


def test_line_window_can_be_changed_without_changing_legacy_pixel_snr_window():
    config = load_config()
    original = config['gv_elwidth']
    config['line_measurement'] = {'line_width_pixels': {'GV': 13, 'GI': 17},
                                  'sideband_width_pixels': 7, 'sideband_gap_pixels': 2}
    validated = load_config(config)
    assert validated['line_measurement'] == config['line_measurement']
    assert validated['gv_elwidth'] == original


@pytest.mark.parametrize('design', [None, [], {},
    {'line_width_pixels': {'GV': 40, 'GI': 40}},
    {'line_width_pixels': {'GV': 40, 'GI': 40}, 'sideband_width_pixels': 20, 'sideband_gap_pixels': 0, 'typo': 1},
    {'line_width_pixels': {'GV': 40}, 'sideband_width_pixels': 20, 'sideband_gap_pixels': 0},
    {'line_width_pixels': {'GV': 0, 'GI': 40}, 'sideband_width_pixels': 20, 'sideband_gap_pixels': 0},
    {'line_width_pixels': {'GV': True, 'GI': 40}, 'sideband_width_pixels': 20, 'sideband_gap_pixels': 0},
    {'line_width_pixels': {'GV': 40.5, 'GI': 40}, 'sideband_width_pixels': 20, 'sideband_gap_pixels': 0},
    {'line_width_pixels': {'GV': 40, 'GI': 40}, 'sideband_width_pixels': 1, 'sideband_gap_pixels': 0},
    {'line_width_pixels': {'GV': 40, 'GI': 40}, 'sideband_width_pixels': 20, 'sideband_gap_pixels': -1},
    {'line_width_pixels': {'GV': 40, 'GI': 40}, 'sideband_width_pixels': 20, 'sideband_gap_pixels': float('nan')}])
def test_bad_line_design_is_rejected(design):
    config = load_config()
    config['line_measurement'] = design
    with pytest.raises(ValueError, match='line_measurement|sideband|GV'):
        load_config(config)
