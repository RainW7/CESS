"""Shared one-dimensional schema contracts; no pipeline or I/O dependency."""
import numpy as np


SCHEMA_VERSION = '2.5'

REPAIR_REVISION = '20261024-D4'

COVARIANCE_MODEL = 'independent-extracted-bins-linear-shift-v1'

MGE_MORPHOLOGY_FIELDS = (
    'selected_parameters', 'selected_indices', 'source_fit_l1_estimate',
    'conditional_fit_error_indicator', 'kernel_window_fraction',
    'kernel_absolute_fraction', 'kernel_n_bins', 'quadrature_error_estimate',
    'omitted_probability_bound', 'valid', 'kernel_applied')

MGE_MORPHOLOGY_ATTRIBUTES = {
    'selected_parameter_names': ['n', 're_pixel', 'pa_deg', 'axis_ratio'],
    'selected_parameter_units': ['1', 'pixel', 'degree', '1'],
    'kernel_definition': 'P(x in bin,y in strip)/P(y in strip); finite x-window sum retained',
    'absolute_fraction_definition': 'source_fraction * kernel_window_fraction; source_fraction applied once',
    'fit_error_definition': '2*source_fit_l1_estimate/source_fraction; uncertified; nearest-grid error excluded'}

LINE_MEASUREMENT_DTYPE = np.dtype([
    ('source_row', '<i8'), ('peak_id', '<i8'), ('center_index', '<i8'), ('center_wave', '<f8'),
    ('line_start', '<i8'), ('line_stop', '<i8'), ('left_start', '<i8'), ('left_stop', '<i8'),
    ('right_start', '<i8'), ('right_stop', '<i8'), ('nline', '<i8'), ('nside', '<i8'),
    ('flux', '<f8'), ('flux_error', '<f8'), ('snr', '<f8'),
    ('model_flux', '<f8'), ('model_snr', '<f8'), ('valid', '?'),
    ('flag', '<u4'), ('contaminated', '?'), ('reason', 'S128')])

OPTION_DEFAULTS = {'apply_morphology_effect': False,
                   'apply_wavelength_calibration_error': False,
                   'apply_photometric': False, 'apply_el_detect': False}

