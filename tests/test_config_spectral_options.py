"""Independent template LSF configuration, without inferring it from sampling."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
sys.path.insert(0, str(CORE))
from cess_config import load_config, validate_config


@pytest.fixture
def parameters():
    return load_config(CORE / 'emulator_parameters.json')


def test_default_lsf_is_explicitly_unknown_and_broader_policy_errors(parameters):
    parameters.pop('template_lsf')
    parameters.pop('template_lsf_broader_policy')
    result = validate_config(parameters)
    assert result['template_lsf'] == {'mode': 'unknown'}
    assert result['template_lsf_broader_policy'] == 'error'
    assert 'template_lsf' not in parameters


@pytest.mark.parametrize('lsf', [
    {'mode': 'unknown'}, {'mode': 'delta'},
    {'mode': 'resolving_power', 'value': 300},
    {'mode': 'fwhm_rest', 'value': 2.5},
])
@pytest.mark.parametrize('policy', ['error', 'preserve'])
def test_valid_template_lsf_and_explicit_policy(parameters, lsf, policy):
    parameters.update(template_lsf=lsf, template_lsf_broader_policy=policy)
    result = validate_config(parameters)
    assert result['template_lsf'] == lsf
    assert result['template_lsf_broader_policy'] == policy
    if 'value' in lsf:
        assert isinstance(result['template_lsf']['value'], float)
    assert result['template_lsf'] is not lsf


@pytest.mark.parametrize('lsf', [
    None, False, [], 'delta', {}, {'value': 300},
    {'mode': None}, {'mode': True}, {'mode': []}, {'mode': 'sampling'},
    {'mode': 'delta', 'value': 0},
    {'mode': 'unknown', 'value': None},
    {'mode': 'unknown', 'units': 'Angstrom'},
    {'mode': 'delta', 'extra': 'unrecognised'},
    {'mode': 'resolving_power'},
    {'mode': 'fwhm_rest'},
    {'mode': 'resolving_power', 'value': 300, 'units': 'dimensionless'},
    {'mode': 'fwhm_rest', 'value': 2., 'extra': 1},
])
def test_template_lsf_rejects_ambiguous_unknown_or_extra_keys(parameters, lsf):
    parameters['template_lsf'] = lsf
    with pytest.raises(ValueError, match='template_lsf'):
        validate_config(parameters)


@pytest.mark.parametrize('mode', ['resolving_power', 'fwhm_rest'])
@pytest.mark.parametrize('value', [0, -1, True, False, np.bool_(True),
                                   np.nan, np.inf, -np.inf, None, '300', [300]])
def test_template_lsf_width_or_resolution_is_finite_positive_numeric(
        parameters, mode, value):
    parameters['template_lsf'] = {'mode': mode, 'value': value}
    with pytest.raises(ValueError, match='template_lsf'):
        validate_config(parameters)


@pytest.mark.parametrize('policy', ['clamp', 'deconvolve', 'ERROR', True, None,
                                    [], {}, 0, np.array(['error'])])
def test_template_broader_policy_rejects_silent_or_invalid_alternatives(parameters, policy):
    parameters['template_lsf_broader_policy'] = policy
    with pytest.raises(ValueError, match='template_lsf_broader_policy'):
        validate_config(parameters)


def test_template_lsf_copies_are_independent_and_survive_json_roundtrip(parameters, tmp_path):
    original = deepcopy(parameters)
    original['template_lsf'] = {'mode': 'fwhm_rest', 'value': 3.}
    result = validate_config(original)
    result['template_lsf']['value'] = 9.
    assert original['template_lsf']['value'] == 3.
    source = tmp_path / 'spectral.json'
    source.write_text(json.dumps(original))
    loaded = load_config(source)
    assert loaded['template_lsf'] == {'mode': 'fwhm_rest', 'value': 3.}
    assert loaded['template_lsf_broader_policy'] == 'error'
