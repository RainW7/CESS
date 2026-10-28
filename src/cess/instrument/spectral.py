"""Explicit one-dimensional instrumental LSF, sampling and analysis windows.

All wavelengths and FWHM values are observed-frame Angstrom. A resolving power
quoted only at a central wavelength defines the FWHM there; extending that width
unchanged across a band is an explicit approximation, not a measured R(lambda).
Sampling is a separate choice. Legacy mappings retain their historical center
grid and index windows unless they explicitly select the newer contracts.

Importing this module reads no configuration or calibration resources.
"""
from collections.abc import Mapping
import math
from numbers import Real

BANDS = ("gu", "gv", "gi")
MAX_SPECTRAL_PIXELS = 1_000_000


def _prefix(band):
    if not isinstance(band, str) or band.lower() not in BANDS:
        raise ValueError("Invalid grism; expected GU, GV or GI.")
    return band.lower()


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(name + " must be a finite positive number.")
    return float(value)


def _integer(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value < 0 or int(value) != value):
        raise ValueError(name + " must be a nonnegative integer.")
    return int(value)


def _required(config, name):
    if name not in config:
        raise ValueError("Missing required CESS configuration field: " + name)
    return config[name]


def _model(config, key, default, allowed):
    model = config.get(key, {"mode": default})
    if not isinstance(model, Mapping):
        raise ValueError(key + " must be a mapping containing an explicit mode.")
    model = dict(model)
    mode = model.get("mode")
    if not isinstance(mode, str) or mode not in allowed:
        raise ValueError(key + ".mode must be " + ", ".join(allowed) + ".")
    if set(model) != allowed[mode]:
        raise ValueError(key + " mode " + mode + " requires exactly: " +
                         ", ".join(sorted(allowed[mode])))
    return model


def _lsf_model(config):
    model = _model(config, "instrument_lsf", "fixed_fwhm", {
        "fixed_fwhm": {"mode"},
        "central_resolving_power": {"mode", "wavelength_law"},
    })
    if model["mode"] == "central_resolving_power" and model["wavelength_law"] != "constant_fwhm":
        raise ValueError("instrument_lsf.wavelength_law must explicitly be constant_fwhm.")
    return model


def _sampling_model(config):
    model = _model(config, "spectral_sampling", "legacy", {
        "legacy": {"mode"},
        "resolution_elements": {"mode", "pixels_per_resolution_element"},
    })
    if model["mode"] == "resolution_elements":
        model["pixels_per_resolution_element"] = _positive(
            model["pixels_per_resolution_element"], "spectral_sampling.pixels_per_resolution_element")
    return model


def _window_model(config):
    return _model(config, "spectral_window", "indices", {
        "indices": {"mode"}, "bandpass": {"mode"},
    })


def _bounds(config, prefix):
    minimum = _positive(_required(config, prefix + "_min_wl"), prefix + "_min_wl")
    maximum = _positive(_required(config, prefix + "_max_wl"), prefix + "_max_wl")
    if minimum >= maximum:
        raise ValueError(prefix + " requires min_wl < max_wl.")
    return minimum, maximum


def target_fwhm(config, band):
    """Return the declared pre-pixel Gaussian target FWHM, in observed Angstrom.

    ``central_resolving_power`` uses central_wl/Res, held constant across the
    band. With no explicit instrument_lsf, only dlambda is operative: old Res
    entries remain descriptive and do not silently change old configurations.
    """
    prefix = _prefix(band)
    if _lsf_model(config)["mode"] == "central_resolving_power":
        center = _positive(_required(config, prefix + "_central_wl"), prefix + "_central_wl")
        resolving_power = _positive(_required(config, prefix + "_Res"), prefix + "_Res")
        result = center / resolving_power
        return _positive(result, prefix + " central_wl/Res")
    return _positive(_required(config, prefix + "_dlambda"), prefix + "_dlambda")


def output_pixel_count(config, band, fwhm_arcsec=None):
    """Return a bounded spectral pixel count without allocating a grid.

    Resolution-element sampling uses ceil(bandwidth/(target_FWHM/pixels_per_RE)).
    Legacy sampling exactly retains the start/end/dlambda coarse count followed
    by its historical angular factor. An explicit fwhm_arcsec overrides ree80
    only for that legacy contract; new sampling does not infer pixel dispersion
    from a spatial size.
    """
    prefix = _prefix(band)
    model = _sampling_model(config)
    if model["mode"] == "resolution_elements":
        minimum, maximum = _bounds(config, prefix)
        step = target_fwhm(config, prefix) / model["pixels_per_resolution_element"]
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Spectral sampling step must be finite and positive.")
        unrounded = (maximum - minimum) / step
        if not math.isfinite(unrounded) or unrounded > MAX_SPECTRAL_PIXELS:
            raise ValueError("Spectral sampling exceeds the maximum of {} pixels.".format(MAX_SPECTRAL_PIXELS))
        count = math.ceil(unrounded)
    else:
        first = _positive(_required(config, prefix + "_start_wl"), prefix + "_start_wl")
        last = _positive(_required(config, prefix + "_end_wl"), prefix + "_end_wl")
        delta = _positive(_required(config, prefix + "_dlambda"), prefix + "_dlambda")
        if first >= last:
            raise ValueError(prefix + " requires start_wl < end_wl.")
        # Keep the original floor, +1, multiplication and Python round order.
        # Reject overflow before converting a pathological ratio to an integer.
        coarse_ratio = (last - first) / delta
        if not math.isfinite(coarse_ratio):
            raise ValueError("Legacy spectral sampling exceeds the finite grid limit.")
        coarse = int(coarse_ratio) + 1
        spatial = _positive(_required(config, prefix + "_ree80") if fwhm_arcsec is None else fwhm_arcsec,
                            prefix + "_ree80" if fwhm_arcsec is None else "fwhm_arcsec")
        scale = _positive(_required(config, "arcsecperpix"), "arcsecperpix")
        unrounded = coarse * spatial / scale
        if not math.isfinite(unrounded) or unrounded > MAX_SPECTRAL_PIXELS:
            raise ValueError("Spectral sampling exceeds the maximum of {} pixels.".format(MAX_SPECTRAL_PIXELS))
        count = round(unrounded)
    if count < 2:
        raise ValueError("Spectral sampling requires at least two pixels.")
    return int(count)


def output_wavelengths(config, band, fwhm_arcsec=None):
    """Return pixel centers in observed Angstrom under the declared grid mode.

    Resolution-element sampling interprets min/max as the outer integration
    boundaries, constructing uniform edges and taking their midpoints. Legacy
    sampling keeps min/max as its first and last pixel centers.
    """
    prefix = _prefix(band)
    minimum, maximum = _bounds(config, prefix)
    count = output_pixel_count(config, prefix, fwhm_arcsec)
    import numpy as np
    if _sampling_model(config)["mode"] == "legacy":
        return np.linspace(minimum, maximum, count)
    edges = np.linspace(minimum, maximum, count + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def window_indices(config, band, wave):
    """Return the half-open analysis interval on the actual supplied wave grid.

    Bandpass mode includes centers in [min_wl,max_wl], using searchsorted on the
    real parent/current grid rather than indexes cached for a different grid.
    Indices mode retains the explicit historical integers without clamping to
    the supplied grid: existing consumers preserve their own handling of an
    out-of-range window instead of silently analyzing a different interval.
    """
    prefix = _prefix(band)
    if _window_model(config)["mode"] == "indices":
        start = _integer(_required(config, prefix + "_init_idx"), prefix + "_init_idx")
        stop = _integer(_required(config, prefix + "_end_idx"), prefix + "_end_idx")
        if start >= stop:
            raise ValueError(prefix + " requires init_idx < end_idx.")
        return start, stop
    import numpy as np
    wavelengths = np.asarray(wave, dtype=float)
    if (wavelengths.ndim != 1 or wavelengths.size == 0 or
            not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0) or
            np.any(np.diff(wavelengths) <= 0)):
        raise ValueError("Analysis wavelength centers must be finite, positive and strictly increasing.")
    minimum, maximum = _bounds(config, prefix)
    return int(np.searchsorted(wavelengths, minimum, side="left")), int(
        np.searchsorted(wavelengths, maximum, side="right"))


def normalise_spectral_configuration(config):
    """Normalize the caller's private config copy in place and return it.

    Derived dlambda and index values are compatibility caches, not independent
    controls in the new modes. Original caller mappings must be copied by the
    configuration loader before invoking this helper.
    """
    config["instrument_lsf"] = _lsf_model(config)
    config["spectral_sampling"] = _sampling_model(config)
    config["spectral_window"] = _window_model(config)
    for band in BANDS:
        minimum, maximum = _bounds(config, band)
        for suffix in ("start_wl", "end_wl"):
            config[band + "_" + suffix] = _positive(
                _required(config, band + "_" + suffix), band + "_" + suffix)
        if not (config[band + "_start_wl"] <= minimum < maximum <= config[band + "_end_wl"]):
            raise ValueError(band + " requires start <= min < max <= end wavelength.")
        if config["instrument_lsf"]["mode"] == "central_resolving_power":
            for suffix in ("central_wl", "Res"):
                config[band + "_" + suffix] = _positive(
                    _required(config, band + "_" + suffix), band + "_" + suffix)
            if not minimum <= config[band + "_central_wl"] <= maximum:
                raise ValueError(band + "_central_wl must lie within the bandpass.")
        config[band + "_dlambda"] = target_fwhm(config, band)
        # Validate the new finite grid even when its analysis window happens to
        # use old explicit indexes. Legacy grids retain the historical lazy check.
        if config["spectral_sampling"]["mode"] == "resolution_elements":
            output_pixel_count(config, band)
        if config["spectral_window"]["mode"] == "bandpass":
            config[band + "_init_idx"], config[band + "_end_idx"] = window_indices(
                config, band, output_wavelengths(config, band))
        else:
            config[band + "_init_idx"], config[band + "_end_idx"] = window_indices(config, band, None)
    return config
