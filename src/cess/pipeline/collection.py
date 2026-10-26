"""Collect deterministic spectral results without detector or RNG operations."""
import numpy as np


_BANDS = ('GU', 'GV', 'GI')
_BASE_FIELDS = ('elec_per_column', 'fnu_with_noise', 'efnu', 'rms_in_e',
                'snr', 'snr_mean', 'spec_mask', 'snr_mask')
_WCE_FIELDS = ('flux_wce', 'elec_per_column_wce', 'fnu_with_noise_wce',
               'efnu_wce', 'snr_wce')
_PAIRED_FIELDS = ('elec_per_column', 'elec_per_column_wce', 'fnu_with_noise',
                  'fnu_with_noise_wce', 'efnu', 'efnu_wce', 'rms_in_e',
                  'snr', 'snr_wce', 'snr_mean', 'spec_mask', 'snr_mask', 'flux_wce')
_STATISTICS = ('rms_in_e', 'snr_mean')


class SpectralResults:
    """Own spectral lists for one runner task, excluding covariance and photometry.

    ``waves`` maps GU/GV/GI to their wavelength arrays. Standard base and _0th
    variants each collect the eight ``_BASE_FIELDS`` in that order. Paired base
    uses those same fields; its _wce tuple instead contains the five
    ``_WCE_FIELDS``. Flux/error inputs retain the runner's fnu units until
    ``spectra`` applies the original factor 1e29. RMS and mean S/N are scalars.

    Noise tuple unpacking and calls remain in the runner. ``spectra`` runs before
    padding and ``offsets`` after padding, preserving their conversion boundary.
    """

    def __init__(self, waves, *, paired=False, wave_error=False):
        self.waves, self.paired = waves, paired
        self._prefixes = tuple(prefix for band in _BANDS for prefix in
                               ((band.lower(),) if paired else (band.lower(), band.lower() + '0')))
        fields = _PAIRED_FIELDS if paired else _BASE_FIELDS
        self._rows = {prefix + '_' + field: [] for prefix in self._prefixes for field in fields}
        offset_prefixes = tuple(band.lower() for band in _BANDS)
        if not paired:
            offset_prefixes += tuple(band.lower() + '0' for band in _BANDS)
        self._offset_rows = {prefix + '_wave_off': [] for prefix in offset_prefixes} if wave_error else {}

    def append(self, band, variant, values, *, wave_offset=None):
        """Append already-unpacked noise values without coercion or arithmetic."""
        prefix = band.lower() + ('0' if variant == '_0th' else '')
        fields = _WCE_FIELDS if variant == '_wce' else _BASE_FIELDS
        for field, value in zip(fields, values, strict=True):
            self._rows[prefix + '_' + field].append(value)
        offset_key = prefix + '_wave_off'
        if variant != '_wce' and offset_key in self._offset_rows:
            self._offset_rows[offset_key].append(wave_offset)

    def append_invalid(self):
        """Keep one all-NaN source row, using scalar placeholders for statistics."""
        fields = _PAIRED_FIELDS if self.paired else _BASE_FIELDS
        for prefix in self._prefixes:
            width = len(self.waves[prefix[:2].upper()])
            for field in fields:
                value = np.nan if field in _STATISTICS else np.full(width, np.nan)
                self._rows[prefix + '_' + field].append(value)
        for rows in self._offset_rows.values():
            rows.append(np.nan)

    def spectra(self, source_count):
        """Return ordered padding inputs and Nx1 statistics with legacy dtypes.

        Convert with np.array, then scale fnu spectra/errors, or reshape scalar
        statistics, in the original runner order. Masks retain their raw dtype;
        the caller's existing padding operation applies their bool conversion.
        """
        data, statistics = {}, {}
        fields = ('flux_wce',) + _PAIRED_FIELDS[:-1] if self.paired else _BASE_FIELDS
        for prefix in self._prefixes:
            for field in fields:
                values = np.array(self._rows[prefix + '_' + field])
                output_field = field
                if field.startswith('fnu_with_noise'):
                    values = values * 1e29
                    output_field = field.replace('fnu_with_noise', 'ujy_with_noise')
                elif field.startswith('efnu'):
                    values = values * 1e29
                    output_field = field.replace('efnu', 'ferr')
                elif field == 'flux_wce':
                    values = values * 1e29
                    output_field = 'ujy_wce'
                if field in _STATISTICS:
                    statistics[prefix + '_' + field] = values.reshape((source_count, 1))
                else:
                    data[prefix + '_' + output_field] = values
        return data, statistics

    def offsets(self, source_count):
        """Convert optional offsets to Nx1 arrays only after caller-side padding."""
        return {name: np.array(rows).reshape((source_count, 1))
                for name, rows in self._offset_rows.items()}
