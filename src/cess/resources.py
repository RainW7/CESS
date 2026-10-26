"""Locations of bundled defaults, without reading resource contents at import.

CESS is installed as ordinary files (zip_safe=False). Returned Paths therefore
remain valid for FITS/HDF5/pickle consumers throughout a worker's lifetime; no
temporary extraction/cache or cwd-dependent lookup is used.
"""
from pathlib import Path, PurePosixPath


DATA_ROOT = Path(__file__).resolve().with_name('data')
CALIBRATION_PATHS = (
    'basic_file/updated_bkg.fits', 'basic_file/gems_cat.fits',
    'basic_file/widthlib_20x20x10x10.pkl', 'basic_file/heightlib_20x20x10x10.pkl',
    'csst_sls_tp/GU_throughput_1st.fits', 'csst_sls_tp/GV_throughput_1st_no1567.fits',
    'csst_sls_tp/GI_throughput_1st.fits', 'csst_sls_tp/GU_throughput_0th.fits',
    'csst_sls_tp/GV_throughput_0th_no56.fits', 'csst_sls_tp/GI_throughput_0th.fits',
    'csst_phot_tp/csst_phot.nuv.fits', 'csst_phot_tp/csst_phot.u.fits',
    'csst_phot_tp/csst_phot.g.fits', 'csst_phot_tp/csst_phot.r.fits',
    'csst_phot_tp/csst_phot.i.fits', 'csst_phot_tp/csst_phot.z.fits',
    'csst_phot_tp/csst_phot.y.fits',
)


def resource_path(relative):
    """Return a package-data Path; reject absolute/traversal input, do not read it."""
    value = PurePosixPath(str(relative))
    if value.is_absolute() or not value.parts or '..' in value.parts or '\\' in str(relative):
        raise ValueError('Resource names must be nonempty relative paths without parent traversal.')
    return DATA_ROOT.joinpath(*value.parts)


def default_config_path():
    """Return the bundled JSON path without opening or parsing that file."""
    return resource_path('emulator_parameters.json')


def calibration_files():
    """Return the 17 bundled default-calibration Paths, without loading them."""
    return tuple(resource_path(name) for name in CALIBRATION_PATHS)
