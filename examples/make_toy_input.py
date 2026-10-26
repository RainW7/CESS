"""Create a tiny BayeSED-shaped interface example, not a physical galaxy library.

Rest wavelength is in microns; spec_toy is f_nu in microJy, using the simulator's
existing input convention. Line amplitudes/widths and catalogue values are
arbitrary. This exercises the software interface and cannot calibrate CSST.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def make_input(path, count=4, invalid_first=False):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('count must be a nonnegative integer')
    wave_aa = np.geomspace(100., 100000., 2079)
    flux = np.ones((count, len(wave_aa)))
    for center in (3727., 4861., 5007., 6563.):
        flux += 20. * np.exp(-.5 * ((wave_aa - center) / 12.) ** 2)
    if invalid_first and count:
        flux[0] = np.nan
    z = np.linspace(.2, .35, count)
    columns = ['RA', 'DEC', 'MAG_G', 'MAG_R', 'MAG_Z', 'log(M*)[0,1]_{MAL}', 'z_{MAL}']
    parameters = np.column_stack([np.full(count, 120.), np.full(count, 20.),
        np.full(count, 20.5), np.full(count, 20.2), np.full(count, 20.), np.full(count, 10.), z])
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(str(path))
    with h5py.File(str(path), 'x') as file:
        file.attrs['example_only'] = True
        file.attrs['description'] = 'Arbitrary synthetic interface example; not a calibrated physical population'
        file['ID'] = np.asarray(['toy-{}'.format(i).encode() for i in range(count)], dtype='S20')
        file['parameters'] = parameters
        file['parameters_name'] = np.asarray(columns, dtype='S40')
        model = file.create_group('best_fit')
        model['wavelength_rest'] = wave_aa / 1.e4
        model['wavelength_rest'].attrs['unit'] = 'um'
        model['z'] = z
        model['spec_toy'] = flux
        model['spec_toy'].attrs['unit'] = 'uJy'
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sources', type=int, default=4)
    parser.add_argument('--invalid-first', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(make_input(args.output, args.sources, args.invalid_first))
    except (OSError, ValueError) as error:
        parser.exit(2, 'CESS example: {}\n'.format(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
