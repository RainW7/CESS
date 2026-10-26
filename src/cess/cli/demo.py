#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot CESS spectra using named tables and explicit input-model provenance.

The legacy positional interface is preserved:
    demo.py output.hdf5 INDEX_OR_random emissionline_OR_none
Add --save figure.png for noninteractive rendering. Importing performs no I/O.
"""
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from cess.io.parameters import read_parameter_table
from cess.io.products import open_product

GBAND_UM, RBAND_UM, ZBAND_UM = .4954320432589493, .6493723922982192, .9248643145995256


def _text(value):
    return value.decode('utf-8') if isinstance(value, (bytes, np.bytes_)) else str(value)


def mAB2uJy(mag):
    return np.power(10., (np.asarray(mag) - 23.9) / -2.5)


def _table_row(file, name, index, required=()):
    table = read_parameter_table(file[name], ids=file['ID'], required=required)
    return {column: table[column][index] for column in table.names}


def _input_model(data, model_input=None, config=None):
    """Follow input_product + input_row_indices; never infer from chunk filenames."""
    provenance = data['input_product']
    candidates = []
    if model_input is not None:
        candidates.append(Path(model_input).expanduser())
    elif provenance:
        source = Path(provenance).expanduser()
        candidates.append(source if source.is_absolute() else data['path'].parent / source)
        if config is not None:
            from cess.config.loader import load_config
            parameters = load_config(config)
            candidates.append(Path(parameters['input_path']) / source.name)
    available = next((path for path in candidates if path.is_file()), None)
    if available is None:
        return None, 'Input model unavailable; simulated spectra only. Use --model-input to supply it.'
    try:
        from astropy import units as u
        with h5py.File(str(available), 'r') as file:
            if 'ID' not in file or 'best_fit' not in file:
                raise ValueError('input is not an identified BayeSED model catalogue')
            ids = file['ID']
            row = data['input_row']
            if row is None:
                matches = [i for i, value in enumerate(ids[:]) if _text(value) == _text(data['id'])]
                if len(matches) != 1:
                    raise ValueError('legacy source ID is missing or ambiguous in the model catalogue')
                row = matches[0]
            if row < 0 or row >= len(ids) or _text(ids[row]) != _text(data['id']):
                raise ValueError('input row/ID provenance does not match the model catalogue')
            group = file['best_fit']
            names = sorted(name for name in group if name.startswith('spec_'))
            if not names:
                raise ValueError('model catalogue has no spec_* dataset')
            wavelength = group['wavelength_rest']
            flux = group[names[0]]
            redshift = float(group['z'][row])
            # BayeSED defaults match the simulator contract; explicit units win.
            rest_um = (wavelength[:] * u.Unit(_text(wavelength.attrs.get('unit', 'um')))).to_value(u.um)
            flux_ujy = (flux[row] * u.Unit(_text(flux.attrs.get('unit', 'uJy')))).to_value(u.uJy)
            if (rest_um.ndim != 1 or flux_ujy.shape != rest_um.shape or
                    not np.isfinite(redshift) or redshift <= -1 or
                    np.any(~np.isfinite(rest_um)) or np.any(np.diff(rest_um) <= 0) or
                    np.any(~np.isfinite(flux_ujy))):
                raise ValueError('model wavelength/flux/redshift is invalid')
            return dict(path=available.resolve(), row=row, wave_um=rest_um * (1 + redshift),
                        flux_ujy=flux_ujy, redshift=redshift), 'Input model: {}, row {}'.format(available, row)
    except (OSError, KeyError, ValueError, TypeError) as error:
        return None, 'Input model unavailable: {}. Simulated spectra only.'.format(error)


def read_demo_data(path, index=0, model_input=None, config=None):
    """Read one source; all HDF5 handles close before plotting."""
    path = Path(path).expanduser().resolve()
    with open_product(path) as product:
        file = product.legacy_view()
        count = len(file['ID'])
        if count == 0:
            raise ValueError('Cannot plot a catalogue with zero sources.')
        if index == 'random':
            index = int(np.random.randint(count))
        else:
            index = int(index)
        if not 0 <= index < count:
            raise ValueError('Source index must be in [0, {}).'.format(count))
        desi = _table_row(file, 'parameters_desi', index)
        grism = _table_row(file, 'parameters_grism', index,
                          required=('gu_snr_mean', 'gv_snr_mean', 'gi_snr_mean'))
        phot = _table_row(file, 'parameters_phot', index) if 'parameters_phot' in file else None
        input_row = None
        if 'input_row_indices' in file:
            mapping = file['input_row_indices']
            if mapping.shape != (count,) or mapping.dtype.kind not in 'iu':
                raise ValueError('input_row_indices must contain one integer per output ID.')
            input_row = int(mapping[index])
            if input_row < 0:
                raise ValueError('input_row_indices must be nonnegative.')
        data = dict(path=path, index=index, id=file['ID'][index], desi=desi,
                    grism=grism, phot=phot, input_row=input_row,
                    input_product=_text(file.attrs.get('input_product', '')), bands={})
        for band in ('GU', 'GV', 'GI'):
            group = file[band]
            wave = np.asarray(group['wave'][:], dtype=float)
            model = np.asarray(group['flux_ujy'][index], dtype=float)
            observed = np.asarray(group['flux_ujy_with_noise'][index], dtype=float)
            if wave.ndim != 1 or model.shape != wave.shape or observed.shape != wave.shape:
                raise ValueError('{} spectral arrays do not match the wavelength grid.'.format(band))
            data['bands'][band] = dict(wave_um=wave / 1.e4, model=model, observed=observed)
    data['model'], data['model_note'] = _input_model(data, model_input, config)
    return data


def create_figure(data, mode='none'):
    """Retain the original black models, blue/green/red spectra and log axes."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    if data['model'] is not None:
        model = data['model']
        ax.plot(model['wave_um'], model['flux_ujy'], color='silver', linewidth=1.,
                alpha=.8, label='Input model SED', zorder=1)
    for order, (band, color) in enumerate(zip(('GU', 'GV', 'GI'), ('dodgerblue', 'limegreen', 'red'))):
        values = data['bands'][band]
        ax.plot(values['wave_um'], values['model'], c='k', linewidth=1.5,
                label=band + ' SED at obs_wave', zorder=6 + order)
        ax.plot(values['wave_um'], values['observed'], color=color, alpha=.8,
                label=band + ' observed flux', zorder=3 + order)
    parameters = data['desi']
    mags = np.asarray([parameters.get(name, np.nan) for name in ('MAG_G', 'MAG_R', 'MAG_Z')])
    ax.scatter([GBAND_UM, RBAND_UM, ZBAND_UM], mAB2uJy(mags), marker='^',
               color='black', s=80, label='DESI observed flux', alpha=.7, zorder=11)
    positives = np.concatenate([values['model'][np.isfinite(values['model']) & (values['model'] > 0)]
                                for values in data['bands'].values()])
    low, high = (float(positives.min()) / 2, float(positives.max()) * 10) if positives.size else (1.e-4, 1.)
    redshift = parameters.get('z_best', np.nan)
    if mode == 'emissionline' and np.isfinite(redshift):
        for wavelength, label in ((3729., '[OII]'), (4861., 'Hb'), (5007., '[OIII]'),
                                  (6563., 'Ha'), (6717., '[SII]')):
            position = (1 + redshift) * wavelength / 1.e4
            ax.axvline(position, linestyle='--', zorder=10)
            ax.text(position, high / 2, label, zorder=10)
    ax.set_yscale('log')
    ax.set_xlim(.2, 1.1)
    ax.set_ylim(low, high)
    ax.set_xlabel(r'observed wavelength [$\mu m$]' if mode == 'emissionline' else r'wavelength [$\mu m$]')
    ax.set_ylabel('fnu [uJy]')
    ax.legend(loc='upper left')
    def display(name, precision=2):
        value = parameters.get(name, np.nan)
        return format(float(value), '.' + str(precision) + 'f') if np.isfinite(value) else 'n/a'
    snr = data['grism']
    fig.suptitle('File: {}, index = {}, DESI_ID = {}, (RA, DEC) = ({}, {}), z = {}\n'
                 'MAG_G = {}, MAG_R = {}, MAG_Z = {}\n'
                 'Sersic = {}, Re = {} pix, PA = {} deg, b/a = {}, mass = 10^{} M$_\\odot$\n'
                 'GU_SNR = {:.2f}, GV_SNR = {:.2f}, GI_SNR = {:.2f}'.format(
                     data['path'].name, data['index'] + 1, _text(data['id']), display('RA'), display('Dec'),
                     display('z_best'), display('MAG_G'), display('MAG_R'), display('MAG_Z'),
                     display('n'), display('Re'), display('PA'), display('baratio'), display('str_mass', 1),
                     snr['gu_snr_mean'], snr['gv_snr_mean'], snr['gi_snr_mean']), fontsize=9)
    if data['model'] is None:
        ax.text(.01, .015, 'Input model unavailable; simulated spectra only.', transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('product', type=Path)
    parser.add_argument('index', help='zero-based source index, or random')
    parser.add_argument('mode', choices=('emissionline', 'none'))
    parser.add_argument('--save', type=Path, help='save this figure without opening a GUI')
    parser.add_argument('--model-input', type=Path, help='explicit BayeSED input catalogue')
    parser.add_argument('--config', type=Path, help='resolve a relocated input_product basename under input_path')
    args = parser.parse_args(argv)
    try:
        if args.save is not None:
            import matplotlib
            matplotlib.use('Agg')
        data = read_demo_data(args.product, args.index, args.model_input, args.config)
        figure = create_figure(data, args.mode)
        destination = args.save or Path('{}_{}.png'.format(data['path'].name, data['index'] + 1))
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(destination), dpi=200)
        print(data['model_note'])
        print('Saved spectrum ID={} to {}'.format(_text(data['id']), destination))
        import matplotlib.pyplot as plt
        if args.save is None:
            plt.show()
        plt.close(figure)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print('CESS demo: {}'.format(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
