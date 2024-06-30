# CSST Emulator for Slitless Spectroscopy (CESS)
Some work about the emulator for CSST slitless spectroscopy.

# Abstract
The Chinese Space Station Telescope (CSST) slitless spectroscopic survey will observe objects to a magnitude limit of ~ 23 mag (5σ, point sources) in U, V, and I over 17,500 square degrees.  The spectroscopic observations are expected to be highly efficient and complete for mapping galaxies over 0<z<1 with secure redshift measurements at spectral resolutions of R ~ 200, providing unprecedented datasets for cosmological studies.  To examine the survey potential in a quantitative manner, we develop a software tool, namely the CSST Emulator for Slitless Spectroscopy (CESS), to quickly generate simulated one-dimensional slitless spectra with limited computing resources.  

# Flowchart
![Flowchart of CSST grism emulator](https://github.com/RainW7/CSST-grism-emulator/blob/main/flowchart.png)

# result file data structure
How to show the data structure: 
```python
import h5py
file = h5py.File('/Users/rain/emulator/seedcat_0702/seedcat2_0702_0_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5','r')
for name in file1:
    print(name)
    if isinstance(file1[name], h5py.Group):
        for subname in file1[name]:
           print(f"  {subname}")
```
How to show column names of parameters: 
```python
file['parameters'].attrs['name'],
array(['RA', 'Dec', 'z_best', 'MAG_G', 'MAG_R', 'MAG_Z', 'n', 'Re', 'PA',
       'baratio', 'str_mass', 'gu_rms_in_e', 'gv_rms_in_e', 'gi_rms_in_e',
       'gu_snr_mean', 'gv_snr_mean', 'gi_snr_mean', 
       'gu_wave_off', 'gv_wave_off', 'gi_wave_off', #wavelength error
       'gv_el_flag', 'gi_el_flag'], dtype=object) #emission line flags,
  #0 = no el detection, 1 = only intrinsic el detection, 2 = both intrinsic and noisy el detection
```
Detailed structures: 
```
'ID' # ids of each source in the DESI photometry catalog
'parameters' # paramters of each source, in array
```
```
'GU' # hdf5 group
├── wave
├── flux_ujy
├── flux_ujy_with_noise
├── ferr
├── flux_elec
└── snr
```
```
'GV' # hdf5 group
-----simulated spectrum information-----
├── wave  # wavelength grid
├── flux_ujy # simulated CSST intrinsic slitless spectra in ujy
├── flux_ujy_with_noise # simulated CSST observed slitless spectra with noise in ujy
├── ferr # simulated CSST observed slitless spectra error in ujy
├── flux_elec # simulated CSST observed slitless spectra in electrons
├── snr # signal-to-noise ratio of each spectrum   
-----intrinsic spectrum emission line information-----
├── intri_el_id # ids for source detected with emission lines in intrinsic spectra, i.e., 'flux_ujy'
├── intri_el_wave # arrays of the emission line wavelengths in intrinsic spectra
├── intri_el_idx # arrays of the emission line wavelength grid corresponding index in intrinsic spectra
├── intri_el_elnumber # numbers of detected emission lines in intrinsic spectra    
-----noisy spectrum emission line information-----
├── detect_el_id # ids for source detected with emission lines in noisy spectra, i.e., 'flux_ujy_with_noise'
├── detect_el_wave # arrays of the emission line wavelengths in noisy spectra
├── detect_el_idx # arrays of the emission line wavelength grid corresponding index in noisy spectra
├── detect_el_elnumber # numbers of detected emission lines in noisy spectra
└── detect_el_snr # mean snr of detected emission lines in noisy spectra
```
```
'GI' # hdf5 group
├── wave
├── flux_ujy
├── flux_ujy_with_noise
├── ferr
├── flux_elec
├── snr
├── intri_el_id
├── intri_el_wave
├── intri_el_idx
├── intri_el_elnumber
├── detect_el_id
├── detect_el_wave
├── detect_el_idx
├── detect_el_elnumber
└── detect_el_snr
```

----------24/06/30 update----------
1. Update the emulator version 0.8.4.
2. Add photometric magnitude (AB mag) estimation for CSST photometry.
3. Update the kernel convolution function following the ```varsmooth``` function in ```pPXF``` package from https://pypi.org/project/ppxf/ by Cappellari

----------23/10/27 update----------

1. Update the emulator version 0.8.
2. For those sources with error data, the emulator creates a nan spectrum now. 

----------23/08/06 update----------

1. Divided the files into 3 documents, the basic files used for the emulator, the main Python files of the emulator, and some of the used jupyter notebooks.
2. Uploaded some jyputer notebooks that I created and used during the development of the emulator.

----------23/07/19 update----------

Main files for emulator version 0.7 are uploaded！
1. main.py for the convolution for the CSST intrinsic slitless spectra, simulating observed slitless spectra and emission line detection.
2. morphology.py for the 2-D parameters fitting and 2-D profile distribution function extraction.
3. utils.py for some small functions.
4. runtest.py for the running code file in the local environment, i.e., my MacBook.
5. More details needed to be updated.

----------23/03/25 update----------

redshift range figure updated: 
1. using redshift data in the loop instead of creating and loading files.
2. setting z = 0.5 x_label as a float while other labels are int using FuncFormatter
3. hide the y-axis major label instead of setting the color as 'white'

----------21/12/02 update----------

redshift range figure updated with customized x-axis scale.

----------21/08/25 update----------

Jupyter notebook of emission lines redshift range in 3 CSST grism band.
