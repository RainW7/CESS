The main files of emulator version 0.7
create_lib.py is used to create the 2-D profile library (``height'' and ``width'' profile) and stored in .pkl files (which is very convenient for saving numpy.array.
demo.py is used to create a demo figure for the simulated spectra of the CSST emulator, see demo figures for a review. 
emulator_parameters.json is a JSON file that stores the CSST instrumental parameters, observational strategies, and algorithm settings. 
main.py includes the convolution of the intrinsic spectra, the noise simulation of the noisy spectra (including the wavelength error and broadening effect), and the emission line detection process in both intrinsic and noisy spectra.
morphology.py includes the 2-D morphological parameter fitting process based on the empirical correlation in GEMS morphological catalog, the ``height'' and ``width'' profile extraction, and the pre-established library matching. 
run.py is the file for running the emulator and storing the data in the form of hdf5.
utils.py is a collection of some useful functions. 
