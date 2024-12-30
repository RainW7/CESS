# Manual for CESS (version 0.8.5, 2024/12/30)

## 1. Download

There are two ways to download the latest version of CESS to the local environment. 

### 1.1 'git' command

Using the 'git' command to clone the whole package.

```shell
git clone https://github.com/RainW7/CESS.git
```

### 1.2 Release

The CESS GitHub repository is at https://github.com/RainW7/CESS. Click the 'Release' at the right of the website and download the latest CESS. Unpacking the package to your path, like '~/CESS_<version>/'.

## 2. Setup

Before running the executable script of CESS, there are three steps to set the CESS input parameters.

### 2.1 Create the morphological parameters libraries 

Using the 'create_lib.py' file in '~/CESS_<version>/basic_file/' to create the two morphological parameters libraries -- 'widthlib.pkl' and 'heightlib.pkl'. A detailed description of the morphological parameters library can be found in Section 3.3 in [Wen et al. 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2770W/abstract). 

The default libraries contain 20 *n* in the range of [0.5, 5], 20 *R*_e in the range of [0.3, 7.4], 10 PA in the range of [0◦, 90◦] with an interval of 10◦, and 10 *b*/*a* in the range of [0, 1] with an interval of 0.1. 

Users can costume their morphological parameters libraries by updating the parameter series in 'create_lib.py'.

To create the morphological parameters libraries, using command:

```python
python ~/CESS_<version>/basic_file/create_lib.py
```



<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1732527701154.png" alt="QQ_1732527701154" style="zoom:75%;" />

### 2.2 Update the 'emulator_parameters.json'

The beginning lines of the 'emulator_parameters.json' file are working directories of paths, and users should update the paths for their environment. 

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1732709756011.png" alt="QQ_1732709756011" style="zoom:75%;" />

'input_path' and 'output_path' are the paths of the input model spectra library (see appendix for details of input_model_spectra.hdf5 file) and the output CESS simulated spectra library.

'bkg_spec_path' and 'gems_path' are the paths of the sky background file and GEMS morphological catalog file (see appendix for details).

'widthlib_pkl' and 'heightlib_pkl' are the paths of the morphological libraries (see Section 2.1). 

'g*tp_path' are the paths of filter throughputs of CSST slitless spectroscopic bands.

'*tp_path' are the paths of filter throughputs of CSST photometric bands.

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1732793274403.png" alt="QQ_1732793274403" style="zoom:75%;" />

In the middle of the 'emulator_parameters.json' file are the CSST slitless spectroscopy equipment parameters to date. Users can update the resolution, central wavelength, delta lambda, and R_EE80 of PSF (in arcsec) of three slitless bands. The current values are based on the latest CSST instrumental parameters. The 'g*_start/end/min/max_wl' means the starting wavelength of convolving, the ending wavelength of convolving, the lower wavelength cut of CSST slitless band, and the upper wavelength cut of the CSST slitless band of three bands, respectively. 

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1733820501466.png" alt="QQ_1733820501466" style="zoom:75%;" />

The last lines describe the parameters of the emission line detection module. Currently, only the GV and GI band detects emission line information. Due to the low transmission at both ends of the filter, CESS will skip a small part at the ends and only detect the middle part of the simulated slitless spectrum. The parameters (e.g., detection threshold) used for emission line detection are also listed here.

### 2.3 Fill the input file variable

The input filenames should be added into the 'run.py' file at the ending lines (e.g., starting from line 759 in run.py file of CESS_0.8.5). Currently, we offer a simple way to collect and add the filenames with the collecting script 'glob_seedcat.py'. Users can display all the input filenames at the terminal and copy them to the 'run.py'.

```python
python glob_seedcat.py <input_file_path>
```

The detailed output example in terminal is here:

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1735280267920.png" alt="QQ_1735280267920" style="zoom:75%;" />

Copy and paste the output filenames into the 'hdf5filenames' variable:

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1735280543958.png" alt="QQ_1735280543958" style="zoom:75%;" />



## 3. Run  

### 3.1 The running command

After the setup, CESS can be run with the command

```py
python run.py <NUM1> <NUM2> morph=yes wave_cal=yes photo=yes el_detect=yes
```

The <NUM1> and <NUM2> here stands for the staring index and ending index of the 'hdf5filenames' variable. The expected computing source of one input file is one CPU and about 10GB memory, thus choosing a suitable <NUM1> and <NUM2> that fits the user's environment to avoid 'ArrayMemoryError' during the run.

Addtionally, the running script sets four calculation switches, which are controlled by the keywords in the running command. The 'morph' means adding morphological effects, 'wave_cal' means adding wavelength calibration error effects, 'photo' means using the photometric calculation module and 'el_detect' means using the emission line detection module. These calculation modules are off in default, users should set them with 'true' or 'yes' to switch on. 

<img src="/Users/rain/Library/Application Support/typora-user-images/image-20241227153702108.png" alt="image-20241227153702108" style="zoom:75%;" />

### 3.2 The output file

The output file of CESS is named as 'CSST_grism\_<input_file_name>\_<length_of_the_file>.hdf5 ', e.g., 

```shell 
CSST_grism_seedcat2_0420_727_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10_23360.hdf5
```



## 4. Check output

CESS offers two ways 'demo.py' and 'demo.ipynb' to show the results of the output file, including showing simulated spectrum figures and hdf5 file structures.

### 4.1 To display the spectrum

The 'demo.py' file is used to display the simulated CSST grism spectrum with detailed parameters and original model spectrum. The command is:

```python
python demo.py <CESS_output_file_path> <NUM or random> <emissionline or none>
```

Where the <CESS_output_file_path> is the path of CESS output file, <NUM or random> means a certain index within the length of CESS output file at the end (e.g., 23360 for 727_MzLS) or a random index using <random>, and <emissionline or none> means showing intrinsic emission line information in the figure or not. 

<img src="/Users/rain/Library/Application Support/typora-user-images/image-20241229205822785.png" alt="image-20241229205822785" style="zoom:75%;" />

<img src="/Users/rain/Library/Application Support/typora-user-images/image-20241229205851201.png" alt="image-20241229205851201" style="zoom:75%;" />

### 4.2 To show the file structure

The Jupyter file 'demo.ipynb' is used to show the detailed CESS input, output file structure and simulated spectra with simulated photometric data. 

The input file (e.g., <seedcat2_0420_727_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5>) structure is like:

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1735546111115.png" alt="QQ_1735546111115" style="zoom:75%;" />

While the output file (e.g., <CSST_grism_23514_seedcat2_0420_1204_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5>) structure is like:

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1735547529940.png" alt="QQ_1735547529940" style="zoom:75%;" />

A demo for simulated CSST slitless spectrum with photometric data (conversion functions for AB mag and flux in μJy is offered) is like:

<img src="/Users/rain/Library/Containers/com.tencent.qq/Data/tmp/QQ_1735547965036.png" alt="QQ_1735547965036" style="zoom:75%;" />