# CSST-grism-emulator
Some work about the CSST grism emulator

# Abstract
The accuracy of spectroscopic redshift measurement is essential for cosmic surveys since cosmic parameters are extremely sensitive to the purity and completeness of spectroscopic redshift. 
Therefore, the slitless spectroscopic redshift survey of the China Space Station Telescope (CSST) requires large simulated data to test the dependence. 
In order to investigate these key points, we established an emulator based on empirical relations of small samples which is capable of quickly generating a hundred million simulated 1-D slitless spectra. 
We introduce the scientific goals, design concepts, instrumental parameters, and computational methods used in the emulator, as well as how the simulated CSST slitless spectra are generated. 
The self-blending effects caused by galaxy morphological parameters, e.g., Sers\'{\i}c ($n$), effective radius ${\rm (R_e)}$, position angle (${\rm PA}$), and axial ratio (${\rm b/a}$) on the 1-D slitless spectra are considered in our simulation. 
In addition, we also develop an algorithm to estimate the overlap contamination rate of our mock data in the dense galaxy clusters. 
With a high-resolution mock galaxy spectra library of $sim$ 140 million samples generated from DESI DR9, we obtained the corresponding simulated CSST slitless spectra with our emulator. 
Our results indicate that these mock spectra data can be used to study the dependence of measurement errors on different types of galaxy redshifts due to instrument and observation effects. 
Furthermore, we are able to analyze the feasibility of the CSST slitless spectroscopic redshift survey and offer reasonable observation strategies and constraints for constructing the mock galaxy catalog. 

# Flowchart
![Flowchart of CSST grism emulator](https://github.com/RainW7/CSST-grism-emulator/blob/main/flowchart.png)

----------23/07/19 update----------

Main files for emulator version 0.7 is uploaded！
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
