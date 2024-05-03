## Requirements
Python-casacore: https://github.com/casacore/python-casacore
## General notes
1. Data extraction is performed in `./ms2mat` which contains the python script `./ms2mat/ms2mat.py`.
2. Extracted `.mat` file is saved in `../data/`. The file contain the following fields:
``` matlab
   "frequency"       % channel frequency
   "y"               % data (Stokes I)
   "u"               % u coordinate (in units of the wavelength)
   "v"               % v coordinate (in units of the wavelength)
   "w"               % w coordinate (in units of the wavelength)
   "nW"              % inverse of the noise standard deviation 
   "nWimag"          % square root of the imaging weights if available (Briggs or uniform), empty otherwise
   "maxProjBaseline" % maximum projected baseline (in units of the wavelength)
   ```    
3. The function in `ms2mat.py` is an example. The user is encouraged to tailor it if the measurement set is organised differently.

## Example
Extracting (monochromatic) data at the frequency channel  `0` of the source with field ID `0`.

The user must provide the name/path to the measurement set `--msfile`. The following inputs are optional.
```bash
--srcname  # default srcname="" : source nametag which defines the main directory of the extracted data. 
--srcid    # default srcid=0 : field ID of the source of interest, 
--freqid   # default freqid=0 : ID of the channnel to be extracted, 
```

From the terminal launch:
```bash
python ms2mat.py --msfile=myms.ms --srcname=3c353 --srcid=0 --freqid=0  
```

Data will be saved as .mat files in the directory `../data/`. The outcome is as follows
```bash
../data/3c353_data_ch_1.mat
```
