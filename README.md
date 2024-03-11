# R2D2 Algorithm

# Description
The R2D2 algorithm takes a hybrid structure between a Plug-and-Play (PnP) algorithm and a learned version of the well-know "Matching Pursuit" algorithm. Its reconstruction is formed as a series of residual images, iteratively estimated as outputs of Deep Neural Networks (DNNs) taking the previous iteration’s image estimate and associated data residual as inputs. R2D2's primary application is to solve large-scale high-resolution high-dynamic range inverse problems in radio astronomy, more specifically 2D planar monochromatic intensity imaging. 
Please refer to the following papers:

>[1] Aghabiglou, A., Chu, C. S., Dabbech, A., & Wiaux, Y. (2024). [The R2D2 deep neural network series paradigm for fast precision imaging in radio astronomy](https://researchportal.hw.ac.uk/en/publications/ultra-fast-residual-to-residual-dnn-series-for-high-dynamic-range), submitted to AJ, preprint researchportal.hw.ac.uk:94082117.
>
>[2] Aghabiglou, A., Chu, C. S., Jackson, A., Dabbech, A., & Wiaux, Y. (2023). [Ultra-fast high-dynamic range imaging of Cygnus A with the R2D2 deep neural network series](https://arxiv.org/abs/2309.03291), submitted to ApJL, preprint arXiv:2309.03291.
>

This repository provides a MATLAB implementation of the R2D2 algorithm.

# Installation

### Cloning the project
To clone the project, you may consider one of the following set of instructions.

- Cloning the project using `https`
```bash
git clone -b matlab https://github.com/basp-group/R2D2-RI.git
```
- Cloning the project using SSH key for GitHub
```bash
git clone -b matlab git@github.com:basp-group/R2D2-RI.git
```
The full path to this repository is referred to as `$R2D2RI` in the rest of the documentation.

### MATLAB Dependencies
To build the radio-interferometric measurement operator, the repository relies on the submodule
   [`RI-measurement-operator`](https://github.com/basp-group/RI-measurement-operator), implemented in MATLAB. Codes are associated with the following papers:
   
> [3] Fessler, J. A., & Sutton, B. P. (2003). Nonuniform fast Fourier transforms using min-max interpolation. *IEEE TSP*, 51(2), 560-574.
>
> [4] Onose, A., Dabbech, A., & Wiaux, Y. (2017). An accelerated splitting algorithm for radio-interferometric imaging: when natural and uniform weighting meet. *MNRAS, 469*(1), 938-949.
> 
> [5] Dabbech, A., Wolz, L., Pratley, L., McEwen, J. D., & Wiaux, Y. (2017). The w-effect in interferometric imaging: from a fast sparse measurement operator to superresolution, *MNRAS*, 471(4), 4300-4313.
      
Clone the submodule in `$R2D2RI/lib/` using the command below:

```bash
cd $R2D2RI/lib/
git clone   https://github.com/basp-group/RI-measurement-operator.git
```
# Input files
The available R2D2 and R3D3 series are trained specifically to form images of size `512 x 512` from data acquired with the Very Large Array (VLA), using a super resolution factor (defining the pixel-resolution) equal to `1.5`, and Briggs weighting. 

To run the R2D2 algorithm, data and DNN files are required.

### VLA-trained DNN series
The VLA-trained R2D2 and R3D3 DNN series are available at the DOI:[10.17861/99cbe654-5071-4625-b59d-a26c790cbeb4](https://researchportal.hw.ac.uk/en/datasets/r2d2-deep-neural-network-series-for-radio-interferometric-imaging). DNN files need to be saved in a desired path `$ONNX_DIR`.  
- For R2D2,  its `$I` U-Net components are structured inside `$ONNX_DIR` as follows:
``` bash
$ONNX_DIR'/R2D2_UNet_N1.onnx'
$ONNX_DIR'/R2D2_UNet_N2.onnx'
..
$ONNX_DIR'/R2D2_UNet_N'$I'.onnx'
```
  
- For R3D3, its `$I` R2D2-Net components are composed of `$J` network layers, structured in sub-directories inside `$ONNX_DIR`.  The U-Net layers of each R2D2-Net are saved in separate files.
``` bash
$ONNX_DIR'/R3D3_R2D2Net_'$J'Layers_N1/R3D3_R2D2Net_'$J'Layers_N1_L1.onnx'
..
$ONNX_DIR'/R3D3_R2D2Net_'$J'Layers_N1/R3D3_R2D2Net_'$J'Layers_N1_L'$J'.onnx'
$ONNX_DIR'/R3D3_R2D2Net_'$J'Layers_N2/R3D3_R2D2Net_'$J'Layers_N2_L1.onnx'
..
$ONNX_DIR'/R3D3_R2D2Net_'$J'Layers_N2/R3D3_R2D2Net_'$J'Layers_N2_L'$J'.onnx'
..
$ONNX_DIR'/R3D3_R2D2Net_'$J'Layers_N'$I'/R3D3_R2D2Net_'$J'Layers_N'$I'_L1.onnx'
..
$ONNX_DIR'/R3D3_R2D2Net_'$J'Layers_N'$I'/R3D3_R2D2Net_'$J'Layers_N'$I'_L'$J'.onnx'
```
### Input data (measurement) file
The input data file `$DATA_FILE` is expected to be in `.mat` format, with the following fields: 
   ``` matlab 
   "y"               %% vector; data (Stokes I)
   "u"               %% vector; u coordinate (in units of the wavelength)
   "v"               %% vector; v coordinate (in units of the wavelength)
   "w"               %% vector; w coordinate (in units of the wavelength)
   "nW"              %% vector; inverse of the noise standard deviation 
   "nWimag"          %% vector; square root of the imaging weights if available (Briggs or uniform), empty otherwise
   "frequency"       %% scalar; channel frequency
   "maxProjBaseline" %% scalar; maximum projected baseline (in units of the wavelength; formally  max(sqrt(u.^2+v.^2)))
   ```
- **Notes:**
    - Briggs weights are generated using the [WSClean software](https://wsclean.readthedocs.io/en/latest/) with the Briggs parameter set to `0`.
    - To extract the data file from Measurement Set Tables (MS), you can use the utility Python script `$R2D2RI/pyxisMs2mat/pyxis_ms2mat.py`. Full instructions are available in the [ReadMe File](https://github.com/basp-group-private/R2D2-RI/blob/matlab/pyxisMs2mat/ReadMe.md).

### Groundtruth file (optional)
The groundtruth file `$GT_FILE` is in `.fits` format. The file is optional, used to compute the reconstruction evaluation metrics.


# Imaging with the R2D2 algorithm
The R2D2 algorithm (R2D2/R3D3) is run in MATLAB. Input parameters are specified in the configuration file `$R2D2RI/config/$JSON_FILE` which is in  `.json` format, structured as follows:

 ``` matlab
     ``main``
            "imDimx"             % scalar; image width.
            "imDimy"             % scalar; image height.
            "dataFile"           % string; path to the input data file (.mat).
            "srcName"            % (optional) string; target source name.
            "projectDir"         % (optional) string; project directory $R2D2RI. Default: "./".
            "resultPath"         % (optional) string; results path. Default: $R2D2RI"/results".           
            "verbose"            % (optional) logical; Default: true.
            "flagSaveAllOutputs" % (optional) logical; save files output of all iterations. Default: true.

     ``measurement operator``
            "imPixelSize"          % scalar; pixel size in arcsec. If not defined, "nufftSuperresolution" is used.
            "nufftSuperresolution" % scalar; super-resolution factor (>= 1), used when the pixel size is not defined. Default: 1.
            "flagDataWeighting "   % (optional) logical; flag to indicate if imaging weights are available (Briggs or uniform). Default: false.
            "nCpus"                % (optional) scalar; number of cpu cores used in MATLAB.

     ``algorithm``
            "dnnSeries"          % string; incarnation of the R2D2 algorithm: "R2D2" or "R3D3".
            "onnxPath"           % string; path to the directory of the DNN files $ONNX_DIR.
            "nIterations"        % scalar; number of DNNs $I in the R2D2/R3D3 series
            "nlayers"            % scalar; number of network layers in the DNN architecture $J. The parameter is compulsory for R3D3. For the available series, its value should be set to 3 or 6.
            "dnnGPU"             % (optional) logical; run inference on gpu. Default: true.

      ``simulations``
            "groundtruthFile"    % (optional) string; path to groundtruth file $GT_FILE (.fits).
            "targetDynamicRange" % (optional) scalar; target dynamic range for computation of logSNR.
   ```    

To run the R2D2 algorithm run the command below in MATLAB:
``` bash
    run_imager($R2D2RI/config/$JSON_FILE);
```
**Note:** Examples of the configuration files are in `$R2D2RI/config`.

