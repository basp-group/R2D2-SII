# R2D2 Algorithm

# Description
The R2D2 algorithm takes a hybrid structure between a Plug-and-Play (PnP) algorithm and a learned version of the well-know "Matching Pursuit" algorithm. Its reconstruction is formed as a series of residual images, iteratively estimated as outputs of Deep Neural Networks (DNNs) taking the previous iteration’s image estimate and associated data residual as inputs. R2D2's primary application is to solve large-scale high-resolution high-dynamic range inverse problems in radio astronomy, more specifically 2D planar monochromatic intensity imaging. 
Please refer to the following papers:

>[1] Aghabiglou, A., Chu, C. S., Dabbech, A., & Wiaux, Y. (2024). [The R2D2 deep neural network series paradigm for fast precision imaging in radio astronomy](https://researchportal.hw.ac.uk/en/publications/ultra-fast-residual-to-residual-dnn-series-for-high-dynamic-range), submitted to AJ, preprint researchportal.hw.ac.uk:94082117.
>
>[2] Aghabiglou, A., Chu, C. S., Jackson, A., Dabbech, A., & Wiaux, Y. (2023). [Ultra-fast high-dynamic range imaging of Cygnus A with the R2D2 deep neural network series](https://arxiv.org/abs/2309.03291), submitted to ApJL, preprint arXiv:2309.03291.
>
<!-- R2D2 DNN training platform and R2D2 reconstruction algorithm (real data processing excluded) -->

This repository provides a full Python implementation of the R2D2 algorithm at both training and imaging stages.

<!--
# Usage
This section describes the usage of the R2D2-RI platform for:
- Testing the DNNs
- training the DNNs (include generating appropriate data as input to DNNs)
-->

# Dependencies
PyTorch and torchvision should be installed separately by following instruction [here](https://pytorch.org/get-started/locally/) to ensure their latest version available for your CUDA version is installed. If your CUDA version is older than 11.8, then you can find the instruction [here](https://pytorch.org/get-started/previous-versions/). For example, to install PyTorch with CUDA version 11.6, use the command below:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
All other required Python packages are listed in the [requirements](requirements.txt) file. Python version `3.10` or higher is required.
   
   Install the packages using the command below:
   ``` bash
   pip install -r requirements.txt
   ```
# Input files
Both R2D2 and R3D3 series are trained specifically for images of size `512x512`, with a super resolution factor (defining the pixel-resolution) set to `1.5`, and using the data-weighting scheme Briggs. To run the R2D2 algorithm, data and DNN files are required.

### VLA-trained DNN series 
The VLA-trained R2D2 and R3D3 DNN series are available at the DOI:[10.17861/99cbe654-5071-4625-b59d-a26c790cbeb4](https://researchportal.hw.ac.uk/en/datasets/r2d2-deep-neural-network-series-for-radio-interferometric-imaging). DNN checkpoints need to be saved in a desired path `$CHECKPOINT_DIR`.  
- For R2D2, the checkpoints of its `$I` U-Net components are structured inside `$CHECKPOINT_DIR` as follows:
  ``` bash
  $CHECKPOINT_DIR'/R2D2_UNet_N1.ckpt'
  $CHECKPOINT_DIR'/R2D2_UNet_N2.ckpt'
  ..
  $CHECKPOINT_DIR'/R2D2_UNet_N'$I'.ckpt'
  ```
  
- For R3D3, the checkpoints of its `$I` R2D2-Net components, each composed of `$J` network layers, are structured inside `$CHECKPOINT_DIR` as follows:
  ``` bash
   $CHECKPOINT_DIR'/R3D3_R2D2Net_'$J'Layers_N1.ckpt'
   $CHECKPOINT_DIR'/R3D3_R2D2Net_'$J'Layers_N2.ckpt'
   ..
   $CHECKPOINT_DIR'/R3D3_R2D2Net_'$J'Layers_N'$I'.ckpt'
  ```
### Data (measurement) file
The current code takes as input data a measurement file in ``.mat`` format containing the following fields:

 ``` matlab 
   "y"               %% vector; data (Stokes I)
   "u"               %% vector; u coordinate (in units of the wavelength)
   "v"               %% vector; v coordinate (in units of the wavelength)
   "w"               %% vector; w coordinate (in units of the wavelength)
   "nW"              %% vector; inverse of the noise standard deviation 
   "nWimag"          %% vector; square root of the imaging weights if available (Briggs or uniform), empty otherwise
   "frequency"       %% scalar; observation frequency
   "maxProjBaseline" %% scalar; maximum projected baseline (in units of the wavelength; formally  max(sqrt(u.^2+v.^2)))
   ```

- **Notes:**
  - An example measurement file ``data_3c353.mat`` is provided in the folder ``$R2D2RI/data/3c353/``. 
  - Briggs weights are generated using the [WSClean software](https://wsclean.readthedocs.io/en/latest/) with the Briggs parameter set to `0`.
  - To extract the data file from Measurement Set Tables (MS), you can use the utility Python script `$R2D2RI/pyxisMs2mat/pyxis_ms2mat.py`. Full instructions are available in the [ReadMe File](https://github.com/basp-group-private/R2D2-RI/blob/main/pyxisMs2mat/ReadMe.md).

### Groundtruth file
The groundtruth file `$GT_FILE` is in `.fits` format. The file is optional, used to compute the reconstruction evaluation metrics.


# Imaging / Test stage
The R2D2 algorithm (R2D2/R3D3) can be run using the following command. The final reconstructions which consist of the image estimate and associated residual dirty image are saved in `$RESULTS_DIR`. The intermediate reconstructions can also be saved by using the `--save_all_outputs` argument.
``` python
python3 ./src/run_series.py \   
--data_file $DATA_FILE \       # Path to the input .mat data file.
--ckpt_path $CHECKPOINT_DIR \  # Path to the directory of the DNN checkpoints.
--output_path $RESULTS_DIR \   # Path to the final fits files.
--series $INCARNATION \        # Incarnation of the R2D2 algorithm: "R2D2" or "R3D3".
--num_iter $I \                # Number of DNNs in the R2D2/R3D3 series
--layers $J \                  # Number of network layers in the DNN architecture. Currently acceptable values 1, 3, 6.
--super_resolution 1.5 \       # Super resolution factor.
--im_dim_x 512 \               # Image width.
--im_dim_y 512 \               # Image height.
--save_all_outputs \           # (optional) Save all intermediate outputs, otherwise only final iteration results will be saved.
--gdth_file $GT_FILE \         # (optional) Path to the ground truth fits file.
--target_dynamic_range $DR \   # (optional) Target dynamic range for computation of logSNR metric.
--res_on_gpu \                 # (optional) Compute residual dirty images on GPU to significantly accelerate overall imaging time.
--operator_type $OP_TYPE       # (optional) choose from [table, sparse_matrix], default to `table` which is faster, `sparse_matrix` is relatively more accurate.
```
- **Notes:** 
   - The parameter `layers` takes different values depending on the considered incarnation of the R2D2 algorithm.
     -  R2D2 series: set `layers` to `1`.
     -  R3D3 series: set `layers` to a value higher than `1`. For the trained R3D3 realizations, `layers` can take the value `J=3` or `J=6`.

   - To run the first term in the R2D2 (respectively, R3D3) series which corresponds to the end-to-end DNN U-Net (respectively, R2D2-Net) set `num_iter` to `I=1`.
   - The parameter `target_dyanamic_range` (`$DR`) is optional and is used to compute the logSNR metric when the groundtruth image is available.

   - Examples are provided in [`./example`](example).

 # Training
 The instruction on training will be available soon.

<!--, where:
- **series:** Option to run R2D2 or R3D3.
- **num_iter**: Number of iterations/terms of the R2D2/R3D3 series. When set to 1, it runs using the first DNN of the R2D2/R3D3 series (U-Net/R2D2-Net).
- **super_resolution:** Super resolution factor.
- **im_dim_x & im_dim_y:** Specify the image dimensions.
- **gdth_file**: (Optional) Path to ground truth image, required for the evaluation metrics.
- **output_path**: Path to the final reconstructions (image estimates and associated residual dirty images). 
Images will be saved in `$RESULTS_DIR` folder.-->


<!--The first iteration of R2D2 N<sup>(1)</sup> is end-to-end U-Net model which can be run as follows:
```
# declare path to data
DATA_PATH=path_to_data
SCNAME_TEST=testset_fullnumpy
UV_PATH=$DATA_PATH/uv/
DIRTY_PATH=$DATA_PATH/$SCNAME_TEST\_dirty/

# test with N1
python3 src/train.py --mode test \
--N_num 1 \
--data_path $DATA_PATH --im_size 512 \
--scname_test $SCNAME_TEST \
--res_ext _dirty --res_file_ext _dirty \
--save_output \
--rec2_ext _recN1 \
--checkpoint path_to_N1_checkpoint
```

And for running next iteration of R2D2 series N<sup>(i)</sup>, following directive should be followed:
```
python3 src/train.py --mode test \
--N_num i \
--data_path $DATA_PATH --im_size 512 \
--scname_test $SCNAME_TEST \
--res_ext _resN1 --res_file_ext _res --rec_ext _recN1 \
--save_output \
--rec2_ext _recNi \
--checkpoint path_to_N2_checkpoint
```
Where **DATA_PATH** is the path where your data is located. 
### R3D3 inference
$N_1$ (R2D2Net):
```
sh example/test_R3D3_G1.sh
```
$N_2$:
```
sh example/test_R3D3_G2.sh
```

For inference of the full R2D2 series (with 3 DNNs):
```
sh example/R2D2_series.sh
```

For inference of the full R2D2 series (with 3 DNNs, 3 layers in each DNN):
```
sh example/R3D3_series.sh
```-->
<!--
# Training stage

For simplicity, during the training process, scripts expect images instead of raw data.

### Training dataset generation
1. To initiate training, the model need training dataset include ground truth, dirty images and noise detail (if pruning will be applied).
2. To create ground truth and dirty images, we  need  path to the folder containing raw images and another folder containing UV files.
   2.1 Ground truth images will be generated by exponentiating the raw images. 

   2.2 Subsequently, dirty images will be created using the UV files and the already generated ground truth images.
       To generate dirty images for $\mathsf{N}_{\widehat{\boldsymbol{\theta}}^{(1)}}$ (i.e. the first network component in the R2D2 series, hereafter referred to as N<sup>(1)</sup>):
```
sh example/gen_dirty.sh
or 
python3 src/data_generation.py dirty \
--uv_path $UV_PATH \
--expo \
--gdth_path $GDTH_PATH \
--briggs \
--sigma0 0.001 \
--sigma_range 2e-6 1e-3 \
--output_dirty_path $OUT_DIRTY \
--output_gdth_path $OUT_GDTH
```
Where:

- **dirty:** Indicates whether the creation of dirty or residual images is specified.
- **uv_path:** Specifies the path to the folder containing UV files.
- **expo:** Determines if exponentiation is required.
- **gdth_path:** Specifies the path to the raw images or ground truth images (if exponentiation is not necessary).
- **briggs:** Specifies whether Briggs weighting is to be applied.
- **sigma0:** Represents the initial value for sigma (required for numerical solution).
- **sigma_range:** Defines the range of sigma values to obtain random noise level and dynamic range.
- **output_dirty_path:** Specifies the path to the folder where the created dirty images will be saved.
- **output_gdth_path:** Specifies the path to the folder where the exponentiated ground truth images will be saved.

   2.3 If pruning is necessary, noise details will be preserved.
   
3. The default folder names are as follows:

```
path_to_data
.
|- trainingset_fullnumpy (to be **provided by user**)
|- trainingset_fullnumpy_dirty (to be created)
|- trainingset_fullnumpy_epsilon (optional for pruning procedure, to be **provided by user**, must contain `true_noise_norm`)
|- validationset_fullnumpy (to be **provided by user**)
|- validationset_fullnumpy_dirty (to be created)
|- validationset_fullnumpy_epsilon (optional for pruning procedure, to be **provided by user**, must contain `true_noise_norm`)
|- testset_fullnumpy (to be **provided by user**)
|- testset_fullnumpy_dirty (to be created)
|- uv (to be **provided by user** in .mat file formats, containing variables `u` and `v`, and optionally `imweight` for specific weighting scheme.)
```
  3.1 The filenames of all files in the `path_to_data/uv` directory must follow the naming convention `uv_id_xxxxx.mat`, where `xxxxx` is any identifier specified by the user. The ground truth images in the `path_to_data/trainingset_fullnumpy`, `path_to_data/validationset_fullnumpy` and `path_to_data/testset_fullnumpy` must then follow the naming convention `yyyy_id_xxxxx.fits`, where `yyyy` is the identifier of the ground truth image and `xxxxx` is the same identifier as in the `uv` files. 

4. For the use of R2D2Net and R3D3, a path to the PSF corresponding to the sampling pattern used for creating the dirty images is required in the `path_to_data` directory.

5. After the creation of training and validation set, the training can be started as follows. 
For simplicity, the following example scripts only contain the necessary arguments for the specific scenario and the user is encouraged to refer to the [train.py](src/train.py) file for the full list of arguments that can be passed to the specific script. 
<!--All examples for $N_2$ can be used for any $N_i$ for $i > 2$ by changing the argument `N_num`.-->
<!--
### Training process
All incarnations of R2D2 algorithm can be trained using below-mentioned directive.
```
python3 src/train.py --mode train \
--num_epochs i \
--exp_dir /path/to/experiment/ \
--exp experiment_name \
--num_N 1 \
--layers 3 \
--data_path $DATA_PATH \
--im_size 512 \
--scname_train 'training_fullnumpy' \
--scname_val 'validation_fullnumpy' \
--res_ext _dirty or _resNi \
--res_file_ext _dirty or or _res \ 
--rec_ext _recNi \
--lr 1e-4
```
Where:

- **mode:** Specifies whether it's the training or testing step.
- **num_epochs:** Specifies the number of epochs needed to train the model at each iteration of the R2D2 series.
- **exp_dir:** Represents the experiment directory.
- **exp:** Refers to the experiment name.
- **num_N:** Indicates the iteration of the R2D2 series. 
When set to 1,
  it runs the initial iteration of the R2D2/R3D3 series as either an end-to-end U-Net model or an unrolled R2D2-Net,
  respectively.
- **layers:** Parameter offering two options:
  - Setting it to 1 will execute the R2D2 series using the end-to-end U-Net DNN.
  - Setting it to more than 1 will execute the unrolled R2D2-Net with "i" layers of U-Net DNN.
- **data_path:** Path to the training and validation set folder.
- **im_size:** The pixel size of the training and validation images.
- **scname_train:** The name of the folder that includes the training ground truth images. By default, set to "training_fullnumpy" but can be changed to a desired name.
- **scname_val:** The name of the folder that includes the validation ground truth images. By default, set to "validation_fullnumpy" but can be changed to a desired name.
- **res_ext:** The folder extension for the folder containing the residual dirty images.
`_dirty` for folder of dirty images and `_resNi` for folder of residual images. 
- **res_file_ext:** The file extension for the residual dirty images.
`_dirty` for dirty images and `_res` for residual images.
- **rec_ext:** The folder extension for the folder containing the estimated images. Not needed for first iteration of R2D2 series.
- **lr:** Learning rate.

**Note:** Examples are provided for each scenario and corresponding scripts are available in the [example](example) directory. Each example script is a template that can be modified to suit the specific requirements of the user. 


### Residual dirty image generation
To generate residual dirty images for $\mathsf{N}_{\widehat{\boldsymbol{\theta}}^{(i)}}$ for any $i > 1$ (i.e. any network component after the first in the series, hereafter referred to as N<sup>(i)</sup>):
```
python3 src/data_generation.py residual \
--uv_path $UV_PATH \
--rec_path $REC_PATH \
--dirty_path $DIRTY_PATH \
--output_res_path $OUT_RES \
--prune \
--epsilon_path $EPS_PATH \
```
Where:

- **residual:** Indicates whether the creation of dirty or residual images is specified.
- **uv_path:** Specifies the path to the folder containing UV files.
- **rec_path:** Specifies the path to the folder containing previous model estimations.
- **dirty_path:** Specifies the path to the folder containing dirty images.
- **output_res_path:** Specifies the path to the folder where the created residual images will be saved.
- **prune:** Indicates whether the pruning process will be applied or not.
- **epsilon_path:** Specifies the path for folder containing files with noise information

-->

