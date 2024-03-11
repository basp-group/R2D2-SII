# R2D2 Algorithm

# Description
The R2D2 algorithm takes a hybrid structure between a Plug-and-Play (PnP) algorithm and a learned version of the well-know "Matching Pursuit" algorithm. Its reconstruction is formed as a series of residual images, iteratively estimated as outputs of Deep Neural Networks (DNNs) taking the previous iteration’s image estimate and associated data residual as inputs. R2D2's primary application is to solve large-scale high-resolution high-dynamic range inverse problems in radio astronomy, more specifically 2D planar monochromatic intensity imaging. 
Please refer to the following papers:

>[1] Aghabiglou, A., Chu, C. S., Dabbech, A., & Wiaux, Y. (2024). [The R2D2 deep neural network series paradigm for fast precision imaging in radio astronomy](https://researchportal.hw.ac.uk/en/publications/ultra-fast-residual-to-residual-dnn-series-for-high-dynamic-range), submitted to ApJ, preprint arXiv:2403.05452.
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

