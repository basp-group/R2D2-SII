#!/usr/bin/env python3
# Author: Taylor C.
"""
Functions to parse the configuration file in .yaml format, and validate all
arguments set from the configuration file using Pydantic model.
"""

###################################################
# imports

import os
import argparse
import pathlib
import yaml
from enum import Enum, IntEnum
from pydantic import BaseModel, ConfigDict, FilePath, DirectoryPath, field_validator
from typing import List, Optional, Union

###################################################

def parse_yaml_file():
    """
    Parse a YAML file containing configuration arguments and return the parsed arguments.

    :return: parsed argument with yaml file path.
    :rtype: argparse.Namespace
    """    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, required=True,
                    help='Path to yaml file containing all the arguments.')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--super_resolution', type=float, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--gdth_file', type=str, default=None)
    parser.add_argument('--series', type=SeriesEnum, default=None)
    parser.add_argument('--layers', type=LayersEnum, default=None)
    parser.add_argument('--num_iter', type=int, default=None)
    args = parser.parse_args()
    return args

###################################################

class SeriesEnum(str, Enum):
    """
    Enum class for the series of R2D2 and R3D3.
    """
    R2D2 = 'R2D2'
    R3D3 = 'R3D3'
    

class LayersEnum(IntEnum):
    """
    Enum class for the number of layers in the R2D2Net for the R3D3 series.
    """
    one = 1
    three = 3
    six = 6
    
class OpTypeEnum(str, Enum):
    """
    Enum class for the type of measurement operator using TorchKbNufft.
    """
    table = 'table' # table interpolation, default, faster but less accurate
    sparse_matrix = 'sparse_matrix' # precomputed sparse matrix, slower but more accurate

class DataTypeEnum(str, Enum):
    """
    Enum class for the type of data to generate.
    """
    visibilities = 'visibilities'
    residual = 'residual'
    
class WeightTypeEnum(str, Enum):
    """
    Enum class for the type of weighting to generate.
    """
    briggs = 'briggs'
    uniform = 'uniform'
    none = 'none'
    
class WeightRobustnessEnum(str, Enum):
    """
    Enum class for the type of weighting to generate.
    """
    random = 'random'
    zero = 'zero'
    
class CommonArgs(BaseModel):
    """
    Pydantic model class containing common arguments for different tasks.
    """
    model_config = ConfigDict()
    
    # algorithm 
    num_iter: Union[int, str]
    series: SeriesEnum = SeriesEnum.R2D2
    layers: LayersEnum = LayersEnum.one
    
    # measurement operator
    im_dim_x: int = 512
    im_dim_y: int = 512
    super_resolution: float = 1.5
    operator_type: OpTypeEnum = OpTypeEnum.table
    
    # imaging weight
    gen_nWimag: bool = False
    natural_weight: bool = False
    weight_type: WeightTypeEnum = WeightTypeEnum.briggs
    weight_gridsize: float = 2
    weight_robustness: WeightRobustnessEnum = WeightRobustnessEnum.zero
    weight_robustness_min: float = -1.
    weight_robustness_max: float = 1.
    
    # network architecture
    num_chans: int = 64
    num_pools: int = 4
    drop_prob: float = 0.0
    
    # miscellaneous
    verbose: int = 1
    
    # validation    
    @field_validator('layers')
    @classmethod
    def _check_layers(cls, v, values):
        if values.data['series'] == SeriesEnum.R2D2:
            if v != LayersEnum.one.value:
                # raise ValueError('R2D2 series must have only one layer.')
                print('WARNING: R2D2 series must have only one layer, this will be set automatically.')
                v = LayersEnum.one
        elif values.data['series'] == SeriesEnum.R3D3:
            if v == LayersEnum.one.value:
                raise ValueError('R3D3 series must have more than one layer, please change `layers` value in config file!')
        return v
    
    @field_validator('series', 'layers', 'operator_type')
    @classmethod
    def _return_value(cls, v):
        return v.value
    
###################################################

class InferenceArgs(CommonArgs):
    """
    Pydantic model class containing arguments specific to R2D2 inference.
    """
    
    # i/o
    ckpt_path: DirectoryPath
    data_file: FilePath
    output_path: str = './results/'
    gdth_file: Optional[FilePath] = None
    imweight_name: str = 'nWimag'
    save_all_outputs: bool = False
    
    # hardware
    res_on_gpu: bool = True
    cpus: int = 1
    
    # metrics
    target_dynamic_range: float = 0.0
    
    # validation
    @field_validator('gdth_file', mode='before')
    @classmethod
    def _check_empty(cls, v):
        if type(v) == str and len(v) == 0:
            return None
    
    @field_validator('data_file')
    @classmethod
    def _check_data_file(cls, v):
        assert str(v).endswith('.mat') or str(v).endswith('.fits'), 'The provided data_file format is not currently supported. (only .fits or .mat are supported)'
        return v
    
    @field_validator('save_all_outputs', mode='before')
    @classmethod
    def _check_save_all_outputs(cls, v):
        if v is None:
            return False
        else:
            return v
    
    @field_validator('target_dynamic_range', mode='before')
    @classmethod
    def _check_target_dynamic_range(cls, v):
        if v is None:
            return 0.0
        else:
            return v
    
def parse_args_inference():
    """
    Parses the arguments for inference from a YAML file and updates the argument object with additional parameters.

    This function reads a YAML file specified by the `--yaml_file` argument, updates the argument object with the parameters
    specified in the YAML file, and sets additional parameters for inference. It also performs some checks and prints
    information messages.

    :return: The updated argument object.
    :rtype: argparse.Namespace

    :raises AssertionError: If the `data_file` argument does not end with '.mat' or '.fits'.
                             If the `series` argument is 'R3D3' and `layers` is not greater than 1.
    """
    args_yaml = parse_yaml_file()
    with open(args_yaml.yaml_file, 'r') as file:
        yaml_loaded = yaml.safe_load(file)
        for k, v in yaml_loaded.items():
            if k not in args_yaml.__dict__ or args_yaml.__dict__[k] is None:
                args_yaml.__dict__.update({k: v})
        # args_yaml.__dict__.update(yaml.safe_load(file))
    args = InferenceArgs(**args_yaml.__dict__)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args

# Pytorch lightning (training/ testing) related argparse functions

def parse_args_pl():
    """Parse all required and optional arguments from command line.

    Returns
    -------
    _ArgumentParser
        Arguments parsed from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    
    # Model specific hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--num_pools', type=int, default=4, 
                        help='Number of U-Net pooling layers')
    parser.add_argument('--drop_prob', type=float, default=0.0, 
                        help='Dropout probability')
    parser.add_argument('--num_chans', type=int, default=64, 
                        help='Number of U-Net channels')
    parser.add_argument('--num_chans_in', type=int, default=2,
                        help='Number of input channels')
    parser.add_argument('--num_chans_out', type=int, default=1,
                        help='Number of output channels')
    parser.add_argument('--batch_size', default=1, type=int, 
                        help='Mini batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=1000, 
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--positivity', action='store_true',
                        help='If True, enforce positivity constraint on the output')
    parser.add_argument('--series', choices=['R2D2', 'R3D3'], required=True,
                        help='series to train')
    parser.add_argument('--layers', type=int, default=1, 
                        help='If > 1, R2D2-Net would be used instead of U-Net in R2D2')
    parser.add_argument('--num_iter', type=int, required=True, 
                        help='Iteration number in the R2D2/ R3D3 series to train')
    
    # Resource related arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus in each node')
    parser.add_argument('--nodes', type=int, default=1, 
                        help='Number of gpu nodes to be used')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='experiments',
                        help='Path where the checkpoint and hyparameters will be saved')
    parser.add_argument('--exp', type=str, 
                        help='Name of the experiment')
    parser.add_argument('--resume', action='store_true',
                        help='If True, resume the training from a previous model checkpoint. ')
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        help='Path to pre-trained model. Use with --mode test')
    
    # Dataset related arguments
    parser.add_argument('--data_path', type=pathlib.Path, required=True,
                        help='Path to the datasets')
    parser.add_argument('--im_dim_x', type=int, default=512,
                        help='Image dimension in x direction.')
    parser.add_argument('--im_dim_y', type=int, default=512,
                        help='Image dimension in y direction.')
    parser.add_argument('--scname_train', type=str, default='trainingset_fullnumpy', 
                        help='GT training set folder name')
    parser.add_argument('--scname_val', type=str, default='validationset_fullnumpy', 
                        help='GT validation set folder name')
    parser.add_argument('--scname_test', type=str, default='testset_fullnumpy', 
                        help='GT test set folder name')
    parser.add_argument('--rec_ext', type=str, default='_recN1', 
                        help='reconstruction input folder extension')
    parser.add_argument('--rec_file_ext', type=str, default='_rec', 
                        help='reconstruction image filename ending')
    parser.add_argument('--res_ext', type=str, default='_resN1', 
                        help='residual input folder extension')
    parser.add_argument('--dirty_file_ext', type=str, default='_dirty', 
                        help='dirty image filename ending')
    parser.add_argument('--res_file_ext', type=str, required=True,
                        help='residual image filename ending, e.g. _dirty for N1, _res for Ni')
    parser.add_argument('--gdth_file_ext', type=str, default='_gdth', 
                        help='ground truth image filename ending')
    parser.add_argument('--mat_ext', type=str, default=None,
                        help='ending of the folder with .mat files containing the dynamic range of the exponentiated image (1/a_expo).')
    parser.add_argument('--mat_file_ext', type=str, default=None,
                        help='filename ending of the files in mat_ext.')
    
    # R2D2Net/ R3D3 specific arguments
    parser.add_argument('--dirty_ext', type=str, default='_dirty', 
                        help='dirty image folder extension')
    parser.add_argument('--PSF_path', type=str, 
                        help='path to PSF files for unrolled R2D2Net')
    parser.add_argument('--PSF_file_ext', type=str, default='_PSF', 
                        help='PSF filename ending')
    
    # Inference related arguments
    parser.add_argument('--rec2_ext', type=str, default='_recN1', 
                        help='reconstruction output folder extension')
    parser.add_argument('--save_output', action='store_true', 
                        help='Save output')
    return parser.parse_args()

###################################################


class DataCommonArgs(BaseModel):
    """
    Pydantic model class containing common arguments for different tasks.
    """
    model_config = ConfigDict()
    
    data_type: DataTypeEnum
    seed: Union[int, str]
    
    # i/o
    gdth_path: DirectoryPath
    output_path: str
    uv_path: DirectoryPath
    
    dataset: str = 'test'
    save_vis: bool = False
    save_PSF: bool = False
    save_dirty: bool = False # option to save dirty images when generating visibilties
    return_sub_op: bool = False
    
    # measurement operator
    SR_from_filename: bool = False
    operator_type: OpTypeEnum = OpTypeEnum.table
    imweight_name: str = 'nWimag'
    on_gpu: bool = False
    briggs: bool = False
    
    # noise
    sigma_range_min: float = 0.
    sigma_range_max: float = 0.
    multi_noise : bool = False
    
    # exponentiation
    sigma0: float = 0.
    expo: bool = False
    
    # miscellaneous
    verbose: int = 1
    uv_random: bool = False
    
    # validation
    @field_validator('seed')
    @classmethod
    def _check_seed(cls, v):
        if type(v) == int:
            if v == 0:
                print('WARNING: Seed value 0 is not recommended, this will be set to 1377.')
                v = 1377
            assert v > 0, 'Seed value must be positive.'
        elif type(v) == 'str':
            assert v == 'uv', 'Only string value `uv` is accepted for seed, which will use the uv_id in the filename of the uv file as seed.'
        return v
    
    @field_validator('save_dirty')
    @classmethod
    def _check_layers(cls, v, values):
        if values.data['data_type'] != DataTypeEnum.visibilities:
            print('WARNING: `save_dirty` is only applicable for visibilities, this will be set to False.')
            v = False
        return v
    
    @field_validator('sigma_range_min', 'sigma_range_max')
    @classmethod
    def _check_sigma_range(cls, v):
        if v <= 0:
            raise ValueError('Sigma range must be non-negative and non-zero.')
        return v
    
    @field_validator('sigma_range_max')
    @classmethod
    def _check_sigma_range_max(cls, v, values):
        if v < values.data['sigma_range_min']:
            raise ValueError('Sigma range min must be less than sigma range max.')
        return v
    
    @field_validator('expo')
    @classmethod
    def _check_expo(cls, v, values):
        if v:
            assert values.data['sigma0'] > 0, 'Sigma0 must be positive for exponentiation.'
            assert values.data['sigma0'] > values.data['sigma_range_max'], 'Sigma0 must be greater than sigma range max for exponentiation.'
        return v
    
    @field_validator('data_type', 'operator_type')
    @classmethod
    def _return_value(cls, v):
        return v.value

###################################################

def parse_args_data_gen():
    args_yaml = parse_yaml_file()
    with open(args_yaml.yaml_file, 'r') as file:
        args_yaml.__dict__.update(yaml.safe_load(file))
    args = DataCommonArgs(**args_yaml.__dict__)
    return args
