#!/bin/bash
gdth_path=              # string, path to the ground truth files
uv_path=                # string, path to the uv files
output_path=            # string, main path to save the output, subdirectories will be created
dataset=                # string, choose from training, validation and test
super_resolution=       # scalar, super resolution factor
imweight_name=nWimag    # string, name of weighting variable in uv files, default is nWimag, if multiple, separate by comma
# Operator
operator_type=table     # choose between table and sprase_matrix, former is faster, latter is slightly more accurate

# For exponentiation, optional, comment out arguments if not used
sigma0=                 # scalar, 1/ current dynamic range of the ground truth images
expo=--expo             # flag, use exponentiation
# NOTE: please replace the sigma_range argument

# For Briggs weighting
briggs=--briggs         # flag, use Briggs weighting

python3 src/data_generation.py dirty \
--gdth_path $gdth_path \
--output_path $output_path \
--uv_path $uv_path \
--dataset $dataset \
--super_resolution $super_resolution \
--sigma0 $sigma0 $expo \
--sigma_range 2e-6 1e-3 $briggs \
--operator_type $operator_type \
--imweight_name $imweight_name