#!bin/bash

repo_pth=
data_file=$repo_pth/data/3c353/data_3c353.mat
output_path=
ckpt_path=
series=R3D3
num_iter=8
layers=6
gdth_file=$repo_pth/data/3c353/3c353_GTfits.fits
# target_dynamic_range=

mkdir -p $output_path

python3 $repo_pth/src/run_series.py \
--data_file $data_file \
--ckpt_path $ckpt_path \
--output_path $output_path \
--series $series \
--num_iter $num_iter \
--layers $layers \
--super_resolution 1.5 \
--im_dim_x 512 --im_dim_y 512 \
--gdth_file $gdth_file \
# --target_dynamic_range $target_dynamic_range \
# --save_all_outputs