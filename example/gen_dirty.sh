#!/bin/bash
DATA_PATH=path_to_data
UV_PATH=$DATA_PATH/uv/
EXT=testset_fullnumpy
GDTH_PATH=$DATA_PATH/$EXT/
OUT_DIRTY=$DATA_PATH/$EXT\_dirty/
OUT_GDTH=$DATA_PATH/$EXT/

python3 src/data_generation.py dirty --uv_path $UV_PATH \
--gdth_path $GDTH_PATH \
--briggs \
--sigma0 0.001 --expo \
--sigma_range 2e-6 1e-3 \
--output_dirty_path $OUT_DIRTY \
--output_gdth_path $OUT_GDTH