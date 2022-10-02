#!/usr/bin/env bash

#DATA_FILE="./dataset/pop909.pickle.small"
#DATA_FILE="./dataset/pop909.pickle"
TN_DATA_FILE="./dataset/pop909.pickle.training"
VAL_DATA_FILE="./dataset/pop909.pickle.testing"
python main.py --train --cuda \
    --infilling \
    --struct-ratio 0.8 \
    --model transformer_xl \
    --dim-model 512 \
    --dim-inner 2048 \
    --dim-subembed 512 \
    --num-layer 6 \
    --num-head 8 \
    --seg-size 2048 \
    --mem-len 2048 \
    --max-struct-len 512 \
    --batch-size 4 \
    --epoch-num 1000 \
    --training-data $TN_DATA_FILE \
    --testing-data $VAL_DATA_FILE \
    --accm-step 1 \
    --save-path "./trained_model_transxl_single_080_struct/loss_%d.ckpt" \
    #--ckpt-path "./trained_model_transxl_struct_infilling_1/validation_loss_0.ckpt"