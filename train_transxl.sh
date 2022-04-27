#!/usr/bin/env bash


#DATA_FILE="/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle.small"
DATA_FILE="/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle"
python main.py --train --cuda \
    --infilling \
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
    --data-file $DATA_FILE \
    --accm-step 1 \
    --save-path "/screamlab/home/tanch/structure-aware_infilling/trained_model_transxl_struct_infilling_enc/loss_%d.ckpt" \
    #--ckpt-path "/screamlab/home/tanch/structure-aware_infilling/trained_model_transxl_struct_infilling_1/validation_loss_0.ckpt"
