#!/usr/bin/env bash

#DATA_FILE="/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle.small"
DATA_FILE="/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle"
python main.py --train --cuda \
    --infilling \
    --bar-pe \
    --model xlnet \
    --dim-model 1024 \
    --dim-subembed 512 \
    --dim-inner 4096 \
    --num-layer 8 \
    --num-head 8 \
    --seg-size 2048 \
    --mem-len 2048 \
    --batch-size 2 \
    --epoch-num 100 \
    --data-file $DATA_FILE \
    --accm-step 1 \
    --save-path "/screamlab/home/tanch/structure-aware_infilling/trained_model_xlnet_bar_pe_infilling/loss_%d.ckpt"
