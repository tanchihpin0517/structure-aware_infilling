#!/usr/bin/env zsh

CKPT="/screamlab/home/tanch/structural_expansion/trained_model_transxl_cp.20220313214011/loss_3.ckpt"
DATA_FILE="/screamlab/home/tanch/structural_expansion/dataset/pop909.pickle.small"
python main.py --generate --cuda \
    --model transformer_xl \
    --seg-size 2048 \
    --gen-num 16 \
    --max-gen-len 2048 \
    --ckpt-path $CKPT \
    --data-file $DATA_FILE \
    --save-path "/screamlab/home/tanch/structural_expansion/gen_midi_test"

#for i in {0..3}; do
#    CKPT="./trained_model/loss_$i.ckpt"
#    python main.py --generate --cuda --ckpt-path $CKPT
#done
