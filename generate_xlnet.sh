#!/usr/bin/env zsh

CKPT="/screamlab/home/tanch/structure-aware_infilling/trained_model_xlnet_no_cp_0/loss_2.ckpt"
DATA_FILE="/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle.small"
python main.py --generate --cuda --no-cp --infilling \
    --model xlnet \
    --seg-size 2048 \
    --gen-num 16 \
    --max-gen-len 1024 \
    --ckpt-path $CKPT \
    --data-file $DATA_FILE \
    --save-path "/screamlab/home/tanch/structure-aware_infilling/gen_midi_xlnet_no_cp_infilling_0"

#for i in {0..3}; do
#    CKPT="./trained_model/loss_$i.ckpt"
#    python main.py --generate --cuda --ckpt-path $CKPT
#done
