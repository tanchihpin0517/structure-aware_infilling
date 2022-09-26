#!/usr/bin/env zsh

CKPT="/screamlab/home/tanch/structure-aware_infilling/trained_model_transxl_single_half_struct/training_loss_3.ckpt"
#CKPT="/screamlab/home/tanch/structure-aware_infilling/trained_model_transxl_no_cp_struct_infilling_val_2/training_loss_2.ckpt"
DATA_FILE="/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle.small"
python main.py --generate --cuda \
    --infilling \
    --model transformer_xl \
    --seg-size 2048 \
    --gen-num 16 \
    --max-gen-len 4096 \
    --half-struct \
    --ckpt-path $CKPT \
    --data-file $DATA_FILE \
    --save-path "/screamlab/home/tanch/structure-aware_infilling/gen_midi_transxl_single_half_struct"

#for i in {0..3}; do
#    CKPT="./trained_model/loss_$i.ckpt"
#    python main.py --generate --cuda --ckpt-path $CKPT
#done
