#!/usr/bin/env zsh

CKPT="./trained_model_xlnet.20220223115607/loss_1.ckpt"
python main.py --generate --cuda \
    --model xlnet \
    --ckpt-path $CKPT


exit

for i in {0..3}; do
    CKPT="./trained_model/loss_$i.ckpt"
    python main.py --generate --cuda --ckpt-path $CKPT
done
