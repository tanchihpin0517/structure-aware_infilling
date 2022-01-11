for i in {0..9}; do
    CKPT="./trained_model/loss_$i.ckpt"
    python main.py --generate --cuda --ckpt-path $CKPT
done
