for i in {0..3}; do
    CKPT="/screamlab/home/tanch/structural_expansion/trained_model/loss_$i.ckpt"
    python main.py --generate --cuda --ckpt-path $CKPT
done
