import torch
import os
import shutil
from datetime import datetime
import numpy as np
import sys
import math
import model as mm

this = sys.modules[__name__]

def nucleus(probs: torch.FloatTensor, p=0.9, k=8):
    dims = probs.size()
    pre_size = 1
    for d in dims[:-1]:
        pre_size *= d

    t = probs.view(pre_size, dims[-1])
    t, idx = torch.sort(t, dim=-1, descending=True)
    not_sel = torch.cumsum(t, dim=-1) >= p
    not_sel[..., :k] = False # make sure there are at least k candidates
    t[not_sel] = 0.0
    choice = torch.multinomial(t, 1) # The rows of input do not need to sum to one
    final_choice = idx[torch.arange(pre_size), choice.view(-1)]

    return final_choice.view(dims[:-1])

def check_save_path(save_path):
    save_dir = os.path.dirname(save_path)
    parent, dir_name = os.path.dirname(save_dir), os.path.basename(save_dir)
    if not os.path.exists(save_dir):
        print(f"Create {save_dir} because directory not exist.")
        os.mkdir(save_dir)
    if len(os.listdir(save_dir)) != 0:
        print(f"Current saving directory: {save_dir}")
        print("Saving directory is not empty, would you like to keep this directory? [y/n/q]: ", end="")
        while True:
            c = input().lower()
            if c == "y":
                dscp = input("Add discription to old saving directory: ")
                dist_dir = os.path.join(parent, "%s_%s.%s" % (dir_name, dscp, datetime.now().strftime("%Y%m%d%H%M%S")))
                _ = input(f"Rename {save_dir} to {dist_dir}.\nEnter to continue:")
                # rename
                os.rename(save_dir,
                          os.path.join(parent, "%s_%s.%s" % (dir_name, dscp, datetime.now().strftime("%Y%m%d%H%M%S")))
                )
                # create new
                os.mkdir(save_dir)
                break
            elif c == "n":
                confirm = input(f"Delete all files in {save_dir}? [y/n]: ").lower()
                if confirm == "y":
                    # wipe out
                    shutil.rmtree(save_dir)
                    # create new
                    os.mkdir(save_dir)
                else:
                    exit()
                break
            elif c == "q":
                exit()
            else:
                print("Please enter [y/n], or enter [q] to quit: ", end="")

min_training_loss = 1.0
min_validation_loss = 1.0

def save_ckpt(save_path, epoch_idx, config, model, optimizer, training_loss, validation_loss, tokenizer):
    ckpt = mm.general.Checkpoint(
        epoch = epoch_idx,
        config = config,
        model_state_dict = model.state_dict(),
        optim_state_dict = optimizer.state_dict(),
        training_loss = training_loss,
        validation_loss = validation_loss,
        tokenizer = tokenizer,
    )

    if training_loss < this.min_training_loss:
        this.min_training_loss = training_loss

        file_dir, file = os.path.dirname(save_path), os.path.basename(save_path)
        file = "training_" + file
        file = file.replace("%d", str(math.floor(training_loss*10)))
        torch.save(ckpt, os.path.join(file_dir, file))

    if validation_loss is not None and validation_loss < this.min_validation_loss:
        this.min_validation_loss = validation_loss

        file_dir, file = os.path.dirname(save_path), os.path.basename(save_path)
        file = "validation_" + file
        file = file.replace("%d", str(math.floor(validation_loss*10)))
        torch.save(ckpt, os.path.join(file_dir, file))

def get_max_seq_len(songs, verbose=True):
    song_lens = np.array(list(map(lambda s: len(s), songs)))
    mean = song_lens.mean()
    sd = song_lens.var() ** 0.5
    max_seq_len = int(mean + 2*sd)
    if verbose:
        print("mean:", song_lens.mean())
        print("standard deviation:", song_lens.var()**0.5)
        print("max sequence length:", song_lens.max())
        match = song_lens[song_lens < max_seq_len]
        print("set max sequence length to:", f"{max_seq_len},",
              f"including {len(match)} songs of total ({round(len(match)/len(song_lens)*100, 2)}%)")
    return max_seq_len

log_on: bool = False
log_stdout: bool = True
log_file = None

def enable_log():
    this.log_on = True

def set_log_file(file):
    this.log_file = open(file, 'w')

def log(*args, **kargs):
    if this.log_on:
        if this.log_stdout:
            print(*args, **kargs)
        if this.log_file is not None:
            print(*args, **kargs, file=this.log_file)

def log_status():
    print("enable:", this.log_on)
    print("log stdout", this.log_stdout)
    print("log file:", this.log_file)
