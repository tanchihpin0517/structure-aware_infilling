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

def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    this.min_training_loss = ckpt.min_training_loss
    this.min_validation_loss = ckpt.min_validation_loss
    return ckpt

def save_ckpt(save_path, epoch_idx, config, model, optimizer, scheduler, training_loss, validation_loss, tokenizer):
    ckpt = mm.general.Checkpoint(
        epoch = epoch_idx,
        config = config,
        model_state_dict = model.state_dict(),
        optim_state_dict = optimizer.state_dict(),
        sched_state_dict = scheduler.state_dict(),
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
              f"including {len(match)} items of total data ({round(len(match)/len(song_lens)*100, 2)}%)")
    return max_seq_len

def melody_simularity(q, c):
    """
    c: compared
    q: query

    In the paper, the query is transposed into 12 possible key because the pitches of a melody can be shifted.
    We exchange c and q to get the mean of 2 scores because in the original paper, this metric is used for find a melody from a long song.

    reference: https://www.cs.cmu.edu/~rbd/papers/icmc02melodicsimilarity.pdf
    """
    if len(q) == 0 or len(c) == 0:
        return 0

    c = np.array(c)
    q = np.array(q)
    d = []
    for s in range(12):
        r1 = _melody_simularity_impl(c+s, q)
        #r2 = _melody_simularity_impl(q, c+s)
        #d.append((r1+r2) / 2)
        d.append(r1)
        #print(c, q+s, d[-1])
    return min(d)

def _linear_distance(p1, p2):
    d = (p1-p2) % 12
    return min(d, 12-d)

def _melody_simularity_impl(a, b, verbose=False):
    """
    A = (a1, a2, ..., am) : compared
    B = (b1, b2, ..., bn) : query

    d(i, j) represents the dissimilarity between (a1, a2, ..., ai) and (b1, b2, ..., bj)
    """
    a = np.array(a)
    b = np.array(b)

    a = np.concatenate((np.array([(a%12).mean()]), a))
    b = np.concatenate((np.array([(b%12).mean()]), b))
    d = np.zeros([len(a), len(b)])
    w = _linear_distance

    # initialize
    for i in range(d.shape[0]):
        d[i, 0] = d[i-1, 0] + w(a[i], b[0])
    for j in range(1, d.shape[1]):
        d[0, j] = d[0, j-1] + w(a[0], b[j])

    for i in range(d.shape[0]):
        d[i, -1] = math.inf
    for j in range(d.shape[1]):
        d[-1, j] = math.inf

    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            d[i, j] = min(
                d[i-1, j-1],
                #d[i-2, j-1] + (w(a[i-1], b[j]) if i > 1 else 0),
                #d[i-1, j-2] + (w(a[i], b[j-1]) if j > 1 else 0),
                d[i-2, j-1] + w(a[i-1], b[j]),
                d[i-1, j-2] + w(a[i], b[j-1]),
            ) + w(a[i], b[j])

    if verbose:
        print(d)

    #candidates = np.concatenate((d[1:, -1], d[-1, 1:]))
    candidates = d[:, -1]
    result = candidates.min()

    return result