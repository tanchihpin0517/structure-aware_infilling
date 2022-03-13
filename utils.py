import torch
import os
import shutil
from datetime import datetime

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
