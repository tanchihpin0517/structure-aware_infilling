import os
import argparse
import preprocess
import pickle
import utils
from music import Song
from tqdm import tqdm
from model.transformer_xl import TransformerXLConfig, TransformerXL
from model.xlnet import XLNetConfig, XLNet
from model.tokenizer import Tokenizer
import torch
import numpy as np
import math
import pretty_midi
from model.general import Checkpoint
import random
import multiprocessing as mp
from typing import Tuple
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="xlnet")
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--generate', default=False, action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--vocab-file', type=str, default='dataset/vocab_debug.txt')
    parser.add_argument('--data-file', type=str, default='dataset/pop909.pickle')
    parser.add_argument('--preprocess', default=False, action='store_true')
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--no-cp', default=False, action='store_true')
    parser.add_argument('--gen-num', type=int, default=16)
    parser.add_argument('--infilling', default=False, action='store_true')

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seg-size', type=int, default=1024)
    parser.add_argument('--epoch-num', type=int, default=1, help='number of training epochs')
    parser.add_argument('--accm-step', type=int, default=1)
    parser.add_argument('--max-seq-len', type=int, default=None)
    parser.add_argument('--only-middle', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    # model configuration
    parser.add_argument('--dim-model', type=int, default=512)
    parser.add_argument('--dim-inner', type=int, default=2048)
    parser.add_argument('--dim-subembed', type=int, default=128)
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-layer', type=int, default=8)
    parser.add_argument('--mem-len', type=int, default=1024) # default is same as seg_size

    parser.add_argument('--training-split-ratio', type=int, default=9)
    parser.add_argument('--max-gen-len', type=int, default=4096, help='number of tokens in generation')
    return parser.parse_args()

def main():
    args = parse_args()
    use_cp = (not args.no_cp)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.preprocess:
        gen_data(args.data_file, small_size=16)
        exit()

    #data = load_data(args.data_file, args.preprocess, melody_only=True, max_song_num=16)
    songs_data = load_data(args.data_file, track_sel=['melody', 'bridge'])
    #songs_data = load_data(args.data_file, track_sel=['melody', 'bridge', 'piano'])

    if args.train:
        utils.check_save_path(args.save_path)

        tokenizer = Tokenizer(args.vocab_file, use_cp=use_cp)
        song_ids, bar_ids = tokenize(songs_data, tokenizer)

        assert len(song_ids) == len(bar_ids)
        sp = len(song_ids)*(10-args.training_split_ratio)//10
        training_song_ids = song_ids[sp:]
        training_bar_ids = bar_ids[sp:]

        if args.model == "transformer_xl":
            config = TransformerXLConfig(
                tokenizer.vocab_size(),
                d_model = args.dim_model,
                d_inner = args.dim_inner,
                n_head = args.num_head,
                n_layer = args.num_layer,
                mem_len = args.mem_len,
                use_cp=use_cp,
                d_subembed=args.dim_subembed,
                class_ranges=tokenizer.class_ranges(),
                infilling=args.infilling,
            )
            model = TransformerXL(config)
            train_transxl(
                model,
                training_song_ids,
                args.epoch_num,
                args.batch_size,
                args.seg_size,
                args.cuda,
                config,
                tokenizer,
                args.save_path,
                accm_step=args.accm_step,
                max_seq_len=args.max_seq_len,
                only_middle=args.only_middle,
            )
        elif args.model == "xlnet":
            config = XLNetConfig(
                tokenizer.vocab_size(),
                d_model = args.dim_model,
                d_inner = args.dim_inner,
                n_head = args.num_head,
                n_layer = args.num_layer,
                mem_len = args.mem_len,
                use_cp=use_cp,
                d_subembed=args.dim_subembed,
                class_ranges=tokenizer.class_ranges(),
                attn_type="bi",
                infilling=args.infilling,
            )
            model = XLNet(config)
            train_xlnet(
                model,
                training_song_ids,
                training_bar_ids,
                args.epoch_num,
                args.batch_size,
                args.seg_size,
                args.cuda,
                config,
                tokenizer,
                args.save_path,
                accm_step=args.accm_step,
                max_seq_len=args.max_seq_len,
                only_middle=args.only_middle,
            )
        else:
            raise Exception(f"Unknow model type: {args.model}")

    if args.test:
        ckpt = torch.load(args.ckpt_path)
        model = TransformerXL(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        tokenizer = ckpt.tokenizer
        print("ckpt:", "args.ckpt_path", ", epoch:", ckpt.epoch, ", loss:", ckpt.loss)
        with torch.no_grad():
            test(model, songs[0],
                 args.cuda, args.seg_size, tokenizer)

    if args.generate:
        songs = songs_data[:args.gen_num]
        ckpt = torch.load(args.ckpt_path)
        tokenizer = ckpt.tokenizer

        if args.model == "transformer_xl":
            model = TransformerXL(ckpt.config)
            model.load_state_dict(ckpt.model_state_dict)
            print("ckpt:", args.ckpt_path, ", epoch:", ckpt.epoch, ", loss:", ckpt.loss)
            with torch.no_grad():
                prompts = select_bars(songs, 0, 8)
                prompt_ids = tokenize(prompts, tokenizer, with_eos=False)
                result_ids = generate_transxl(
                    model,
                    prompt_ids,
                    args.cuda,
                    args.seg_size,
                    tokenizer,
                    max_gen_len=args.max_gen_len
                )

                gen_song_ids = []
                for i in range(len(prompt_ids)):
                    gen_song_ids.append(prompt_ids[i] + result_ids[i])

            for i in range(len(gen_song_ids)):
                gen_song = tokenizer.decode(tokenizer.id_to_token(gen_song_ids[i]), Song.copy(songs[i], with_content=False))

                save_dir = os.path.join(args.save_path, f"{math.floor(ckpt.loss*10)/10.0}")
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                gen_song.save(os.path.join(save_dir, f"{gen_song.name}.midi"))
                gen_song.save_fig(os.path.join(save_dir, f"{gen_song.name}.png"))
                #with open(f"./gen_midi/{song.name}.txt", "w") as f:
                #    f.write("\n".join(text))
        elif args.model == 'xlnet':
            model = XLNet(ckpt.config)
            model.load_state_dict(ckpt.model_state_dict)
            print("ckpt:", "args.ckpt_path", ", epoch:", ckpt.epoch, ", loss:", ckpt.loss)
            with torch.no_grad():
                prompts_past = tokenize(select_bars(songs, 0, 8), tokenizer, with_eos=False)
                #prompts_future = select_bars(songs, 24, 32)
                prompts_future = [ [] for _ in range(len(prompts_past))]
                result_ids = generate_xlnet(
                    model,
                    prompts_past,
                    prompts_future,
                    args.max_gen_len,
                    args.cuda,
                    args.seg_size,
                    tokenizer
                )

                gen_song_ids = []
                for i in range(len(prompts_past)):
                    gen_song_ids.append(prompts_past[i] + result_ids[i] + prompts_future[i])

            for i in range(len(gen_song_ids)):
                gen_song = tokenizer.decode(tokenizer.id_to_token(gen_song_ids[i]), Song.copy(songs[i], with_content=False))

                save_dir = os.path.join(args.save_path, f"{math.floor(ckpt.loss*10)/10.0}")
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                gen_song.save(os.path.join(save_dir, f"{gen_song.name}.midi"))
                gen_song.save_fig(os.path.join(save_dir, f"{gen_song.name}.png"))
        else:
            raise Exception(f"Unknow model type: {args.model}")

def gen_data(data_file, small_size=16):
    #data, err_cnt = preprocess.pop909("./dataset/pop909/POP909", "./dataset/pop909_struct/POP909", song_sel=5, multi_task=False)
    data, err_cnt = preprocess.pop909("./dataset/pop909/POP909", "./dataset/pop909_struct/POP909")
    print("number of error in preprocessing:", err_cnt)

    with open(data_file, 'wb') as f:
        if os.path.exists(data_file):
            print(f"update existing file: {data_file}")
        else:
            print(f"create file: {data_file}")
        pickle.dump(data, f)

    small_file = f"{data_file}.small"
    with open(small_file, 'wb') as f:
        if os.path.exists(small_file):
            print(f"update existing file: {small_file}")
        else:
            print(f"create file: {small_file}")
        pickle.dump(data[:small_size], f)

def load_data(data_file, track_sel=['melody', 'bridge', 'piano'], max_song_num=None):
    print(f"Load data: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    if max_song_num is not None:
        data = data[:max_song_num]

    for song in data:
        for event in song.flatten_events():
            tmp = []
            for note in event.notes:
                if note.track in track_sel:
                    tmp.append(note)
            event.notes = tmp

    return data

def tokenize(songs_data, tokenizer, with_eos=True) -> Tuple[list, list]:
    songs = []
    bar_ids = []

    with mp.Pool() as pool:
        map_args = []
        for song in songs_data:
            map_args.append((song, tokenizer, with_eos))

        pbar = tqdm(desc="Tokenize", total=len(songs_data))
        for i, (song, bar_id) in enumerate(pool.imap(tokenize_map, map_args)):
            assert len(song) == len(bar_id), f"song {songs_data[i].name}: len(song, bar_id) = ({len(song)}, {len(bar_id)})"
            songs.append(song)
            bar_ids.append(bar_id)
            pbar.update(1)
        pbar.close()

    return songs, bar_ids

def tokenize_map(args):
    song, tokenizer, with_eos = args
    song, bar_id = tokenizer.encode(song, with_eos=with_eos)
    song = tokenizer.token_to_id(song)
    return song, bar_id

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

def to_tensor_with_zero_padding(songs, max_seq_len):
    tokens = []
    #masks = []
    for song in songs:
        tokens.append(song[:max_seq_len] + [0 for i in range(max_seq_len-len(song))])
        #masks.append([1]*len(song[:max_seq_len]) + [0 for i in range(max_seq_len-len(song))])
    tokens = torch.LongTensor(tokens)
    masks = (tokens != 0).to(torch.float)
    #masks = torch.FloatTensor(masks)

    return tokens, masks

def save_ckpt(save_path, epoch_idx, config, model, optimizer, loss, tokenizer):
    ckpt = Checkpoint(
        epoch = epoch_idx,
        config = config,
        model_state_dict = model.state_dict(),
        optim_state_dict = optimizer.state_dict(),
        loss = loss,
        tokenizer = tokenizer,
    )
    if ckpt.loss < 1:
        torch.save(ckpt, save_path.replace("%d", str(math.floor(ckpt.loss*10))))

def default_optimizer(model, lr=1e-3):
    return torch.optim.Adam(model.parameters(), lr=lr)

def default_scheduler(optimizer, lr_max=1.0, lr_min=1.0, T=10, warmup=0, refine=50):
    #lr_lambda = lambda epoch: 2 if epoch < 10 else 1
    lr_lambda = lambda epoch: (epoch+1)/warmup if epoch < warmup else \
                              lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos(epoch/T*math.pi)) if epoch < refine else lr_min
    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

def train_transxl(model, song_ids, epoch_num, batch_size, seg_size, cuda, config, tokenizer, save_path, accm_step=1, max_seq_len=None, only_middle=False):
    model.train()
    model = model.cuda() if cuda else model

    # permute songs before count max sequence length because permutation will add 2 segment tokens into songs
    song_ids, seg_ids, ignore_labels = model.permute(song_ids, 1/3, 2/3, tokenizer, only_middle=only_middle)

    """
    To reduce training time, we set the max sequence length to (mean + 2*standard_deviation)
    """
    if max_seq_len is None:
        max_seq_len = get_max_seq_len(song_ids)
    else:
        print(f"max sequence length is set to {max_seq_len}")

    song_ids, _ = tokenizer.pad(song_ids, 0, max_seq_len, gen_mask=False)
    seg_ids, _ = tokenizer.pad(seg_ids, 0, max_seq_len, gen_mask=False, use_cp=False)
    ignore_labels, _ = tokenizer.pad(ignore_labels, 0, max_seq_len, gen_mask=False, use_cp=False)

    """
    if use_cp:
        labels: (C, B, L)
    else:
        labels: (1, B, L)
    """
    labels = tokenizer.get_labels(song_ids, ignore_labels=ignore_labels)

    optimizer = default_optimizer(model, lr=1e-4)
    scheduler = default_scheduler(optimizer, lr_max=2.0, lr_min=1.0, T=10, warmup=10, refine=50)

    for epoch_idx in range(epoch_num):
        pbar = tqdm(desc=f"epoch {epoch_idx+1}", total=len(song_ids))
        running_loss = 0.0
        n_tokens = 0

        for batch_step, batch_idx in enumerate(range(0, len(song_ids), batch_size)):
            batch = song_ids[batch_idx:batch_idx+batch_size]
            mems = None
            for seg_idx in range(0, max_seq_len, seg_size): # split a long sequence into small segments
                songs_batch = song_ids[batch_idx:batch_idx+batch_size, seg_idx:seg_idx+seg_size].to(model.device)
                labels_batch = labels[:, batch_idx:batch_idx+batch_size, seg_idx+1:seg_idx+seg_size+1].to(model.device)
                type_batch = seg_ids[batch_idx:batch_idx+batch_size, seg_idx:seg_idx+seg_size].to(model.device)

                output = model(
                    input_ids=songs_batch,
                    mems=mems,
                    labels=labels_batch,
                    token_type_ids=type_batch,
                )
                loss = torch.mean(torch.stack(output.losses, dim=0))
                loss.backward()

                mems = output.mems

                n = len(labels[labels != tokenizer.ignore_idx])
                running_loss += loss.item() * n
                n_tokens += n

            if (batch_step+1) % accm_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.update(len(batch))
        pbar.close()

        running_loss = running_loss / n_tokens
        print(" "*4, "average loss:", running_loss, " "*4, "learning rate:", scheduler.get_last_lr()[0], "\n")
        scheduler.step()

        save_ckpt(save_path, epoch_idx, config, model, optimizer, running_loss, tokenizer)

def train_xlnet(model, song_ids, bar_ids, epoch_num, batch_size, seg_size, cuda, config, tokenizer, save_path, accm_step=1, max_seq_len=None, only_middle=False):
    model.train()
    model = model.cuda() if cuda else model

    """
    To reduce training time, we set the max sequence length to (mean + 2*standard_deviation)
    """
    if max_seq_len is None:
        max_seq_len = get_max_seq_len(song_ids)
    else:
        print(f"max sequence length is set to {max_seq_len}")

    # attention mask: 0 -> not attend, 1 -> attend
    song_ids, attention_masks = tokenizer.pad(song_ids, 0, max_seq_len)

    # extend bar_ids to max_seq_len
    bar_ids = copy.deepcopy(bar_ids) # why not?
    for i, bid in enumerate(bar_ids):
        bar_ids[i] = bid[:max_seq_len] + [bid[-1]]*(max_seq_len-len(bid)) # extend the last bid to padding part
    bar_ids = torch.LongTensor(bar_ids)
    assert bar_ids.shape == song_ids.shape

    # permutation mask: 0 -> attend, 1 -> not attend
    permutation_masks, tgt_mappings, tgt_labels = model.gen_mask_and_target(song_ids, max_seq_len, max_seq_len//3, max_seq_len//3*2, only_middle=only_middle)
    tgt_labels = tokenizer.get_labels(tgt_labels)

    optimizer = default_optimizer(model, lr=1e-4)
    scheduler = default_scheduler(optimizer, lr_max=2.0, lr_min=1.0, T=10, warmup=10, refine=50)

    for epoch_idx in range(epoch_num):
        pbar = tqdm(desc=f"epoch {epoch_idx+1}", total=len(song_ids))
        running_loss = 0.0
        n_tokens = 0

        for batch_step, batch_idx in enumerate(range(0, len(song_ids), batch_size)):
            bs, be = batch_idx, batch_idx+batch_size
            mems = None
            mem_pos_ids = None

            # split a long sequence into small segments
            for _, seg_idx in enumerate(range(0, max_seq_len, seg_size)):
                ss, se = seg_idx, seg_idx+seg_size

                segs = song_ids[bs:be, ss:se].to(model.device)
                bids = bar_ids[bs:be, ss:se].to(model.device)
                labels = tgt_labels[:, bs:be, ss:se].to(model.device)
                attn_mask = attention_masks[bs:be, ss:se].to(model.device)
                perm_mask = permutation_masks[bs:be, ss:se, ss:se].to(model.device)
                mapping = tgt_mappings[bs:be, ss:se, ss:se].to(model.device)

                output = model(
                    input_ids=segs,
                    pos_ids=bids,
                    mem_pos_ids=mem_pos_ids,
                    labels=labels,
                    attention_mask=attn_mask,
                    mems=mems,
                    perm_mask=perm_mask,
                    target_mapping=mapping,
                )
                loss = torch.mean(torch.stack(output.losses, dim=0))
                loss.backward()

                mems = output.mems
                mem_pos_ids = output.mem_pos_ids

                n = len(labels[labels != tokenizer.ignore_idx])
                running_loss += loss.item() * n
                n_tokens += n

            if (batch_step+1) % accm_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.update(len(song_ids[bs:be]))
        pbar.close()

        running_loss = running_loss / n_tokens
        print(" "*4, "average loss:", running_loss, " "*4, "learning rate:", scheduler.get_last_lr()[0], "\n")
        scheduler.step()

        save_ckpt(save_path, epoch_idx, config, model, optimizer, running_loss, tokenizer)

def test(model, song, cuda, seg_size, tokenizer):
    model.eval()
    model = model.cuda() if cuda else model

    tokens = torch.tensor(song, dtype=torch.int)
    # limit segment length not longer than memory length
    seg_size = model.mem_len if model.mem_len < seg_size else seg_size

    print("teacher forcing")
    mems = None
    for seg_idx in range(0, len(song), seg_size): # split a long sequence into small segments
        seg = tokens[seg_idx:seg_idx+seg_size][None,:]
        if cuda:
            seg = seg.cuda()
        output = model(input_ids=seg, mems=mems)
        mems = output.mems
        output_ids = torch.argmax(output.pred_scores, dim=-1)[0]
        print(list(map(lambda i: tokenizer[i.item()], output_ids)))

    print("prompt without memory")
    for seg_idx in range(0, len(song), seg_size): # split a long sequence into small segments
        seg = song[seg_idx:seg_idx+seg_size]
        output_ids = [seg[0]]
        if cuda:
            seg = seg.cuda()
        for i in range(len(seg)):
            input_ids = torch.tensor(output_ids, dtype=torch.int)[None, :]
            output = model(input_ids=input_ids, mems=None)
            output_ids.append(torch.argmax(output.pred_scores, dim=-1)[0][-1].item())
        print(list(map(lambda i: tokenizer[i], output_ids)))

    print("prompt with memory")
    mems = None
    for seg_idx in range(0, len(song), seg_size): # split a long sequence into small segments
        seg = song[seg_idx:seg_idx+seg_size]
        output_ids = [seg[0]]
        if cuda:
            seg = seg.cuda()
        for i in range(len(seg)):
            input_ids = torch.tensor(output_ids, dtype=torch.int)[None, -1:]
            output = model(input_ids=input_ids, mems=mems)
            mems = output.mems
            output_ids.append(torch.argmax(output.pred_scores, dim=-1)[0][-1].item())
        print(list(map(lambda i: tokenizer[i], output_ids)))

def select_bars(songs, start, end):
    clipped_songs = []
    for song in songs:
        clipped_songs.append(song.clip(start, end))
    return clipped_songs

def generate_transxl(model, prompt_ids, cuda, seg_size, tokenizer, max_gen_len):
    model.eval()
    model = model.cuda() if cuda else model

    result_ids = []
    pbar = tqdm(desc="Generating", total=len(prompt_ids))
    for prompt in prompt_ids:
        result = []
        prompt = torch.LongTensor(prompt)
        # limit segment length not longer than memory length
        seg_size = model.mem_len if model.mem_len < seg_size else seg_size

        """
        generate momery
        """
        gen_id = None
        mems = None
        for seg_idx in range(0, len(prompt), seg_size): # split a long sequence into small segments
            segs = prompt[None, seg_idx:seg_idx+seg_size]
            segs = segs.cuda() if cuda else segs

            output = model(input_ids=segs, mems=None)
            mems = output.mems

            #output_ids = torch.argmax(output.pred_scores, dim=-1)
            while True:
                output_ids = tokenizer.sample(output.pred_scores)
                gen_id = output_ids[0, -1]
                if tokenizer.is_legal(gen_id):
                    break
                else:
                    print("illegal generated id:", tokenizer.id_to_token(gen_id))
            result.append(gen_id.tolist())

        """
        generate new contents
        """
        #output_ids = torch.argmax(output.pred_scores, dim=-1)
        while True:
            segs = gen_id[None, None]
            segs = segs.cuda() if cuda else segs

            output = model(input_ids=segs, mems=mems)
            mems = output.mems

            #gen_id = torch.argmax(output.pred_scores, dim=-1)[0, -1].item()
            while True:
                output_ids = tokenizer.sample(output.pred_scores)
                gen_id = output_ids[0, -1]
                if tokenizer.is_legal(gen_id):
                    break
                else:
                    print("illegal generated id:", tokenizer.id_to_token(gen_id))
            result.append(gen_id.tolist())
            if tokenizer.is_eos(gen_id):
                break
            if len(result) >= max_gen_len:
                break
        result_ids.append(result)
        pbar.update(1)
    pbar.close()

    return result_ids

def generate_xlnet(model, prompts_past, prompts_future, gen_len, cuda, seg_size, tokenizer):
    model.eval()
    model = model.cuda() if cuda else model

    result_ids = []
    #assert len(prompts_past) == len(prompts_future)
    pbar = tqdm(desc="Generating", total=len(prompts_past))
    for i in range(len(prompts_past)):
        past = prompts_past[i]
        future = prompts_future[i]

        prompt = past + ([0] * gen_len) + future

        total_len = len(prompt)
        gen_start = len(past)
        gen_end = gen_start + gen_len

        prompt = torch.LongTensor(prompt)[None, :] # make shape to (1, L)
        permutation_masks, tgt_mappings, tgt_labels = model.gen_mask_and_target(prompt, total_len, gen_start, gen_end)

        # limit segment length not longer than memory length
        seg_size = model.mem_len if model.mem_len < seg_size else seg_size

        """
        generate memory
        unlike transformerXL, we don't sample the first generated id(s) here
        """
        ss, se = (0, len(past)+1) # generate the first target to get memory
        segs = prompt[:, ss:se].to(model.device)
        tgt_idx = len(past)

        perm_mask = permutation_masks[:, ss:se, ss:se].to(model.device)
        mapping = tgt_mappings[:, tgt_idx:tgt_idx+1, ss:se].to(model.device)

        output = model(
            input_ids=segs,
            labels=None,
            attention_mask=None,
            mems=None,
            perm_mask=perm_mask,
            target_mapping=mapping,
        )

        mems = output.mems
        result = []

        """
        generate result
        """
        ss, se = (gen_start, prompt.shape[1])

        for tgt_idx in range(gen_start, gen_end):
            segs = prompt[:, ss:se].to(model.device)
            perm_mask = permutation_masks[:, ss:se, ss:se].to(model.device)
            mapping = tgt_mappings[:, tgt_idx:tgt_idx+1, ss:se].to(model.device)

            output = model(
                input_ids=segs,
                labels=None,
                attention_mask=None,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=mapping,
            )
            #gen_id = utils.nucleus(output.pred_scores)[0, -1].item()
            while True:
                output_ids = tokenizer.sample(output.pred_scores)
                gen_id = output_ids[0, 0]
                if tokenizer.is_legal(gen_id):
                    break
                else:
                    print("illegal generated id:", tokenizer.id_to_token(gen_id))
            result.append(gen_id.tolist())
            if tokenizer.is_eos(gen_id):
                break
            if len(result) >= gen_len:
                break
            prompt[0][tgt_idx] = gen_id

            #mems = output.mems
        result_ids.append(result)
        pbar.update(1)
    pbar.close()

    return result_ids

if __name__ == "__main__":
    main()
