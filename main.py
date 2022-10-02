import os
import argparse
import preprocess
import pickle
import utils
from music import Song, Bar, Note
from tqdm import tqdm
from model.transformer_xl import TransformerXLConfig, TransformerXL
from model.tokenizer import Tokenizer
import torch
import numpy as np
import math
import random
import multiprocessing as mp
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true', help="run training")
    parser.add_argument('--experiment', default=False, action='store_true', help="run experiment")
    parser.add_argument('--generate', default=False, action='store_true', help="generating with custom song file")
    parser.add_argument('--preprocess', default=False, action='store_true', help="run preprocessing music data")
    parser.add_argument('--cuda', default=False, action='store_true', help="enable gpu")
    parser.add_argument('--vocab-file', type=str, default='dataset/vocab_debug.txt', help="vocabulary file which is not used since the vocabulary is hard coded currently")
    parser.add_argument('--data-file', type=str, default='dataset/pop909.pickle', help="where to save the result of preprocessing")
    parser.add_argument('--save-path', type=str, default=None, help="where to 'save' model checkpoint")
    parser.add_argument('--ckpt-path', type=str, default=None, help="where to 'load' model checkpoint (resume training procedure if this value is assigned)")
    parser.add_argument('--cp', default=False, action='store_true', help="use couponded word representation (deprecated)")
    parser.add_argument('--no-bar-cd', default=False, action='store_true', help="disable bar-count-down technique")
    parser.add_argument('--no-order', default=False, action='store_true', help="disable order embedding")
    parser.add_argument('--gen-num', type=int, default=16)
    parser.add_argument('--infilling', default=False, action='store_true', help='(deprecated)')
    parser.add_argument('--training-data', type=str, default=None)
    parser.add_argument('--testing-data', type=str, default=None)
    parser.add_argument('--song-file', type=str, default=None, help="user's custom input file")

    # data
    parser.add_argument('--batch-size', type=int, default=1, help="batch size")
    parser.add_argument('--seg-size', type=int, default=2048, help="segment size")
    parser.add_argument('--epoch-num', type=int, default=1, help="how many epochs run on training")
    parser.add_argument('--accm-step', type=int, default=1, help="the number of steps for each parameter updating")
    parser.add_argument('--max-seq-len', type=int, default=1024, help="maximum sequence length")
    parser.add_argument('--bar-range-num', type=int, default=8, help="the number of bars of past and future contexts")
    parser.add_argument('--with-past', default=False, action='store_true', help="consider the loss on past context while training")
    parser.add_argument('--seed', type=int, default=0, help="random seed")

    # model configuration
    parser.add_argument('--dim-model', type=int, default=512, help="dimention of the input embedding")
    parser.add_argument('--dim-inner', type=int, default=2048, help="dimention of the feedforward layer")
    parser.add_argument('--dim-subembed', type=int, default=128, help="(deprecated)")
    parser.add_argument('--num-head', type=int, default=8, help="number of attention heads")
    parser.add_argument('--num-layer', type=int, default=8, help="number of attention layers")
    parser.add_argument('--mem-len', type=int, default=2048, help="memory length") # default is same as seg_size
    parser.add_argument('--max-struct-len', type=int, default=512, help="maxmum sequence length of the structure reference in the encoder layer")
    parser.add_argument('--struct-ratio', type=float, default=1.0, help="(deprecated)")
    parser.add_argument('--max-gen-len', type=int, default=4096, help='maximum number of tokens in generation')

    return parser.parse_args()

def main():
    args = parse_args()
    """
    If ckpt_path is assigned, we use this checkpoint to resume the training rather than creating a new one.
    """
    resume = (args.ckpt_path is not None)

    if args.preprocess:
        """
        total data = (testing_data, training_data)
        0 ~ 10% => testing_data
        10% ~ 100% => training_data
        """
        gen_data(args.data_file, small_size=16, split_ratio=0.1)
        exit()

    use_bar_cd = (not args.no_bar_cd)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.train:
        if not resume:
            utils.check_save_path(args.save_path)

        tokenizer = Tokenizer(args.vocab_file)

        ckpt = None
        if resume:
            ckpt = utils.load_ckpt(args.ckpt_path)
            tokenizer = Tokenizer(args.vocab_file)
            print("ckpt:", args.ckpt_path)
            print("epoch:", ckpt.epoch)
            print("training loss:", ckpt.training_loss)
            print("validation loss:", ckpt.validation_loss)
            print("use cp:", ckpt.config.use_cp)
            print("use bar count down:", ckpt.config.use_bar_cd)
            config = ckpt.config
            model = TransformerXL(ckpt.config)
            model.load_state_dict(ckpt.model_state_dict)
        else:
            config = TransformerXLConfig(
                tokenizer.vocab_size(),
                d_model = args.dim_model,
                d_inner = args.dim_inner,
                n_head = args.num_head,
                n_layer = args.num_layer,
                mem_len = args.mem_len,
                struct_len = args.max_struct_len,
                use_cp=args.cp,
                use_bar_cd=use_bar_cd,
                d_subembed=args.dim_subembed,
                class_ranges=tokenizer.class_ranges(),
                infilling=args.infilling,
            )
            model = TransformerXL(config)
        train(
            model,
            args.training_data,
            args.testing_data,
            args.epoch_num,
            args.batch_size,
            args.seg_size,
            args.cuda,
            config,
            tokenizer,
            args.save_path,
            accm_step=args.accm_step,
            max_seq_len=args.max_seq_len,
            only_middle=(not args.with_past),
            max_struct_len=args.max_struct_len,
            struct_ratio=args.struct_ratio,
            ckpt=ckpt,
            no_order=args.no_order,
            bar_range_num = args.bar_range_num,
        )

    if args.experiment:
        with open(args.data_file, "rb") as f:
            testing_data = pickle.load(f)

        ckpt = utils.load_ckpt(args.ckpt_path)
        tokenizer = Tokenizer(args.vocab_file)

        assert ckpt.config.use_cp == args.cp
        assert ckpt.config.use_bar_cd == use_bar_cd
        assert ckpt.config.infilling == args.infilling

        model = TransformerXL(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        print("ckpt:", args.ckpt_path)
        print("epoch:", ckpt.epoch)
        print("training loss:", ckpt.training_loss)
        print("validation loss:", ckpt.validation_loss)
        print("use cp:", ckpt.config.use_cp)
        print("use bar count down:", ckpt.config.use_bar_cd)

        # collate testing data
        songs = []
        songs_struct = []
        for data in testing_data:
            past, target, future, struct = data['past'], data['target'], data['future'], data['struct']
            song = Song.copy(data['original_song'], bars=(past+target+future))
            step = 4
            for i in range(0, len(past), step):
                song.struct_indices.append([None, i, min(i+step, len(past))])
            song.struct_indices.append(["S", len(past), len(past)+len(target)])
            for i in range(len(past)+len(target), len(song.bars), step):
                song.struct_indices.append([None, i, min(i+step, len(song.bars))])

            song_struct = Song.copy(data['original_song'], bars=(struct))
            song_struct.struct_indices.append(["S", 0, len(struct)])

            songs.append(song)
            songs_struct.append(song_struct)

        with torch.no_grad():
            experiment(
                model,
                songs,
                songs_struct,
                args.seg_size,
                tokenizer,
                max_gen_len=args.max_gen_len,
                struct_ratio=args.struct_ratio,
                save_path=args.save_path,
                no_order=args.no_order,
            )

    # read user's input file to generate infilling result
    if args.generate:
        ckpt = utils.load_ckpt(args.ckpt_path)
        tokenizer = Tokenizer(args.vocab_file)

        assert ckpt.config.use_cp == args.cp
        assert ckpt.config.use_bar_cd == use_bar_cd
        assert ckpt.config.infilling == args.infilling

        model = TransformerXL(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        print("ckpt:", args.ckpt_path)
        print("epoch:", ckpt.epoch)
        print("training loss:", ckpt.training_loss)
        print("validation loss:", ckpt.validation_loss)
        print("use cp:", ckpt.config.use_cp)
        print("use bar count down:", ckpt.config.use_bar_cd)

        with torch.no_grad():
            generate(
                model,
                args.song_file,
                args.save_path,
                args.seg_size,
                tokenizer,
                struct_ratio=args.struct_ratio,
                cuda=args.cuda,
            )

def gen_data(data_file, small_size=16, split_ratio=0.1):
    data, err_cnt = preprocess.pop909("./dataset/pop909/POP909", "./dataset/pop909_struct/POP909")
    print("number of error in preprocessing:", err_cnt)

    sp = int(len(data) * split_ratio)

    with open(data_file, 'wb') as f:
        if os.path.exists(data_file):
            print(f"update existing file: {data_file}")
        else:
            print(f"create file: {data_file}")
        pickle.dump(data, f)

    with open(f"{data_file}.training", "wb") as f:
        pickle.dump(data[sp:], f)

    with open(f"{data_file}.testing", "wb") as f:
        pickle.dump(data[:sp], f)

    small_file = f"{data_file}.small"
    with open(small_file, 'wb') as f:
        if os.path.exists(small_file):
            print(f"update existing file: {small_file}")
        else:
            print(f"create file: {small_file}")
        pickle.dump(data[:small_size], f)

def load_data(data_file, track_sel=['melody', 'bridge', 'piano'], max_song_num=None, max_bar_num=32):
    print(f"Load data: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    if max_song_num is not None:
        data = data[:max_song_num]

    tmp = []
    for song in data:
        too_long = False
        for _, start, end in song.struct_indices:
            if end - start > max_bar_num:
                too_long = True
        if not too_long:
            tmp.append(song)
    data = tmp

    for song in data:
        for event in song.flatten_events():
            tmp = []
            for note in event.notes:
                if note.track in track_sel:
                    tmp.append(note)
            event.notes = tmp

    return data

def tokenize(songs_data, tokenizer, with_eos=True):
    songs = []
    bar_ids = []
    struct_ids = []
    struct_indices = []
    struct_ranges = []

    with mp.Pool() as pool:
        map_args = []
        for song in songs_data:
            map_args.append((song, tokenizer, with_eos))

        pbar = tqdm(desc="Tokenize", total=len(songs_data))
        for i, (song, bar_id, struct_id, struct_index, struct_range) in enumerate(pool.imap(tokenize_map, map_args)):
            assert len(song) == len(bar_id), f"song {songs_data[i].name}: len(song, bar_id) = ({len(song)}, {len(bar_id)})"
            songs.append(song)
            bar_ids.append(bar_id)
            struct_ids.append(struct_id)
            struct_indices.append(struct_index)
            struct_ranges.append(struct_range)
            pbar.update(1)
        pbar.close()

    return songs, bar_ids, struct_ids, struct_indices, struct_ranges

def tokenize_map(args):
    song, tokenizer, with_eos = args
    song, bar_id, struct_id, struct_index, struct_range = tokenizer.encode(song, with_eos=with_eos)
    song = tokenizer.token_to_id(song)
    return song, bar_id, struct_id, struct_index, struct_range

def default_optimizer(model, lr=1e-4):
    return torch.optim.Adam(model.parameters(), lr=lr)

def default_scheduler(optimizer, lr_max=1.0, lr_min=1.0, T=10, warmup=10, refine=50):
    lr_lambda = lambda epoch: (epoch+1)/warmup if epoch < warmup else \
                              lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos(epoch/T*math.pi)) if epoch < refine else lr_min
    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

def train(
    model,
    training_data,
    validation_data,
    epoch_num,
    batch_size,
    seg_size,
    cuda,
    config,
    tokenizer,
    save_path,
    accm_step=1,
    max_seq_len=None,
    only_middle=True,
    enable_validation=True,
    max_struct_len=512,
    struct_ratio=1.0,
    ckpt=None,
    no_order = False,
    bar_range_num = 8,
):
    model = model.cuda() if cuda else model

    optimizer = default_optimizer(model, lr=1e-4)
    scheduler = default_scheduler(optimizer, lr_max=1.0, lr_min=1.0, T=10, warmup=10, refine=50)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt.optim_state_dict)
        scheduler.load_state_dict(ckpt.sched_state_dict)

    tn_song_ids, tn_labels, tn_order_ids, tn_struct_ids, tn_struct_masks, tn_struct_seqs, tn_struct_seq_masks = \
        prepare_training_data(
            data_file=training_data,
            tokenizer=tokenizer,
            model=model,
            max_struct_len=max_struct_len,
            struct_ratio=struct_ratio,
            only_middle=only_middle,
            max_seq_len=max_seq_len,
            bar_range_num = bar_range_num,
        )

    val_song_ids, val_labels, val_order_ids, val_struct_ids, val_struct_masks, val_struct_seqs, val_struct_seq_masks = \
        prepare_training_data(
            data_file=validation_data,
            tokenizer=tokenizer,
            model=model,
            max_struct_len=max_struct_len,
            struct_ratio=struct_ratio,
            only_middle=only_middle,
            max_seq_len=max_seq_len,
            bar_range_num = bar_range_num,
        )

    if no_order: # disable order embedding
        tn_order_ids = torch.zeros(tn_order_ids.shape).to(tn_order_ids.dtype)
        val_order_ids = torch.zeros(val_order_ids.shape).to(val_order_ids.dtype)

    for epoch_idx in range(0 if ckpt is None else ckpt.epoch+1, epoch_num):
        model.train()
        pbar = tqdm(desc=f"epoch {epoch_idx+1}", total=len(tn_song_ids))
        total_loss = 0.0
        n_tokens = 0

        for batch_step, batch_idx in enumerate(range(0, len(tn_song_ids), batch_size)):
            """
            (bs, be): (batch start, batch end)
            """
            bs, be = batch_idx, batch_idx+batch_size
            batch = tn_song_ids[bs:be]
            mems = None
            mem_order_ids = None

            for seg_idx in range(0, max_seq_len, seg_size): # split a long sequence into small segments
                """
                (ss, se): (segment start, segment end)
                """
                ss, se = seg_idx, seg_idx+seg_size
                songs_batch = tn_song_ids[bs:be, ss:se].to(model.device)
                labels_batch = tn_labels[:, bs:be, ss+1:se+1].to(model.device)
                order_batch = tn_order_ids[bs:be, ss:se].to(model.device)
                sid_batch = tn_struct_ids[bs:be, ss:se].to(model.device)
                smask_batch = tn_struct_masks[bs:be, ss:se].to(model.device)

                output = model(
                    input_ids=songs_batch,
                    struct_ids=sid_batch,
                    struct_masks=smask_batch,
                    struct_seqs=tn_struct_seqs[bs:be].to(model.device),
                    struct_seq_masks=tn_struct_seq_masks[bs:be].to(model.device),
                    token_order_ids=order_batch,
                    mems=mems,
                    mem_order_ids=mem_order_ids,
                    labels=labels_batch,
                )
                loss = torch.mean(torch.stack(output.losses, dim=0))
                loss.backward()

                mems = output.mems
                mem_order_ids = output.mem_order_ids

                n = len(labels_batch[labels_batch != tokenizer.ignore_idx])
                total_loss += loss.item() * n
                n_tokens += n

            if (batch_step+1) % accm_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.update(len(batch))
        pbar.close()

        training_loss = total_loss / n_tokens
        print(" "*4, "training loss:", training_loss, " "*4, "learning rate:", scheduler.get_last_lr()[0], "" if enable_validation else "\n")

        validation_loss = None
        if enable_validation:
            with torch.no_grad():
                model.eval()
                pbar = tqdm(desc=f"validate", total=len(val_song_ids))
                total_loss = 0.0
                n_tokens = 0

                torch.set_printoptions(threshold=10_000)

                for batch_step, batch_idx in enumerate(range(0, len(val_song_ids), batch_size)):
                    """
                    (bs, be): (batch start, batch end)
                    """
                    bs, be = batch_idx, batch_idx+batch_size
                    batch = val_song_ids[bs:be]
                    mems = None
                    mem_order_ids = None
                    for seg_idx in range(0, max_seq_len, seg_size): # split a long sequence into small segments
                        """
                        (ss, se): (segment start, segment end)
                        """
                        ss, se = seg_idx, seg_idx+seg_size
                        songs_batch = val_song_ids[bs:be, ss:se].to(model.device)
                        labels_batch = val_labels[:, bs:be, ss+1:se+1].to(model.device)
                        order_batch = val_order_ids[bs:be, ss:se].to(model.device)
                        sid_batch = val_struct_ids[bs:be, ss:se].to(model.device)
                        smask_batch = val_struct_masks[bs:be, ss:se].to(model.device)

                        output = model(
                            input_ids=songs_batch,
                            struct_ids=sid_batch,
                            struct_masks=smask_batch,
                            struct_seqs=val_struct_seqs[bs:be].to(model.device),
                            struct_seq_masks=val_struct_seq_masks[bs:be].to(model.device),
                            token_order_ids=order_batch,
                            mems=mems,
                            mem_order_ids=mem_order_ids,
                            labels=labels_batch,
                        )
                        loss = torch.mean(torch.stack(output.losses, dim=0))

                        mems = output.mems
                        mem_order_ids = output.mem_order_ids

                        n = len(labels_batch[labels_batch != tokenizer.ignore_idx])
                        total_loss += loss.item() * n
                        n_tokens += n

                    pbar.update(len(batch))
                pbar.close()

                validation_loss = total_loss / n_tokens
                print(" "*4, "validation loss:", validation_loss, "\n")

        scheduler.step()
        utils.save_ckpt(save_path, epoch_idx, config, model, optimizer, scheduler, training_loss, validation_loss, tokenizer)

def prepare_training_data(
    data_file,
    tokenizer,
    model,
    max_struct_len=None,
    struct_ratio=None,
    only_middle=None,
    max_seq_len=None,
    bar_range_num = 8,
):
    songs_data = load_data(data_file, track_sel=['melody', 'bridge'])
    song_ids, bar_ids, struct_ids, struct_indices, struct_ranges = tokenize(songs_data, tokenizer)

    """
    struct_masks: mask to indicate whether doing cross attention to struct sequence for each input token
        0 => attend
        1 => not attend
    """
    print("the structure length is set to", max_struct_len)
    struct_seqs, struct_seq_masks, struct_masks = tokenizer.extract_struct(
        song_ids,
        struct_ids,
        struct_indices,
        max_struct_len=max_struct_len,
        struct_ratio=struct_ratio)

    song_ids, struct_ids, struct_indices, struct_masks, order_ids, ignore_labels, expand_idx = get_infilling_data(
        song_ids,
        struct_ranges,
        struct_ids,
        struct_indices,
        struct_masks,
        bar_ids,
        model,
        tokenizer,
        only_middle=only_middle,
        bar_range_num=bar_range_num)

    expand_idx = torch.LongTensor(expand_idx)[:, None, None].expand(-1, struct_seqs.shape[1], struct_seqs.shape[2])
    struct_seqs = torch.gather(struct_seqs, 0, expand_idx)
    struct_seq_masks = torch.gather(struct_seq_masks, 0, expand_idx)

    """
    To reduce training time, we set the max sequence length to (mean + 2*standard_deviation)
    """
    if max_seq_len is None:
        max_seq_len = utils.get_max_seq_len(song_ids)
    else:
        utils.get_max_seq_len(song_ids)
        print(f"max sequence length is set to {max_seq_len}")

    song_ids, _ = tokenizer.pad(song_ids, 0, max_seq_len, gen_mask=False)
    order_ids, _ = tokenizer.pad(order_ids, 0, max_seq_len, gen_mask=False)
    ignore_labels, _ = tokenizer.pad(ignore_labels, 0, max_seq_len, gen_mask=False)

    struct_ids, _ = tokenizer.pad(struct_ids, tokenizer.NONE_ID, max_seq_len, gen_mask=False)
    #struct_ids[struct_ids == tokenizer.NONE_ID] = 0 # set NONE_ID (-1) to 0
    struct_masks, _ = tokenizer.pad(struct_masks, 1, max_seq_len, gen_mask=False)
    struct_masks = struct_masks.float()

    """
    labels: (1, B, L)
    """
    labels = tokenizer.get_labels(song_ids, ignore_labels=ignore_labels)

    return song_ids, labels, order_ids, struct_ids, struct_masks, struct_seqs, struct_seq_masks,

def get_infilling_data(
    song_ids: list,
    struct_ranges: list,
    struct_ids: list,
    struct_indices: list,
    struct_masks: list,
    bar_ids: list,
    model,
    tokenizer,
    bar_range_num = 8,
    #max_seq_len = None,
    only_middle=True,
):
    """
    song_ids should not be padded here

    Output:
        song_ids: BOS -> (segment A) -> EOP -> (segment C) -> EOP -> (segment B) -> EOS(optional)
    """
    tn_song_ids = []
    tn_struct_ids = []
    tn_struct_indices = []
    tn_struct_masks = []
    tn_middle_indices = []
    tn_seg_ids = []
    tn_ignore_labels = []
    tn_expand_idx = []

    for i, song in enumerate(song_ids):
        assert len(song_ids[i]) == len(struct_ids[i]) == len(struct_indices[i])
        """
        sid, s_start, s_end: struct_id, struct_start, struct_end
        """
        for sid, s_start, s_end in struct_ranges[i][1:-1]: # avoid data without past content or future content
            #if sid == tokenizer.NONE_ID:
            #    continue # skip content without structure

            p_start = s_start - 1
            while p_start > 0 and bar_ids[i][s_start] - bar_ids[i][p_start] < bar_range_num:
                p_start -= 1

            f_end = s_end + 1
            while f_end < len(bar_ids[i]) and bar_ids[i][f_end] - bar_ids[i][s_end] < bar_range_num:
                f_end += 1

            """
            p_start(past_start) ------> s_start(struct_start), s_end(struct_end) ------> f_end(future_end)
            """
            no_eos = (song_ids[i][f_end-1] != tokenizer.eos_id())
            tn_song_ids.append(song_ids[i][p_start:f_end] + ([tokenizer.eos_id()] if no_eos else []))
            tn_struct_ids.append(struct_ids[i][p_start:f_end] + ([tokenizer.NONE_ID] if no_eos else []))
            tn_struct_indices.append(struct_indices[i][p_start:f_end] + ([struct_indices[i][f_end-1]+1] if no_eos else []))
            tn_struct_masks.append(struct_masks[i][p_start:f_end] + ([1] if no_eos else []))
            tn_middle_indices.append((s_start-p_start, s_end-p_start))
            tn_expand_idx.append(i)

    (tn_song_ids,
     tn_struct_ids,
     tn_struct_indices,
     tn_struct_masks,
     tn_seg_ids,
     tn_ignore_labels) = permute_data(
         tn_song_ids,
         tn_struct_ids,
         tn_struct_indices,
         tn_struct_masks,
         tn_middle_indices,
         model,
         tokenizer,
         only_middle=only_middle,
         ignore_middle_first=True,
     )

    for i in range(len(tn_song_ids)):
        l = len(tn_song_ids[i])
        assert len(tn_struct_ids[i]) == l
        assert len(tn_struct_indices[i]) == l
        assert len(tn_struct_masks[i]) == l
        assert len(tn_seg_ids[i]) == l
        assert len(tn_ignore_labels[i]) == l

    return (
        tn_song_ids,
        tn_struct_ids,
        tn_struct_indices,
        tn_struct_masks,
        tn_seg_ids,
        tn_ignore_labels,
        tn_expand_idx,
    )

def permute_data(song_ids, struct_ids, struct_indices, struct_masks, middle_indices, model, tokenizer, only_middle=False, ignore_middle_first=False):
    assert isinstance(song_ids, list)
    seg_ids = []
    ignore_labels = []

    for i, _ in enumerate(song_ids):
        B_start = middle_indices[i][0]
        B_end = middle_indices[i][1]
        assert B_start != B_end

        tmp = []
        s_tmp = []
        si_tmp = []
        m_tmp = []
        seg = []
        ignore = []

        tmp.extend(song_ids[i][:B_start] + [tokenizer.eop_id()]) # A
        s_tmp.extend(struct_ids[i][:B_start] + [tokenizer.NONE_ID])
        if struct_indices is not None:
            si_tmp.extend(struct_indices[i][:B_start] + [tokenizer.NONE_ID])
        if struct_masks is not None:
            m_tmp.extend(struct_masks[i][:B_start] + [1])
        seg.extend([model.past_id()] * (len(tmp)-len(seg)))
        if only_middle:
            ignore.extend([1] * (len(tmp)-len(ignore)))
        else:
            ignore.extend([0] * (len(tmp)-len(ignore)))
            ignore[-1] = 1 # EOP

        tmp.extend(song_ids[i][B_end: -1] + [tokenizer.eop_id()]) # C without EOS
        s_tmp.extend(struct_ids[i][B_end: -1] + [tokenizer.NONE_ID])
        if struct_indices is not None:
            si_tmp.extend(struct_indices[i][B_end: -1] + [tokenizer.NONE_ID])
        if struct_masks is not None:
            m_tmp.extend(struct_masks[i][B_end: -1] + [1])
        seg.extend([model.future_id()] * (len(tmp)-len(seg)))
        ignore.extend([1] * (len(tmp)-len(ignore)))

        tmp.extend(song_ids[i][B_start: B_end] + [song_ids[i][-1]]) # B + EOS
        s_tmp.extend(struct_ids[i][B_start: B_end] + [tokenizer.NONE_ID])
        if struct_indices is not None:
            si_tmp.extend(struct_indices[i][B_start: B_end] + [tokenizer.NONE_ID])
        if struct_masks is not None:
            m_tmp.extend(struct_masks[i][B_start: B_end] + [1])
        seg.extend([model.middle_id()] * (len(tmp)-len(seg)))
        if ignore_middle_first:
            ignore.extend([1] + ([0] * (len(tmp)-len(ignore)-1)))
        else:
            ignore.extend([0] * (len(tmp)-len(ignore)))

        song_ids[i] = tmp
        struct_ids[i] = s_tmp
        if struct_indices is not None:
            struct_indices[i] = s_tmp
        if struct_masks is not None:
            struct_masks[i] = m_tmp
        seg_ids.append(seg)
        ignore_labels.append(ignore)

    return song_ids, struct_ids, struct_indices, struct_masks, seg_ids, ignore_labels

def select_bars(songs, start, end):
    clipped_songs = []
    for song in songs:
        clipped_songs.append(song.clip(start, end))
    return clipped_songs

def prepare_generation_data(tgt_indices, struct_ranges, bar_ids, song_ids, struct_ids, struct_masks, model, tokenizer):
    song_ids = copy.deepcopy(song_ids)
    struct_ids = copy.deepcopy(struct_ids)
    struct_masks = copy.deepcopy(struct_masks)
    struct_tgt_ids = []
    struct_tgt_lens = []
    seg_ids = []
    past_ids = []
    future_ids = []
    middle_ids = []

    for i in range(len(song_ids)):
        tgt_id, tgt_start, tgt_end = struct_ranges[i][tgt_indices[i]]
        song_id = []
        struct_id = []
        struct_mask = []
        seg_id = []

        struct_tgt_ids.append(tgt_id)
        struct_tgt_lens.append(bar_ids[i][tgt_end] - bar_ids[i][tgt_start])

        # past
        song_id.extend(song_ids[i][:tgt_start] + [tokenizer.eop_id()])
        past_ids.append(song_ids[i][:tgt_start])
        middle_ids.append(song_ids[i][tgt_start:tgt_end])
        struct_id.extend(struct_ids[i][:tgt_start] + [tokenizer.NONE_ID])
        struct_mask.extend(struct_masks[i][:tgt_start] + [1])
        seg_id.extend([model.past_id()] * len(song_id))

        # future
        song_id.extend(song_ids[i][tgt_end:] + [tokenizer.eop_id()])
        future_ids.append(song_ids[i][tgt_end:])
        struct_id.extend(struct_ids[i][tgt_end:] + [tokenizer.NONE_ID])
        struct_mask.extend(struct_masks[i][tgt_end:] + [1])
        seg_id.extend([model.future_id()] * (len(song_ids[i][tgt_end:]) + 1))

        # store data
        song_ids[i] = song_id
        struct_ids[i] = struct_id
        struct_masks[i] = struct_mask
        seg_ids.append(seg_id)

    return song_ids, struct_ids, struct_masks, struct_tgt_ids, struct_tgt_lens, seg_ids, past_ids, middle_ids, future_ids

def get_experiment_tgt_indices(struct_ranges):
    tgt_indices = []
    for struct_range in struct_ranges:
        for i, (sid, start, end) in enumerate(struct_range):
            if sid == 0:
                tgt_indices.append(i)
                break
    assert len(tgt_indices) == len(struct_ranges)
    return tgt_indices

def experiment(model, songs, songs_struct, seg_size, tokenizer, max_gen_len, save_path=None, cuda=True, bar_num=None, struct_ratio=1.0, no_order=False):
    model.eval()
    model = model.cuda() if cuda else model

    song_ids, _, struct_ids, struct_indices, _ = tokenize(songs_struct, tokenizer)
    struct_seqs, struct_seq_masks, _ = tokenizer.extract_struct(song_ids, struct_ids, struct_indices, struct_ratio=struct_ratio)

    song_ids, bar_ids, struct_ids, struct_indices, struct_ranges = tokenize(songs, tokenizer, with_eos=False)
    tgt_indices = get_experiment_tgt_indices(struct_ranges)

    _, _, struct_masks = tokenizer.extract_struct(song_ids, struct_ids, struct_indices, struct_ratio=struct_ratio)

    song_ids, struct_ids, struct_masks, struct_tgt_ids, struct_tgt_lens, seg_ids, past_ids, middle_ids, future_ids = \
        prepare_generation_data(tgt_indices, struct_ranges, bar_ids, song_ids, struct_ids, struct_masks, model, tokenizer)

    pbar = tqdm(desc="Generating", total=len(song_ids))
    for i in range(len(song_ids)):
        song_idx = i

        bar_num = struct_tgt_lens[i]
        result = [tokenizer.bar_id(bar_num)]
        song_id = torch.LongTensor(song_ids[i] + [tokenizer.bar_id(bar_num)])[None].to(model.device)
        struct_id = torch.LongTensor(struct_ids[i] + [struct_tgt_ids[i]])[None].to(model.device)
        struct_mask = torch.LongTensor(struct_masks[i] + [0])[None].to(model.device)
        seg_id = torch.LongTensor(seg_ids[i] + [model.middle_id()])[None].to(model.device)
        if no_order:
            seg_id = torch.zeros(seg_id.shape).to(seg_id.dtype).to(model.device)
        struct_seq = struct_seqs[i][None].to(model.device)
        struct_seq_mask = struct_seq_masks[i][None].to(model.device)

        struct = np.trim_zeros(struct_seq[0][struct_tgt_ids[i]].cpu().numpy())

        # limit segment length not longer than memory length
        seg_size = model.mem_len if model.mem_len < seg_size else seg_size

        """
        generate momery
        """
        gen_id = None
        mems = None

        output = model(
            input_ids=song_id,
            struct_ids=struct_id,
            struct_masks=struct_mask,
            struct_seqs=struct_seq,
            struct_seq_masks=struct_seq_mask,
            token_order_ids=seg_id,
            mems=None
        )
        mems = output.mems
        mem_order_ids = output.mem_order_ids

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
        prev_bar_num = bar_num
        while True:
            song_id = gen_id[None, None].to(model.device)
            seg_id = torch.LongTensor([model.middle_id()])[None].to(model.device)
            if no_order:
                seg_id = torch.LongTensor([0])[None].to(model.device)
            struct_id = torch.LongTensor([struct_tgt_ids[i]])[None].to(model.device)
            struct_mask = None

            #output = model(input_ids=segs, token_type_ids=type_ids, mems=mems)
            output = model(
                input_ids=song_id,
                struct_ids=struct_id,
                struct_masks=struct_mask,
                struct_seqs=struct_seq,
                struct_seq_masks=struct_seq_mask,
                token_order_ids=seg_id,
                mems=mems,
                mem_order_ids=mem_order_ids,
            )
            mems = output.mems
            mem_order_ids = output.mem_order_ids

            #gen_id = torch.argmax(output.pred_scores, dim=-1)[0, -1].item()
            while True:
                output_ids = tokenizer.sample(output.pred_scores)
                gen_id = output_ids[0, -1]
                if tokenizer.is_legal(gen_id):
                    break
                else:
                    print("illegal generated id:", tokenizer.id_to_token(gen_id))
            if tokenizer.is_bar(gen_id):
                if prev_bar_num == 1:
                    break
                else:
                    prev_bar_num = int(tokenizer.id_to_token(gen_id).rsplit(")")[0].split("(")[1])
            if tokenizer.is_eos(gen_id):
                break
            result.append(gen_id.tolist())
            #if len(result) >= max_gen_len:
            #    break
        song_id = past_ids[i] + result + future_ids[i]

        #song_id, past_id, middle_id, future_id, result_id = gen_song_ids[i]
        gen_song = tokenizer.decode(tokenizer.id_to_token(song_id), Song.copy(songs[i], with_content=False))
        past = tokenizer.decode(tokenizer.id_to_token(past_ids[i]), Song.copy(songs[i], with_content=False))
        middle = tokenizer.decode(tokenizer.id_to_token(middle_ids[i]), Song.copy(songs[i], with_content=False))
        future = tokenizer.decode(tokenizer.id_to_token(future_ids[i]), Song.copy(songs[i], with_content=False))
        result = tokenizer.decode(tokenizer.id_to_token(result), Song.copy(songs[i], with_content=False))
        struct = tokenizer.decode(tokenizer.id_to_token(struct), Song.copy(songs[i], with_content=False))

        save_dir = save_path
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        gen_song.save(os.path.join(save_dir, f"{song_idx}.midi"))
        gen_song.save_fig(os.path.join(save_dir, f"{song_idx}.png"))
        past.save(os.path.join(save_dir, f"{song_idx}_past.midi"))
        middle.save(os.path.join(save_dir, f"{song_idx}_middle.midi"))
        future.save(os.path.join(save_dir, f"{song_idx}_future.midi"))
        struct.save(os.path.join(save_dir, f"{song_idx}_struct.midi"))
        result.save(os.path.join(save_dir, f"{song_idx}_result.midi"))
        with open(os.path.join(save_dir, f"{song_idx}_past.pickle"), 'wb') as f:
            pickle.dump(past, f)
        with open(os.path.join(save_dir, f"{song_idx}_middle.pickle"), 'wb') as f:
            pickle.dump(middle, f)
        with open(os.path.join(save_dir, f"{song_idx}_future.pickle"), 'wb') as f:
            pickle.dump(future, f)
        with open(os.path.join(save_dir, f"{song_idx}_result.pickle"), 'wb') as f:
            pickle.dump(result, f)
        with open(os.path.join(save_dir, f"{song_idx}_struct.pickle"), 'wb') as f:
            pickle.dump(struct, f)
        pbar.update(1)
    pbar.close()

def generate(model, song_file, save_dir, seg_size, tokenizer, struct_ratio=1.0, cuda=True):
    song, s_ref_idx, s_tgt_idx = parse_song_file(song_file)
    song_token, bar_id, struct_id, struct_index, struct_range = tokenizer.encode(song, with_eos=False)
    song_id = tokenizer.token_to_id(song_token)

    struct_seqs, struct_seq_masks, struct_masks = \
        tokenizer.extract_struct([song_id], [struct_id], [struct_index], struct_ratio=struct_ratio)
    song_ids, struct_ids, struct_masks, struct_tgt_ids, struct_tgt_lens, seg_ids, past_ids, middle_ids, future_ids = \
        prepare_generation_data([s_tgt_idx], [struct_range], [bar_id], [song_id], [struct_id], struct_masks, model, tokenizer)

    model.eval()
    model = model.cuda() if cuda else model

    pbar = tqdm(desc="Generating", total=len(song_ids))
    songs = [song]

    if not os .path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(song_ids)):
        song_idx = i

        bar_num = struct_tgt_lens[i]
        result = [tokenizer.bar_id(bar_num)]
        song_id = torch.LongTensor(song_ids[i] + [tokenizer.bar_id(bar_num)])[None].to(model.device)
        struct_id = torch.LongTensor(struct_ids[i] + [struct_tgt_ids[i]])[None].to(model.device)
        struct_mask = torch.LongTensor(struct_masks[i] + [0])[None].to(model.device)
        seg_id = torch.LongTensor(seg_ids[i] + [model.middle_id()])[None].to(model.device)
        struct_seq = struct_seqs[i][None].to(model.device)
        struct_seq_mask = struct_seq_masks[i][None].to(model.device)

        struct = np.trim_zeros(struct_seq[0][struct_tgt_ids[i]].cpu().numpy())

        # limit segment length not longer than memory length
        seg_size = model.mem_len if model.mem_len < seg_size else seg_size

        """
        generate momery
        """
        gen_id = None
        mems = None

        output = model(
            input_ids=song_id,
            struct_ids=struct_id,
            struct_masks=struct_mask,
            struct_seqs=struct_seq,
            struct_seq_masks=struct_seq_mask,
            token_order_ids=seg_id,
            mems=None
        )
        mems = output.mems
        mem_order_ids = output.mem_order_ids

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
        prev_bar_num = bar_num
        while True:
            song_id = gen_id[None, None].to(model.device)
            seg_id = torch.LongTensor([model.middle_id()])[None].to(model.device)
            struct_id = torch.LongTensor([struct_tgt_ids[i]])[None].to(model.device)
            struct_mask = None

            output = model(
                input_ids=song_id,
                struct_ids=struct_id,
                struct_masks=struct_mask,
                struct_seqs=struct_seq,
                struct_seq_masks=struct_seq_mask,
                token_order_ids=seg_id,
                mems=mems,
                mem_order_ids=mem_order_ids,
            )
            mems = output.mems
            mem_order_ids = output.mem_order_ids

            while True:
                output_ids = tokenizer.sample(output.pred_scores)
                gen_id = output_ids[0, -1]
                if tokenizer.is_legal(gen_id):
                    break
                else:
                    print("illegal generated id:", tokenizer.id_to_token(gen_id))
            if tokenizer.is_bar(gen_id):
                if prev_bar_num == 1:
                    break
                else:
                    prev_bar_num = int(tokenizer.id_to_token(gen_id).rsplit(")")[0].split("(")[1])
            if tokenizer.is_eos(gen_id):
                break
            result.append(gen_id.tolist())
        song_id = past_ids[i] + result + future_ids[i]

        gen_song = tokenizer.decode(tokenizer.id_to_token(song_id), Song.copy(songs[i], with_content=False))
        past = tokenizer.decode(tokenizer.id_to_token(past_ids[i]), Song.copy(songs[i], with_content=False))
        middle = tokenizer.decode(tokenizer.id_to_token(middle_ids[i]), Song.copy(songs[i], with_content=False))
        future = tokenizer.decode(tokenizer.id_to_token(future_ids[i]), Song.copy(songs[i], with_content=False))
        result = tokenizer.decode(tokenizer.id_to_token(result), Song.copy(songs[i], with_content=False))
        struct = tokenizer.decode(tokenizer.id_to_token(struct), Song.copy(songs[i], with_content=False))

        gen_song.save(os.path.join(save_dir, f"{song_idx}.midi"))
        gen_song.save_fig(os.path.join(save_dir, f"{song_idx}.png"))
        middle.save(os.path.join(save_dir, f"{song_idx}_middle.midi"))
        struct.save(os.path.join(save_dir, f"{song_idx}_struct.midi"))
        result.save(os.path.join(save_dir, f"{song_idx}_result.midi"))
        with open(os.path.join(save_dir, f"{song_idx}_past.pickle"), 'wb') as f:
            pickle.dump(past, f)
        with open(os.path.join(save_dir, f"{song_idx}_middle.pickle"), 'wb') as f:
            pickle.dump(middle, f)
        with open(os.path.join(save_dir, f"{song_idx}_future.pickle"), 'wb') as f:
            pickle.dump(future, f)
        with open(os.path.join(save_dir, f"{song_idx}_result.pickle"), 'wb') as f:
            pickle.dump(result, f)
        with open(os.path.join(save_dir, f"{song_idx}_struct.pickle"), 'wb') as f:
            pickle.dump(struct, f)
        pbar.update(1)
    pbar.close()

def parse_song_file(song_file):
    with open(song_file, 'r') as f:
        bpm, beat_per_bar = map(int, f.readline().strip().split())
        beat_division = 4

        song = Song("", beat_per_bar, beat_division, bpm)
        s_count = 0
        b_count = 0
        struct = None
        s_start = 0
        s_end = 0
        ref = None
        tgt = None

        for line in f.readlines():
            line = line.strip()
            tokens = line.split()

            if len(tokens) in (1, 2): # struct
                if s_count != 0:
                    s_end = b_count
                    song.struct_indices.append((struct, s_start, s_end))

                struct = tokens[0]

                if struct.lower() in ("x", "i", "o"):
                    struct = None

                if len(tokens) == 2:
                    tag = tokens[1]
                    if tag == "S":
                        ref = s_count
                    elif tag == "T":
                        tgt = s_count
                    else:
                        raise Exception(f"Unknow tag: {tag}")

                s_count += 1
                s_start = b_count
            else: # bar
                bar = Bar.new(bpm, beat_per_bar, beat_division)
                for i in range(0, len(tokens), 4):
                    tempo, pos, pitch, dur = map(int, tokens[i:i+4])
                    note = Note(pitch, 100, pos, dur)
                    bar.events[pos].tempo = tempo
                    bar.events[pos].notes.append(note)
                song.bars.append(bar)
                b_count += 1

        if s_count != 0: # append the last phrase
            s_end = b_count
            song.struct_indices.append((struct, s_start, s_end))
    return song ,ref, tgt


if __name__ == "__main__":
    main()
