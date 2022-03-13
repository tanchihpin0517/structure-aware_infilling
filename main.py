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

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seg-size', type=int, default=1024)
    parser.add_argument('--epoch-num', type=int, default=1, help='number of training epochs')

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

    if args.preprocess:
        gen_data(args.data_file, small_size=16)
        exit()

    #data = load_data(args.data_file, args.preprocess, melody_only=True, max_song_num=16)
    songs_data = load_data(args.data_file, track_sel=['melody', 'bridge'])
    #songs_data = load_data(args.data_file, track_sel=['melody', 'bridge', 'piano'])

    if args.train:
        utils.check_save_path(args.save_path)

        tokenizer = Tokenizer(args.vocab_file, use_cp=use_cp)
        song_ids = tokenize(songs_data, tokenizer)
        training_song_ids = song_ids[len(song_ids)*(10-args.training_split_ratio)//10:]

        if args.model == "transformer_xl":
            config = TransformerXLConfig(
                tokenizer.vocab_size(),
                d_model = args.dim_model,
                n_head = args.num_head,
                d_inner = args.dim_inner,
                n_layer = args.num_layer,
                mem_len = args.mem_len,
                clamp_len = args.mem_len,
                use_cp=use_cp,
                d_subembed=args.dim_subembed,
                class_ranges=tokenizer.class_ranges()
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
                args.save_path
            )
        elif args.model == "xlnet":
            config = XLNetConfig(
                tokenizer.vocab_size(),
                d_model = args.dim_model,
                d_inner = args.dim_inner,
                n_head = args.num_head,
                n_layer = args.num_layer,
                mem_len = args.mem_len,
                attn_type="bi",
            )
            model = XLNet(config)
            train_xlnet(
                model,
                training_songs,
                args.epoch_num,
                args.batch_size,
                args.seg_size,
                args.cuda,
                config,
                tokenizer,
                args.save_path,
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
                prompts = select_first_n_bar(songs, 8)
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
                gen_song = tokenizer.decode(tokenizer.id_to_token(gen_song_ids[i]), Song.copy(prompts[i], with_content=False))

                save_dir = os.path.join(args.save_path, f"{math.floor(ckpt.loss*10)/10.0}")
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                gen_song.save(os.path.join(save_dir, f"{gen_song.name}.midi"))
                #with open(f"./gen_midi/{song.name}.txt", "w") as f:
                #    f.write("\n".join(text))
        elif args.model == 'xlnet':
            songs = songs[:8]
            ckpt = torch.load(args.ckpt_path)
            model = XLNet(ckpt.config)
            model.load_state_dict(ckpt.model_state_dict)
            tokenizer = ckpt.vocab
            print("ckpt:", "args.ckpt_path", ", epoch:", ckpt.epoch, ", loss:", ckpt.loss)
            with torch.no_grad():
                prompts_past = select_n_bar(songs, tokenizer, 0, 8)
                prompts_future = select_n_bar(songs, tokenizer, 21, 8)
                result_ids = generate_xlnet(model, prompts_past, prompts_future, len(prompts_past[0])+len(prompts_future[0])+64,
                                  args.cuda, args.seg_size, tokenizer)

                '''
                for i in range(len(prompt_ids)):
                    song = Song(); song.info_copy(prompt_ids[i])
                    song.extend(prompt_ids[i] + result_ids[i])
                    midi_data, text = token_ids_to_midi(song, tokenizer)
                    save_dir = f"./gen_midi/{math.floor(ckpt.loss*10)/10.0}"
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    midi_data.write(os.path.join(save_dir, f"{song.name}.midi"))
                    #with open(f"./gen_midi/{song.name}.txt", "w") as f:
                    #    f.write("\n".join(text))
                '''
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

def tokenize(songs_data, tokenizer, with_eos=True) -> list:
    songs = []
    pbar = tqdm(desc="Tokenize", total=len(songs_data))
    for song in songs_data:
        song = tokenizer.encode(song, with_eos=with_eos)
        song = tokenizer.token_to_id(song)
        songs.append(song)
        pbar.update(1)
    pbar.close()

    return songs

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

def default_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4)

def train_transxl(model, songs, epoch_num, batch_size, seg_size, cuda, config, tokenizer, save_path):
    model.train()
    model = model.cuda() if cuda else model

    """
    To reduce training time, we set the max sequence length to (mean + 2*standard_deviation)
    """
    max_seq_len = get_max_seq_len(songs)

    songs, _ = tokenizer.pad(songs, 0, max_seq_len)

    """
    if use_cp:
        labels: (C, B, L)
    else:
        labels: (1, B, L)
    """
    labels = tokenizer.get_labels(songs)

    optimizer = default_optimizer(model)

    for epoch_idx in range(epoch_num):
        pbar = tqdm(desc=f"epoch {epoch_idx+1}", total=len(songs))
        running_loss = 0.0
        n_tokens = 0

        for batch_idx in range(0, len(songs), batch_size):
            batch = songs[batch_idx:batch_idx+batch_size]
            mems = None
            for seg_idx in range(0, max_seq_len, seg_size): # split a long sequence into small segments
                songs_batch = songs[batch_idx:batch_idx+batch_size, seg_idx:seg_idx+seg_size]
                labels_batch = labels[:, batch_idx:batch_idx+batch_size, seg_idx+1:seg_idx+seg_size+1]
                if cuda:
                    songs_batch = songs_batch.cuda()
                    labels_batch = labels_batch.cuda()

                optimizer.zero_grad()
                output = model(input_ids=songs_batch, mems=mems, labels=labels_batch)
                loss = torch.sum(torch.stack(output.losses, dim=0))
                loss.backward()
                optimizer.step()

                mems = output.mems
                running_loss += loss.item()
                n_tokens += len(songs_batch[songs_batch != 0])

            pbar.update(len(batch))
        pbar.close()
        #running_loss /= ((len(songs) / batch_size) * (max_seq_len / seg_size))
        running_loss = running_loss / n_tokens
        print(" "*4, "average loss:", running_loss, "\n")

        save_ckpt(save_path, epoch_idx, config, model, optimizer, running_loss, tokenizer)

def train_xlnet(model, songs, epoch_num, batch_size, seg_size, cuda, config, tokenizer, save_path):
    model.train()
    model = model.cuda() if cuda else model

    """
    To reduce training time, we set the max sequence length to (mean + 2*standard_deviation)
    """
    max_seq_len = get_max_seq_len(songs)

    # attention mask: 0 -> not attend, 1 -> attend
    songs, attention_masks = to_tensor_with_zero_padding(songs, max_seq_len)

    # permutation mask: 0 -> attend, 1 -> not attend
    permutation_masks, target_mappings, tgt_labels = gen_permutation_mask_and_target(songs, max_seq_len, max_seq_len//3, max_seq_len//3*2)

    optimizer = default_optimizer(model)

    for epoch_idx in range(epoch_num):
        pbar = tqdm(desc=f"epoch {epoch_idx+1}", total=len(songs))
        running_loss = 0.0
        n_tokens = 0

        for batch_idx in range(0, len(songs), batch_size):
            bs, be = batch_idx, batch_idx+batch_size
            mems = None

            for seg_idx in range(0, max_seq_len, seg_size): # split a long sequence into small segments
                ss, se = seg_idx, seg_idx+seg_size

                segs = songs[bs:be, ss:se].to(model.device)
                labels = tgt_labels[bs:be, ss:se].to(model.device)
                attn_mask = attention_masks[bs:be, ss:se].to(model.device)
                perm_mask = permutation_masks[bs:be, ss:se, ss:se].to(model.device)
                tgt_mapping = target_mappings[bs:be, ss:se, ss:se].to(model.device)

                optimizer.zero_grad()
                output = model(
                    input_ids=segs,
                    labels=labels,
                    attention_mask=attn_mask,
                    mems=mems,
                    perm_mask=perm_mask,
                    target_mapping=tgt_mapping,
                )
                output.losses.backward()
                optimizer.step()

                mems = output.mems
                running_loss += output.losses.item()
                n_tokens += len(labels[labels != 0])

            pbar.update(len(songs[bs:be]))
        pbar.close()

        #running_loss /= ((len(songs) / batch_size) * (max_seq_len / seg_size))
        running_loss = running_loss / n_tokens
        print(" "*4, "losses sum:", running_loss, "\n")

        save_ckpt(save_path, epoch_idx, config, model, optimizer, running_loss, tokenizer)

def gen_permutation_mask_and_target(songs, max_seq_len, B_start, B_end):
    """
    permutation: [A, B, C] -> [A, C, B]
    mask: 0 -> attend, 1 -> not attend
    """
    #B_start, B_end = max_seq_len//3*1, max_seq_len//3*2 # [B_start, B_end)
    A_len, B_len, C_len = B_start, B_end-B_start, max_seq_len-B_end

    mask = torch.zeros(max_seq_len, max_seq_len)

    # part A
    mask_a = torch.triu(torch.ones((A_len, A_len)))
    mask[:B_start, :B_start] += mask_a
    mask[:B_start, B_start:] += 1.0

    # part C
    mask_c = torch.triu(torch.ones((C_len, C_len)))
    mask[B_end:, B_end:] += mask_c
    mask[B_start:B_end, B_start:B_end] += 1.0

    # part B
    mask_b = torch.triu(torch.ones((B_len, B_len)))
    mask[B_start:B_end, B_start:B_end] += mask_b

    mask = (mask > 0)[None, :, :].expand(len(songs), -1, -1).to(torch.float)

    mapping = torch.eye(max_seq_len, max_seq_len)[None, :, :].expand(len(songs), -1, -1).to(torch.float)
    label = torch.zeros(len(songs), max_seq_len).to(torch.long)
    label[:, B_start:B_end] = songs[:, B_start:B_end]

    return mask, mapping, label

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

def select_first_n_bar(songs, n_bar):
    clipped_songs = []
    for song in songs:
        clipped_songs.append(song.clip(0, n_bar))
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
        generate momery embeddings
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
        total_gen_num = 1
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
    assert len(prompts_past) == len(prompts_future)
    pbar = tqdm(desc="Generating", total=len(prompts_past))
    for i in range(len(prompts_past)):
        past, future = prompts_past[i], prompts_future[i]
        prompt = Song(); prompt.info_copy(past)
        prompt.extend(past)
        prompt.extend([0] * (gen_len-len(past)-len(future)))
        prompt.extend(future)

        result = Song(); result.info_copy(prompt)
        total_len = len(prompt)
        gen_start = len(past)
        gen_end = total_len-len(future)
        prompt = torch.tensor(prompt, dtype=torch.long)[None, :] # make shape to (1, L)
        permutation_masks, target_mappings, tgt_labels = gen_permutation_mask_and_target(prompt, total_len, gen_start, gen_end)

        # limit segment length not longer than memory length
        seg_size = model.mem_len if model.mem_len < seg_size else seg_size

        mems = None
        for seg_idx in range(gen_start, gen_end, seg_size): # split a long sequence into small segments
            ss, se = seg_idx, seg_idx+seg_size

            for tgt_idx in range(ss, se):
                segs = prompt[:, ss:se].to(model.device)
                #labels = tgt_labels[:, ss:se].to(model.device)
                #attn_mask = attention_masks[:, ss:se].to(model.device)
                perm_mask = permutation_masks[:, ss:se, ss:se].to(model.device)
                tgt_mapping = target_mappings[:, tgt_idx:tgt_idx+1, ss:se].to(model.device)

                output = model(
                    input_ids=segs,
                    labels=None,
                    attention_mask=None,
                    mems=mems,
                    perm_mask=perm_mask,
                    target_mapping=tgt_mapping,
                )
                gen_id = utils.nucleus(output.pred_scores)[0, -1].item()
                result.append(gen_id)
                prompt[0][tgt_idx] = gen_id
                print(tokenizer[gen_id])

            mems = output.mems

        exit()

        """
        generate momery embeddings
        """
        mems = None
        first_gen_id = None
        for seg_idx in range(0, len(prompt), seg_size): # split a long sequence into small segments
            segs = prompt[None, seg_idx:seg_idx+seg_size]
            segs = segs.cuda() if cuda else segs

            output = model(input_ids=segs, mems=mems)
            mems = output.mems

            output_ids = torch.argmax(output.pred_scores, dim=-1)
            first_gen_id = output_ids[0, -1]

        """
        generate new contents
        """
        #output_ids = torch.argmax(output.pred_scores, dim=-1)
        total_gen_num = 0
        gen_id = first_gen_id.item()
        while True:
            segs = torch.tensor(gen_id, dtype=torch.int).view(1,1)
            segs = segs.cuda() if cuda else segs

            output = model(input_ids=segs, mems=mems)
            mems = output.mems

            #gen_id = torch.argmax(output.pred_scores, dim=-1)[0, -1].item()
            gen_id = model.nucleus(output.pred_scores)[0, -1].item()
            total_gen_num += 1
            #print(tokenizer[gen_id])
            result.append(gen_id)
            if tokenizer[gen_id] == "EOS":
                break
            if not total_gen_num < max_gen_len:
                break
        result_ids.append(result)
        pbar.update(1)
    pbar.close()

    return result_ids

def token_ids_to_midi(song, tokenizer):
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=song.bpm)
    inst = pretty_midi.Instrument(program=0)
    text = []

    global_beat = 0
    beat_time = 60 / song.bpm
    for token_id in song:
        if token_id == tokenizer["BOS"] or token_id == tokenizer["EOS"]:
            continue
        token = tokenizer[token_id]
        value = int(token.split("(")[1].split(")")[0])
        if "Bar" in token:
            global_beat += song.beat_per_bar
        elif "Label" in token:
            label = value
        elif "Tempo" in token:
            tempo = value
        elif "Position" in token:
            subbeat = value
        elif "Pitch" in token:
            pitch = value
        elif "Velocity" in token:
            velocity = value
        elif "Duration" in token:
            duration = value
            onbeat = global_beat + subbeat / song.beat_division
            offbeat = onbeat + duration / song.beat_division
            #print(onbeat, offbeat)
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=onbeat*beat_time, end=offbeat*beat_time)
            inst.notes.append(note)
        else:
            raise Exception(f'Unknow token: {token}')

        text.append(token)

    midi_data.instruments.append(inst)
    return midi_data ,text

if __name__ == "__main__":
    main()
