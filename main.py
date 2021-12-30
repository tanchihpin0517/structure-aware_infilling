import os
import argparse
import preprocess
import pickle
from misc import Song, Vocabulary, Checkpoint
from tqdm import tqdm
from model import Config, Transformer
import torch
import torch.nn.functional as F
import numpy as np
import math
import pretty_midi

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--generate', default=False, action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--vocab-file', type=str, default='dataset/vocab.txt')
    parser.add_argument('--data-file', type=str, default='dataset/pop909.pickle')
    parser.add_argument('--preprocess', default=False, action='store_true')
    parser.add_argument('--save-path', type=str, default="trained_model/loss_nnn.ckpt")
    parser.add_argument('--ckpt-path', type=str, default=None)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seg-size', type=int, default=128)
    parser.add_argument('--epoch-num', type=int, default=1000, help='number of training epochs')

    parser.add_argument('--training-split-ratio', type=int, default=9)
    parser.add_argument('--max-gen-len', type=int, default=256, help='number of tokens in generation')
    return parser.parse_args()

def main():
    args = parse_args()

    #data = load_data(args.data_file, args.preprocess, melody_only=True, max_song_num=16)
    data = load_data(args.data_file, args.preprocess, melody_only=True)
    vocab = load_vocab(args.vocab_file)
    songs = tokenize(data, vocab, split_empty_bar=0)

    if args.train:
        training_songs = songs[len(songs)*(10-args.training_split_ratio)//10:]
        config = Config(
            vocab.size(),
            d_embed = 256,
            d_model = 256,
            n_head = 8,
            d_inner = 1024,
            n_layer = 8,
            mem_len = args.seg_size,
        )
        model = Transformer(config)
        train(model, training_songs, args.epoch_num, args.batch_size, args.seg_size, args.cuda,
              config, vocab, args.save_path)

    if args.test:
        ckpt = torch.load(args.ckpt_path)
        model = Transformer(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        vocab = ckpt.vocab
        print("ckpt:", "args.ckpt_path", ", epoch:", ckpt.epoch, ", loss:", ckpt.loss)
        with torch.no_grad():
            test(model, songs[0],
                 args.cuda, args.seg_size, vocab)

    if args.generate:
        songs = songs[:5]
        ckpt = torch.load(args.ckpt_path)
        model = Transformer(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        vocab = ckpt.vocab
        print("ckpt:", "args.ckpt_path", ", epoch:", ckpt.epoch, ", loss:", ckpt.loss)
        with torch.no_grad():
            prompt_ids = select_first_n_bar(songs, vocab, 16)
            result_ids = generate(model, prompt_ids,
                              args.cuda, args.seg_size, vocab, max_gen_len=args.max_gen_len)

            for i in range(len(prompt_ids)):
                song = Song(); song.info_copy(prompt_ids[i])
                song.extend(prompt_ids[i] + result_ids[i])
                midi_data, text = token_ids_to_midi(song, vocab)
                midi_data.write(f"./gen_midi/{song.name}.midi")
                with open(f"./gen_midi/{song.name}.txt", "w") as f:
                    f.write("\n".join(text))


def load_data(data_file, preproc, melody_only=False, max_song_num=None):
    # load the data file if exists, otherwise re-preprocess.
    if os.path.exists(data_file) and not preproc:
        print("Data file already exists: " + data_file)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        if preproc:
            print("Force preprocess: " + data_file)
        else:
            print("Data file doesn't exist: " + data_file)
        data = preprocess.pop909("./dataset/pop909/POP909", "./dataset/pop909_struct/POP909")

        if max_song_num is not None:
            data = data[:max_song_num]

        if melody_only:
            for song in data:
                for event in song:
                    tmp = []
                    for note in event.notes:
                        if note.track == 'melody':
                            tmp.append(note)
                    event.notes = tmp

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

    return data

def load_vocab(vocab_file):
    # load vocabulary file, otherwise create a new one.
    if os.path.exists(vocab_file):
        print("Vocabulary file already exists: " + vocab_file)
        vocab = Vocabulary(vocab_file)
    else:
        print("Vocabulary file doesn't exist. Create a new one: " + vocab_file)
        vocab = Vocabulary()
        vocab.save(vocab_file)

    return vocab

def tokenize(data, vocab, split_empty_bar=0):
    songs = []
    pbar = tqdm(desc="Tokenize", total=len(data))
    for song in data:
        label_count = 0
        label_map = {}
        tokens = ["BOS"]
        bar_idx = 0
        for i, event in enumerate(song):
            if event.struct is not None:
                label = f"Label({event.struct})"
                if label not in label_map:
                    label_map[label] = f"Label({label_count})"
                    label_count += 1
                tokens.append(label_map[label])
            if event.bar is not None:
                if event.bar >= 0:
                    tokens.append("Bar(1)")
                bar_idx = i
            for note in event.notes:
                if note.duration > 32: # clip if note longer than 8 beats
                    note.duration = 32
                tokens.extend([f"Position({i-bar_idx})", f"Pitch({note.pitch})", f"Duration({note.duration})"])
        tokens.append("EOS")

        cont_bar = 0
        splits = [["Bar(1)"]]
        for token in tokens[1:-2]: # exclude BOS and EOS
            splits[-1].append(token)
            if token == "Bar(1)":
                cont_bar += 1
            else:
                cont_bar = 0

            if split_empty_bar > 0 and cont_bar >= split_empty_bar:
                while len(splits[-1]) > 1 and splits[-1][-1] == "Bar(1)":
                    splits[-1].pop()
                if len(splits[-1]) > 1:
                    splits.append(["Bar(1)"])

        # remove redudant bars of the last split
        while len(splits[-1]) > 0 and splits[-1][-1] == "Bar(1)":
            splits[-1].pop()
        if len(splits[-1]) == 0:
            splits.pop()
        splits[0].insert(0, "BOS")
        splits[-1].append("EOS")

        for split in splits:
            songs.append(Song(info=song, content=list(map(lambda t: vocab[t], split))))

        pbar.update(1)
    pbar.close()

    return songs

def train(model, songs, epoch_num, batch_size, seg_size, cuda, config, vocab, save_path):
    model.train()
    model = model.cuda() if cuda else model

    """
    To reduce training time, we set the max sequence length to (mean + 2*standard_deviation)
    """
    song_lens = np.array(list(map(lambda s: len(s), songs)))
    mean = song_lens.mean()
    sd = song_lens.var() ** 0.5
    max_seq_len = song_lens.max()
    print("mean:", song_lens.mean())
    print("standard deviation:", song_lens.var()**0.5)
    print("max sequence length:", song_lens.max())
    max_seq_len = int(mean + 2*sd)
    match = song_lens[song_lens < max_seq_len]
    print("set max sequence length to:", f"{max_seq_len},",
          f"including {len(match)} songs of total ({round(len(match)/len(song_lens)*100, 2)}%)")

    # zero padding
    tokens = []
    for song in songs:
        tokens.append(song[:max_seq_len] + [0 for i in range(max_seq_len-len(song))])
    tokens = torch.tensor(tokens, dtype=int)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch_idx in range(epoch_num):
        pbar = tqdm(desc=f"epoch {epoch_idx+1}", total=len(tokens))
        running_loss = 0.0
        n_tokens = 0

        for batch_idx in range(0, len(tokens), batch_size):
            batch = tokens[batch_idx:batch_idx+batch_size]
            mems = None
            for seg_idx in range(0, max_seq_len, seg_size): # split a long sequence into small segments
                segs = batch[:, seg_idx:seg_idx+seg_size]
                labels = batch[:, seg_idx+1:seg_idx+seg_size+1]
                if cuda:
                    segs = segs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                output = model(input_ids=segs, mems=mems, labels=labels)
                output.losses.backward()
                optimizer.step()

                mems = output.mems
                running_loss += output.losses.item()
                n_tokens += len(segs[segs != 0])

            pbar.update(len(batch))
        pbar.close()
        #running_loss /= ((len(tokens) / batch_size) * (max_seq_len / seg_size))
        running_loss = running_loss / n_tokens
        print(" "*4, "losses sum:", running_loss, "\n")

        # save checkpoint
        ckpt = Checkpoint(
            epoch = epoch_idx,
            config = config,
            model_state_dict = model.state_dict(),
            optim_state_dict = optimizer.state_dict(),
            loss = running_loss,
            vocab = vocab,
        )

        if ckpt.loss < 1:
            torch.save(ckpt, save_path.replace("nnn", str(math.floor(ckpt.loss*10))))
        #torch.save(ckpt, save_path)

def test(model, song, cuda, seg_size, vocab):
    model.eval()
    model = model.cuda() if cuda else model

    tokens = torch.tensor(song, dtype=int)
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
        print(list(map(lambda i: vocab[i.item()], output_ids)))

    print("prompt without memory")
    for seg_idx in range(0, len(song), seg_size): # split a long sequence into small segments
        seg = song[seg_idx:seg_idx+seg_size]
        output_ids = [seg[0]]
        if cuda:
            seg = seg.cuda()
        for i in range(len(seg)):
            input_ids = torch.tensor(output_ids, dtype=int)[None, :]
            output = model(input_ids=input_ids, mems=None)
            output_ids.append(torch.argmax(output.pred_scores, dim=-1)[0][-1].item())
        print(list(map(lambda i: vocab[i], output_ids)))

    print("prompt with memory")
    mems = None
    for seg_idx in range(0, len(song), seg_size): # split a long sequence into small segments
        seg = song[seg_idx:seg_idx+seg_size]
        output_ids = [seg[0]]
        if cuda:
            seg = seg.cuda()
        for i in range(len(seg)):
            input_ids = torch.tensor(output_ids, dtype=int)[None, -1:]
            output = model(input_ids=input_ids, mems=mems)
            mems = output.mems
            output_ids.append(torch.argmax(output.pred_scores, dim=-1)[0][-1].item())
        print(list(map(lambda i: vocab[i], output_ids)))


def select_first_n_bar(songs, vocab, n_bar):
    clipped_songs = []
    for song in songs:
        bar_count = 0
        clip = Song(); clip.info_copy(song)
        for i, token in enumerate(song):
            if token == vocab["Bar(1)"]:
                if song[i+1] == vocab["Bar(1)"]:
                    continue
                if bar_count == n_bar:
                    break
                bar_count += 1
            clip.append(token)
        clipped_songs.append(clip)
    return clipped_songs

def generate(model, prompts, cuda, seg_size, vocab, max_gen_len):
    model.eval()
    model = model.cuda() if cuda else model

    result_ids = []
    pbar = tqdm(desc="Generating", total=len(prompts))
    for prompt in prompts:
        result = Song(); result.info_copy(prompt)
        prompt = torch.tensor(prompt, dtype=int)
        # limit segment length not longer than memory length
        seg_size = model.mem_len if model.mem_len < seg_size else seg_size

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
            segs = torch.tensor(gen_id, dtype=int).view(1,1)
            segs = segs.cuda() if cuda else segs

            output = model(input_ids=segs, mems=mems)
            mems = output.mems

            gen_id = torch.argmax(output.pred_scores, dim=-1)[0, -1].item()
            total_gen_num += 1
            #print(vocab[gen_id])
            result.append(gen_id)
            if vocab[gen_id] == "EOS":
                break
            if not total_gen_num < max_gen_len:
                break
        result_ids.append(result)
        pbar.update(1)
    pbar.close()

    return result_ids

def token_ids_to_midi(song, vocab):
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=song.bpm)
    inst = pretty_midi.Instrument(program=0)
    text = []

    global_beat = 0
    beat_time = 60 / song.bpm
    for token_id in song:
        if token_id == vocab["BOS"] or token_id == vocab["EOS"]:
            continue
        token = vocab[token_id]
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
