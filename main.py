import os
import argparse
import preprocess
import pickle
from misc import Vocabulary, Checkpoint
from tqdm import tqdm
from model import Config, Transformer
import torch
import torch.nn.functional as F
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--vocab-file', type=str, default='dataset/vocab.txt')
    parser.add_argument('--data-file', type=str, default='dataset/pop909.pickle')
    parser.add_argument('--preprocess', default=False, action='store_true')
    parser.add_argument('--ckpt', type=str, default="trained_model/loss_nn.ckpt")

    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--seg-size', type=int, default=256)
    parser.add_argument('--epoch-num', type=int, default=2000, help='number of training epochs')

    parser.add_argument('--target-max-percent', type=float, default=0.25, help="Up to `seq_len *                 target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--n-step-bars', type=int, default=8, help='how many bars to step before next training   data fetching (the smaller the more training data)')
    parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--init-lr', type=float, default=1e-4, help='initial learning rate')

# for prediction phase
    parser.add_argument('--test-data-file', type=str, default='worded_data.pickle')
    parser.add_argument('--song-idx', type=int, default=170)
    return parser.parse_args()

def main():
    args = parse_args()

    #data = load_data(args.data_file, args.preprocess, melody_only=True, max_song_num=16)
    data = load_data(args.data_file, args.preprocess, melody_only=True)
    vocab = load_vocab(args.vocab_file)
    songs = tokenize(data, vocab)

    config = Config(
        vocab.size(),
        d_embed = 512,
        d_model = 512,
        n_head = 8,
        d_inner = 2048,
        n_layer = 8,
        mem_len = args.seg_size, #1600,
    )
    model = Transformer(config)

    if args.train:
        train(model, songs, args.epoch_num, args.batch_size, args.seg_size, args.cuda,
              config, vocab, args.ckpt)

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

def tokenize(data, vocab):
    songs = []
    pbar = tqdm(desc="Tokenize", total=len(data))
    for song in data:
        tokens = []
        tokens.append("BOS")
        bar_idx = 0
        for i, event in enumerate(song):
            if event.bar is not None:
                if event.bar >= 0:
                    tokens.append("Bar(1)")
                bar_idx = i
            if len(event.notes) > 0:
                tokens.append(f"Position({i-bar_idx})")
            for note in event.notes:
                if note.duration > 32: # clip if note longer than 8 beats
                    note.duration = 32
                tokens.extend([f"Pitch({note.pitch})", f"Duration({note.duration})"])
        tokens.append("EOS")
        songs.append(list(map(lambda t: vocab[t], tokens)))
        pbar.update(1)
    pbar.close()

    return songs

def train(model, songs, epoch_num, batch_size, seg_size, cuda, config, vocab, ckpt_path):
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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch_idx in range(epoch_num):
        pbar = tqdm(desc=f"epoch {epoch_idx+1}", total=len(tokens))
        running_loss = 0.0

        for batch_idx in range(0, len(tokens), batch_size):
            batch = tokens[batch_idx:batch_idx+batch_size]
            mems = None
            for seg_idx in range(0, max_seq_len, seg_size): # split a long sequence into small segments
                sen = batch[:, seg_idx:seg_idx+seg_size]
                sen = sen.cuda() if cuda else sen

                optimizer.zero_grad()
                output = model(input_ids=sen, mems=mems)
                loss = criterion(output.word_dist.transpose(1,2), sen)
                loss.backward()
                optimizer.step()

                mems = output.mems
                running_loss += loss.item()
            pbar.update(len(batch))
        pbar.close()
        running_loss /= ((len(tokens) / batch_size) * (max_seq_len / seg_size))
        print(" "*4, "loss:", running_loss, "\n")

        # save checkpoint
        ckpt = Checkpoint(
            epoch = epoch_idx,
            config = config,
            model_state_dict = model.state_dict(),
            optim_state_dict = optimizer.state_dict(),
            loss = running_loss,
            vocab = vocab,
        )
        if ckpt.loss >= 3:
            torch.save(ckpt, ckpt_path.replace("nn", str(round(ckpt.loss)*10)))
        else:
            torch.save(ckpt, ckpt_path.replace("nn", str(round(ckpt.loss*10))))

if __name__ == "__main__":
    main()
