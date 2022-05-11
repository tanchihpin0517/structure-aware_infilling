import pickle
import numpy as np
import os
import sys
import metrics
import pandas as pd
import seaborn as sns
from .model.tokenizer import Tokenizer

def analyze(data_file):
    print("file:", data_file)
    with open(data_file, "rb") as f:
        songs = pickle.load(f)

    empty_bars = []
    for song in songs:
        count = 0
        for bar in reversed(song.bars):
            if bar.empty():
                count += 1
            else:
                break
        empty_bars.append(count)
    empty_bars = np.array(empty_bars)
    print(f"average empty bars in the end: {empty_bars.mean()}")

    notes_num = []
    for song in songs:
        count = 0
        for event in song.flatten_events():
            count += len(event.notes)
        notes_num.append(count)
    notes_num = np.array(notes_num)
    print(f"average notes of each song: {notes_num.mean()}")

    struct_lens = []
    error = 0
    struct_dist = {}
    struct_types = set()
    for song in songs:
        too_long = False
        for label, start, end in song.struct_indices:
            l = end - start
            struct_lens.append(l)
            struct_dist[l] = struct_dist[l] + 1 if l in struct_dist else 1
            if l > 32:
                too_long = True

            if label is not None:
                struct_types.add(label)
        if too_long:
            error += 1

    struct_lens = np.array(struct_lens)
    print(f"average length of structures in each song: {struct_lens.mean()}")
    print(f"longest struct: {struct_lens.max()}")
    print(f"number and ratio of songs which contain at least one struct longer than 32:", error, error/len(songs))
    print(f"number of struct types:", len(struct_types))

    bar_len = []
    for song in songs:
        bar_len.append(len(song.bars))
    bar_len = np.array(bar_len)
    print(f"average bars of each song: {bar_len.mean()}")
    print(f"mean+2*std bars of each song: {bar_len.mean() + 2*bar_len.std()}")

def draw_sim(file_dir):
    files = sorted(os.listdir(file_dir))
    results = []
    middles = []
    pasts = []
    futures = []
    for file in files:
        if "result" in file:
            with open(os.path.join(file_dir, file), 'rb') as f:
                results.append(pickle.load(f))
        if "middle" in file:
            with open(os.path.join(file_dir, file), 'rb') as f:
                middles.append(pickle.load(f))
        if "past" in file:
            with open(os.path.join(file_dir, file), 'rb') as f:
                pasts.append(pickle.load(f))
        if "future" in file:
            with open(os.path.join(file_dir, file), 'rb') as f:
                futures.append(pickle.load(f))

    target_scores = []
    baseline_scores = []
    for i in range(len(results)):
        target_scores.append(metrics.melody_sim_DP(skyline_pitch(middles[i]), skyline_pitch(results[i])))
        baseline_scores.append(metrics.melody_sim_DP(skyline_pitch(middles[i]), skyline_pitch(pasts[i])))
    #target_scores = np.array(target_scores)
    #baseline_scores = np.array(baseline_scores)
    #print(target_scores.mean(), target_scores.std())
    #print(baseline_scores.mean(), baseline_scores.std())

    data = []
    for score in target_scores:
        data.append(['Target', score])
    for score in baseline_scores:
        data.append(["Baseline", score])
    df = pd.DataFrame(data, columns=['Model', 'Distance'])
    print(df)
    ax = sns.barplot(x='Model', y='Distance', data=df, capsize=.2)
    ax.set_title("Melody(skyline) simularity distance (the lower the better)")
    ax.get_figure().savefig('sim.png')

def skyline_pitch(song):
    skyline = []
    for event in song.flatten_events():
        pitches = []
        for note in event.notes:
            pitches.append(note.pitch)
        if len(pitches) > 0:
            skyline.append(sorted(pitches)[-1])
    return skyline

if __name__ == "__main__":
    data_file = "/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle"
    analyze(data_file=data_file)

    tokenizer = Tokenizer()
    #draw_sim("gen_midi_transxl_struct_infilling_enc/validation_loss_1")

