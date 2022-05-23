import os
import sys
import pickle
import numpy as np
import pretty_midi
import music
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils

def main():
    struct_dir = "/screamlab/home/tanch/structure-aware_infilling/experiment_result"
    det_dir = "/screamlab/home/tanch/double_encoder_transformer/experiment_result"
    vli_dir = "/screamlab/home/tanch/variable-length-piano-expansion/infilling_result/loss30"

    struct_h = []
    det_h = []
    vli_h = []
    gt_h = []

    struct_gs, det_gs, vli_gs, gt_gs = [], [], [], []
    struct_d, det_d, vli_d, gt_d = [], [], [], []

    for i in range(156):
        with open(os.path.join(struct_dir, f"{i}_past.pickle"), "rb") as f:
            past = pickle.load(f)
        with open(os.path.join(struct_dir, f"{i}_future.pickle"), "rb") as f:
            future = pickle.load(f)
        with open(os.path.join(struct_dir, f"{i}_middle.pickle"), "rb") as f:
            gt = pickle.load(f)


        with open(os.path.join(struct_dir, f"{i}_result.pickle"), "rb") as f:
            struct_result = pickle.load(f)

        h = H(get_pitch(struct_result), get_pitch(past)+get_pitch(future))
        gs = GS(get_onset(struct_result), get_onset(past)+get_onset(future))
        #print(get_pitch(middle)[:16])
        struct_h.append(h)
        struct_gs.append(gs)
        struct_d.append(utils.melody_simularity(get_pitch(struct_result), get_pitch(gt)))

        with open(os.path.join(det_dir, f"{i}_result.pickle"), "rb") as f:
            det_result = pickle.load(f)

        h = H(get_pitch(det_result), get_pitch(past)+get_pitch(future))
        gs = GS(get_onset(det_result), get_onset(past)+get_onset(future))
        #print(get_pitch(middle)[:16])
        det_h.append(h)
        det_gs.append(gs)
        det_d.append(utils.melody_simularity(get_pitch(det_result), get_pitch(gt)))

        vli_result = load_vli_result(os.path.join(vli_dir, f"{i}_middle.midi"))
        h = H(get_pitch(vli_result), get_pitch(past)+get_pitch(future))
        gs = GS(get_onset(vli_result), get_onset(past)+get_onset(future))
        vli_h.append(h)
        vli_gs.append(gs)
        vli_d.append(utils.melody_simularity(get_pitch(vli_result), get_pitch(gt)))

        h = H(get_pitch(gt), get_pitch(past)+get_pitch(future))
        gs = GS(get_onset(gt), get_onset(past)+get_onset(future))
        #print(get_pitch(middle)[:16])
        gt_h.append(h)
        gt_gs.append(gs)
        gt_d.append(utils.melody_simularity(get_pitch(gt), get_pitch(gt)))



    struct_h = np.array(struct_h)
    det_h = np.array(det_h)
    vli_h = np.array(vli_h)
    gt_h = np.array(gt_h)

    struct_gs = np.array(struct_gs)
    det_gs = np.array(det_gs)
    vli_gs = np.array(vli_gs)
    gt_gs = np.array(gt_gs)

    struct_d = np.array(struct_d)
    det_d = np.array(det_d)
    vli_d = np.array(vli_d)
    gt_d = np.array(gt_d)


    #print(gt_h.mean(), struct_h.mean(), det_h.mean(), vli_h.mean())
    #print(gt_gs.mean(), struct_gs.mean(), det_gs.mean(), vli_gs.mean())
    #print(gt_d.mean(), struct_d.mean(), det_d.mean(), vli_d.mean())
    statistics = []
    statistics.append([struct_h, struct_gs, struct_d])
    statistics.append([vli_h, vli_gs, vli_d])
    statistics.append([det_h, det_gs, det_d])
    statistics.append([gt_h, gt_gs, gt_d])
    for h, gs, d in statistics:
        print(*["%.2f$\\pm%.2f$" % (round(d.mean(),2), round(d.std(),2)) for d in (h, gs, d)], sep=" & ")

    df = []
    for v in struct_h:
        df.append(['Ours', 'H', v])
    for v in vli_h:
        df.append(['VLI', 'H', v])
    for v in det_h:
        df.append(["Hsu et. al's work", 'H', v])
    for v in gt_h:
        df.append(["Real", 'H', v])

    #df = pd.DataFrame([
    #    ['Ours', 'H', struct_h.mean()],
    #    ['Ours', 'GS', struct_gs.mean()],
    #    ['VLI', 'H', vli_h.mean()],
    #    ['VLI', 'GS', vli_gs.mean()],
    #    ["Hsu et. al's work", 'H', det_h.mean()],
    #    ["Hsu et. al's work", 'GS', det_gs.mean()],
    #    ['Real', 'H', gt_h.mean()],
    #    ['Real', 'GS', gt_gs.mean()],
    #], columns=['', '   ', ' '])
    plt.figure()
    df = pd.DataFrame(df, columns=['', '   ', ' '])
    ax = sns.barplot(x='', y=' ', hue='   ', data=df)
    ax.set_title('H')
    ax.get_figure().savefig("H.png")

    df = []
    for v in struct_gs:
        df.append(['Ours', 'GS', v])
    for v in vli_gs:
        df.append(['VLI', 'GS', v])
    for v in det_gs:
        df.append(["Hsu et. al's work", 'GS', v])
    for v in gt_gs:
        df.append(["Real", 'GS', v])

    plt.figure()
    df = pd.DataFrame(df, columns=['', '   ', ' '])
    ax = sns.barplot(x='', y=' ', hue='   ', data=df)
    ax.set_title('GS')
    ax.get_figure().savefig("GS.png")

    df = []
    for v in struct_d:
        df.append(['Ours', 'GS', v])
    for v in vli_d:
        df.append(['VLI', 'GS', v])
    for v in det_d:
        df.append(["Hsu et. al's work", 'GS', v])

    plt.figure()
    df = pd.DataFrame(df, columns=['', '   ', ' '])
    ax = sns.barplot(x='', y=' ', hue='   ', data=df)
    ax.set_title('D')
    ax.get_figure().savefig("D.png")

def get_pitch(song):
    p = []
    for bar in song.bars:
        for event in bar.events:
            for note in event.notes:
                p.append(note.pitch)
    return p

def get_onset(song):
    o = []
    for bar in song.bars:
        for event in bar.events:
            if len(event.notes) > 0:
                o.append(1)
            else:
                o.append(0)
    return o

def H(p1, p2):
    eps = 1e-4
    p1 = np.array(p1)
    p2 = np.array(p2)

    h1, h2 = np.zeros(12), np.zeros(12)

    for p in p1:
        h1[p % 12] += 1
    for p in p2:
        h2[p % 12] += 1

    #print(h1)

    h1 += eps
    h2 += eps

    h1 = h1/h1.sum()
    h2 = h2/h2.sum()

    h = -(h1*np.log2(h2)).sum()

    return h

def GS(g1, g2):
    g1 = np.array(g1)
    g2 = np.array(g2)
    gs = []

    for i in range(len(g1)//16):
        b1 = g1[i*16:(i+1)*16]
        for j in range(len(g2)//16):
            b2 = g2[j*16:(j+1)*16]
            gs.append(1 - np.logical_xor(b1, b2).mean())
    return np.array(gs).mean()

def load_vli_result(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    tempo = midi_data.get_tempo_changes()[1][1]
    time_offset = midi_data.get_tempo_changes()[0][1]
    T = (60/tempo) / 4

    song = music.Song("vli", 4, 4, int(tempo))
    for b in range(4):
        bar = music.Bar()
        for e in range(16):
            event = music.Event()
            bar.events.append(event)
        song.bars.append(bar)

    for note in midi_data.instruments[0].notes:
        onset = round((note.start-time_offset)/T)
        duration = round((note.end-note.start)/T)
        pitch = note.pitch
        n = music.Note(pitch, -1, onset, duration if duration > 0 else 1)

        if onset < 64:
            song.bars[onset//16].events[(onset%16)].notes.append(n)
    return song

def plot_in_vli_style():
    fs = 16
    box_width = 3.2
    plt.style.use('seaborn')
    fnames = ['ours', 'ilm', 'felix', 'real']
    md = {} # metrics dict
    for fname in fnames:
        with open('%s-metrics.pickle' % fname, 'rb') as handle:
            md[fname] = pickle.load(handle)
# `pos`: p:previous and a:after
# data_`metric`_`pos`: list or numpy array of metric values
    data_phe1_p = [md[fname]['phe1'][:, 0] for fname in fnames]
    data_phe1_a = [md[fname]['phe1'][:, 1] for fname in fnames]
    data_phe1_pa = [md['ours']['phe1'][:, 2] for fname in fnames]
    data_phe4_p = [md[fname]['phe4'][:, 0] for fname in fnames]
    data_phe4_a = [md[fname]['phe4'][:, 1] for fname in fnames]
    data_phe4_pa = [md['ours']['phe4'][:, 2] for fname in fnames]
    data_grv_p =  [md[fname]['grv'][:, 0] for fname in fnames]
    data_grv_a =  [md[fname]['grv'][:, 1] for fname in fnames]
    data_grv_pa =  [md['ours']['grv'][:, 2] for fname in fnames]
    ticks = ['Ours', 'ILM', 'FELIX', 'Real']
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    plt.figure()
    bp_phe1_p = plt.boxplot(data_phe1_p, positions=np.array(range(len(data_phe1_p)))*box_width-1.20, sym='', widths=0.25, patch_artist=True,)
    bp_phe1_a = plt.boxplot(data_phe1_a, positions=np.array(range(len(data_phe1_a)))*box_width-0.90, sym='', widths=0.25, patch_artist=True,)
    bp_phe1_pa = plt.boxplot(data_phe1_pa, positions=np.array(range(len(data_phe1_pa)))*box_width-0.60, sym='', widths=0.25, patch_artist=True,)
    bp_phe4_p = plt.boxplot(data_phe4_p, positions=np.array(range(len(data_phe4_p)))*box_width-0.30, sym='', widths=0.25, patch_artist=True,)
    bp_phe4_a = plt.boxplot(data_phe4_a, positions=np.array(range(len(data_phe4_a)))*box_width+0.00, sym='', widths=0.25, patch_artist=True,)
    bp_phe4_pa = plt.boxplot(data_phe4_pa, positions=np.array(range(len(data_phe4_pa)))*box_width+0.30, sym='', widths=0.25, patch_artist=True,)
    bp_grv_p = plt.boxplot(data_grv_p, positions=np.array(range(len(data_grv_p)))*box_width+0.60, sym='', widths=0.25, patch_artist=True,)
    bp_grv_a = plt.boxplot(data_grv_a, positions=np.array(range(len(data_grv_a)))*box_width+0.90, sym='', widths=0.25, patch_artist=True,)
    bp_grv_pa = plt.boxplot(data_grv_pa, positions=np.array(range(len(data_grv_pa)))*box_width+1.20, sym='', widths=0.25, patch_artist=True,)
# draw temporary red and blue lines and use them to create a legend
    circ1 = mpatches.Patch(facecolor='pink', label='H1 (past)')
    circ2 = mpatches.Patch(facecolor='pink', hatch='//',label='H1 (future)')
    circ3 = mpatches.Patch(facecolor='pink', hatch='oo',label='H1 (past-future)')
    circ4 = mpatches.Patch(facecolor='blue', label='H4 (past)')
    circ5 = mpatches.Patch(facecolor='blue', hatch='//',label='H4 (future)')
    circ6 = mpatches.Patch(facecolor='blue', hatch='oo',label='H4 (past-future)')
    circ7 = mpatches.Patch(facecolor='lightgreen', label='GS (past)')
    circ8 = mpatches.Patch(facecolor='lightgreen', hatch='//',label='GS (future)')
    circ9 = mpatches.Patch(facecolor='lightgreen', hatch='oo',label='GS (past-future)')
    plt.legend(handles = [circ1,circ2,circ3, circ4, circ5, circ6, circ7, circ8, circ9], fontsize=fs)
    plt.ylabel('Metric Difference', fontsize=fs)
    plt.xticks(np.arange(0, len(ticks) * box_width, box_width), ticks, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlim(-2, len(ticks)* box_width)
# plt.ylim(0, 1)
    plt.tight_layout()
# colors = ['pink', 'lightblue', 'lightgreen']
    colors = ['pink', 'blue', 'lightgreen']
    for i, bplot in enumerate([bp_phe1_p, bp_phe1_a, bp_phe1_pa, bp_phe4_p, bp_phe4_a, bp_phe4_pa, bp_grv_p, bp_grv_a, bp_grv_pa]):
        for patch in bplot['boxes']:
            if i % 3 == 1:
                patch.set(hatch='//')
            elif i % 3 == 2:
                patch.set(hatch='oo')
            patch.set_facecolor(colors[i // 3])
    plt.show()

if __name__ == "__main__":
    main()
