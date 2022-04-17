import pickle
import numpy as np

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

if __name__ == "__main__":
    data_file = "/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle"
    analyze(data_file=data_file)
