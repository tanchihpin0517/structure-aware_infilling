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
    for song in songs:
        count = 0
        prev_struct = song.bars[0].struct
        for bar in song.bars:
            if bar.struct == prev_struct:
                count += 1
            else:
                #print(prev_struct, count)
                prev_struct = bar.struct
                struct_lens.append(count)
                if count > 16:
                    print(song.name)
                    error += 1
                count = 0
    struct_lens = np.array(struct_lens)
    print(f"average length of structures in each song: {struct_lens.mean()}")
    print(struct_lens.mean(), struct_lens.max())
    print(error)


if __name__ == "__main__":
    data_file = "/screamlab/home/tanch/structure-aware_infilling/dataset/pop909.pickle"
    analyze(data_file=data_file)
