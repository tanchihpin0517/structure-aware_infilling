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


if __name__ == "__main__":
    data_file = "/screamlab/home/tanch/structural_expansion/dataset/pop909.pickle"
    analyze(data_file=data_file)
