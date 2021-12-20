import os
import pretty_midi
import multiprocessing as mp
from tqdm import tqdm
from misc import Event, Song, Note
import re
import copy

def pop909(dir_origin, dir_struct):
    songs = []

    beat_division = 4
    map_args = [(dir_origin, dir_struct, sub_dir, beat_division) for sub_dir in sorted(os.listdir(dir_origin))]
    with mp.Pool() as pool:
        pbar = tqdm(desc="pop909", total=len(map_args))
        for song in pool.imap(pop_909_map_func, map_args):
            if song is not None:
                songs.append(song)
            pbar.update(1)
        pbar.close()
    #for song in map(pop_909_map_func, map_args[:1]):
    #    if song:
    #        songs.append(song)
    return songs

def pop_909_map_func(args):
    dir_origin, dir_struct, sub_dir, beat_division = args
    #print(sub_dir)

    # get original song data
    song_dir = os.path.join(dir_origin, sub_dir)
    if not os.path.isdir(song_dir): # skip other files or directories
        return None
    midi_file = os.path.join(song_dir, f"{sub_dir}.mid")
    beat_audio_file = os.path.join(song_dir, "beat_audio.txt")
    beat_midi_file = os.path.join(song_dir, "beat_midi.txt")
    chord_file = os.path.join(song_dir, "chord_midi.txt")

    # get structural data
    song_dir = os.path.join(dir_struct, sub_dir)
    if not os.path.isdir(song_dir): # skip other files or directories
        return None
    #chord_file = os.path.join(song_dir, "finalized_chord.txt")
    struct_1_file = os.path.join(song_dir, "human_label1.txt")
    struct_2_file = os.path.join(song_dir, "human_label2.txt")

    # initialize song
    song = Song(name=sub_dir, beat_division=beat_division)
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    melody, bridge, piano = midi_data.instruments

    """
    From observation, it seems that neither "beat_audio.txt" of "beat_midi.txt"
    record the correct beat and bar information.
    Thus here we get the first bar by refering the chord file and piano part.
    """
    beat_per_bar = 0
    with open(beat_audio_file) as f:
        for line in [l.strip() for l in f]:
            time, beat = map(float, line.split("\t"))
            if beat > beat_per_bar:
                beat_per_bar = int(beat)
            else:
                break

    # Use beat_midi.txt to create beat events
    with open(beat_midi_file) as f:
        for idx, line in enumerate(map(lambda l: l.strip(), f)):
            time, beat1, beat2 = map(float, line.split(" "))
            song.append(Event(time=time))

    # chord information
    """
    I notice that the chord beat in "finalized_chord.txt" is not correct,
    so use "chord_midi.txt" as the chord reference.
    """
    cur_idx = 0
    with open(chord_file) as f:
        for line in map(lambda l: l.strip(), f.readlines()):
            onset, offset, name = line.split("\t")
            onset, offset = float(onset), float(offset)
            if name == "N":
                continue
            for i in range(cur_idx, len(song)-1):
                if onset >= song[cur_idx+1].time:
                    cur_idx += 1
            if cur_idx == len(song)-1:
                song[cur_idx].chord = name
            else:
                l, r = song[cur_idx].time, song[cur_idx+1].time
                if onset - l < r - onset:
                    song[cur_idx].chord = name
                else:
                    song[cur_idx+1].chord = name

    """
    The accent information in beat file is wrong.
    Instead we use the beat of the first chord occurrence to find the possible position of bars,
    and use the first note of piano to decide where is the first bar.
    """
    # find the first beat of piano notes
    first_note_onset = sorted(piano.notes, key=lambda note: note.start)[0].start
    first_note_beat = 0
    dist = 1e9
    for idx, event in enumerate(song):
        if (event.time - first_note_onset)**2 < dist:
            first_note_beat = idx
            dist = (event.time - first_note_onset)**2

    # get the first bar beat
    first_bar_beat = None
    for i in range(first_note_beat, len(song)):
        if song[i].chord:
            first_bar_beat = i
            break

    # set bars
    song.beat_per_bar = beat_per_bar
    for bar, i in enumerate(range(first_bar_beat, len(song), beat_per_bar)):
        song[i].bar = bar
    for bar, i in enumerate(range(first_bar_beat, 0, -beat_per_bar)):
        song[i].bar = -bar

    # structural information
    with open(struct_1_file) as f:
        cur_idx = first_bar_beat
        line = f.readline().strip()
        labels = re.split("[0-9]+", line)
        nums = re.split("[a-zA-Z]+", line)
        if "" in labels:
            labels.remove("")
        if "" in nums:
            nums.remove("")
        assert len(labels) == len(nums), "Invalid structure: %s" % line

        for i in range(len(labels)):
            try:
                song[cur_idx].struct = labels[i]
                cur_idx += int(nums[i]) * song.beat_per_bar
            except IndexError as e:
                #print(f"error: {sub_dir}")
                return None # skip if the number of bars don't match

    # expand events to the level of 16th(depands on beat_division) notes
    tmp = []
    for i in range(len(song)-1):
        onset = song[i].time
        offset = song[i+1].time
        tmp.append(song[i])
        for j in range(1, beat_division):
            tmp.append(Event(time=onset+(offset-onset)*j/beat_division))
    tmp.append(song[-1])
    song.clear()
    song.extend(tmp)

    # add notes
    notes = [Note(y, track=track) for x, track in [(melody, "melody"), (bridge, "bridge"), (piano, "piano")] for y in x.notes]
    #notes = [Note(y, track=track) for x, track in [(melody, "melody")] for y in x.notes]
    cur_idx = 0
    for note in sorted(notes, key=lambda note: note.midi.start):
        onset = note.midi.start
        for i in range(cur_idx, len(song)-1):
            if onset >= song[cur_idx+1].time:
                cur_idx += 1
        if cur_idx == len(song)-1:
            note.onset = cur_idx
        else:
            l, r = song[cur_idx].time, song[cur_idx+1].time
            if onset - l < r - onset:
                note.onset = cur_idx
            else:
                note.onset = cur_idx + 1

    cur_idx = 0
    for note in sorted(notes, key=lambda note: note.midi.end):
        offset = note.midi.end
        for i in range(cur_idx, len(song)-1):
            if offset > song[cur_idx].time:
                cur_idx += 1
        note.duration = cur_idx - note.onset
        if note.duration == 0:
            note.duration += 1

    for note in notes:
        song[note.onset].notes.append(note)

    return song

def _test_pop909():
    songs = pop909("./dataset/pop909/POP909", "./dataset/pop909_struct/POP909")

    for i, song in enumerate(songs):
        with open(f"test/{song.name}.txt", "w") as f:
            for event in song:
                f.write(f"{event}\n")

if __name__ == '__main__':
    _test_pop909()
