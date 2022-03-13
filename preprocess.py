import os
import pretty_midi
import multiprocessing as mp
from tqdm import tqdm
from music import Event, Song, Bar, POP909Note
import re

def pop909(dir_origin, dir_struct, beat_division=4, N=None, song_sel=None, multi_task=True, verbose=False):
    songs = []
    map_args = [(dir_origin, dir_struct, sub_dir, beat_division, verbose) for sub_dir in sorted(os.listdir(dir_origin))]
    if N is not None:
        map_args = map_args[:N]
    if song_sel is not None:
        map_args = map_args[song_sel-1:song_sel]

    err_cnt = 0
    with mp.Pool() as pool:
        pbar = tqdm(desc="pop909", total=len(map_args))
        if multi_task:
            for song in pool.imap(pop_909_map_func, map_args):
                if song is not None:
                    songs.append(song)
                else:
                    err_cnt += 1
                pbar.update(1)
        else:
            for song in map(pop_909_map_func, map_args):
                if song is not None:
                    songs.append(song)
                else:
                    err_cnt += 1
                pbar.update(1)
        pbar.close()
    #for song in map(pop_909_map_func, map_args[:1]):
    #    if song:
    #        songs.append(song)
    return songs, err_cnt

def pop_909_map_func(args):
    dir_origin, dir_struct, sub_dir, beat_division, verbose = args
    percision = 8
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
    tempo_file = os.path.join(song_dir, "tempo.txt")

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

    # get tempo information
    with open(tempo_file) as f:
        bpm = int(f.readline())

    # initialize song
    song = Song(name=sub_dir, beat_division=beat_division, beat_per_bar=beat_per_bar, bpm=bpm)
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    melody, bridge, piano = midi_data.instruments

    # Use beat_midi.txt to create beat events and set tempo info
    events = []
    tempo_change_times, tempi = midi_data.get_tempo_changes()
    tempo_idx = 0
    with open(beat_midi_file) as f:
        prev_interval = 0
        for idx, line in enumerate(map(lambda l: l.strip(), f)):
            time, beat1, beat2 = map(float, line.split(" "))
            time = round(time, percision)
            for i in range(tempo_idx, len(tempo_change_times)-1):
                if time < tempo_change_times[i+1]:
                    break
                else:
                    tempo_idx = i+1
            if idx > 0:
                prev_interval = time - events[-1].start
                events[-1].end = time
            events.append(Event(start=time, tempo=round(tempi[tempo_idx])))
        events[-1].end = events[-1].start + prev_interval

    """
    Because of the percision problem of float, we check the continuation of events here.
    """
    for i in range(len(events)-1):
        assert events[i].end == events[i+1].start, f"{events[i], events[i+1]}"

    # chord information
    """
    I notice that the chord beat in "finalized_chord.txt" is not correct,
    so use "chord_midi.txt" as the chord reference.
    """
    with open(chord_file) as f:
        cur_idx = 0
        for line in map(lambda l: l.strip(), f.readlines()):
            onset, offset, name = line.split("\t")
            onset, offset = float(onset), float(offset)

            # skip no chord
            if name == "N":
                continue

            if onset < events[0].start: # deal with chord before events[0]
                events[0].chord = name
                continue

            while onset not in events[cur_idx] and cur_idx < len(events)-1:
                cur_idx += 1

            if cur_idx == len(events)-1:
                events[cur_idx].chord = name
            else:
                # only set the chord "anchor" here
                if onset - events[cur_idx].start <= events[cur_idx].end - onset:
                    events[cur_idx].chord = name
                else:
                    events[cur_idx+1].chord = name

        # extend and fill gaps between anchors of chord info
        prev_chord = None
        for event in events:
            if event.chord is None and prev_chord is not None:
                event.chord = prev_chord
            prev_chord = event.chord


    """
    The accent information in beat file is wrong.
    Instead we use the beat of the first chord occurrence to find the possible position of bars,
    and use the first note of piano to decide where is the first bar.
    """
    # find the first beat of piano notes
    first_note_onset = sorted(piano.notes, key=lambda note: note.start)[0].start
    first_note_beat = 0
    dist = 1e9
    for idx, event in enumerate(events):
        if (event.start - first_note_onset)**2 < dist:
            first_note_beat = idx
            dist = (event.start - first_note_onset)**2

    # get the first bar beat
    first_bar_beat = -1
    for i in range(first_note_beat, len(events)):
        if events[i].chord is not None:
            first_bar_beat = i
            break

    # insert bars into song
    assert first_bar_beat != -1
    start_bar_idx = 0
    # we add bar backward because first_bar_beat is found by examing piano notes, not all notes
    for bar, i in enumerate(range(first_bar_beat, 0, -beat_per_bar)):
        song.bars.insert(0, Bar())
        for j in range(beat_per_bar):
            if i-beat_per_bar+j >= 0: # prevent add wrong event because python's list indexing
                song.bars[0].events.append(events[i-beat_per_bar+j])
        start_bar_idx += 1
    for bar, i in enumerate(range(first_bar_beat, len(events), beat_per_bar)):
        song.bars.append(Bar())
        for j in range(beat_per_bar):
            if i+j < len(events):
                song.bars[-1].events.append(events[i+j])

    # structural information
    with open(struct_1_file) as f:
        cur_idx = start_bar_idx
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
                song.bars[cur_idx].struct = labels[i] # like chord, only set "anchor" here
                cur_idx += int(nums[i])
            except IndexError as e:
                if verbose:
                    print(f"error: bar number doesn't match with structure info in {sub_dir}, return None.")
                return None # skip if the number of bars don't match

    # extend and fill gaps between anchors of structure info
    prev_struct = None
    for bar in song.bars:
        if bar.struct is None and prev_struct is not None:
            bar.struct = prev_struct
        prev_struct = bar.struct

    # set bar's (start, end) and expand events to the level of 16th(depands on beat_division) notes
    for bar in song.bars:
        #bar.start = bar.events[0].start
        #bar.end = bar.events[-1].end

        tmp = []
        for i in range(len(bar.events)):
            onset = bar.events[i].start
            offset = bar.events[i].end
            for j in range(0, beat_division):
                start = round(onset + (offset-onset)*j/beat_division, percision)
                end = round(start + (offset-onset)/beat_division, percision)
                tmp.append(Event(start=start, end=end, tempo=bar.events[i].tempo, chord=bar.events[i].chord))
        bar.events.clear()
        bar.events.extend(tmp)

    # Make sure the continuation of events
    events = song.flatten_events()
    for i in range(len(events)-1):
        time = round((events[i].end + events[i+1].start) / 2, percision)
        events[i].end = time
        events[i+1].start = time

    for bar in song.bars:
        bar.start = bar.events[0].start
        bar.end = bar.events[-1].end

    for i in range(len(events)-1):
        assert events[i].end == events[i+1].start, f"{events[i], events[i+1]}"



    # add notes
    notes = []
    for t, track in [(melody, "melody"), (bridge, "bridge"), (piano, "piano")]:
    #for t, track in [(melody, "melody")]:
        for n in t.notes:
            notes.append(POP909Note(track=track, midi=n))
    #notes = [Note(y, track=track) for x, track in [(melody, "melody")] for y in x.notes]
    notes = sorted(notes, key=lambda note: note.midi.start)


    note_idx = 0
    events = song.flatten_events()

    # skip notes before the first event
    while True:
        onset = notes[note_idx].midi.start
        if onset < events[0].start:
            note_idx += 1
        else:
            break

    for i in range(len(events)):
        while note_idx < len(notes):
            onset = notes[note_idx].midi.start
            if onset in events[i]:
                if onset - events[i].start <= events[i].end - onset:
                    events[i].notes.append(notes[note_idx])
                elif i == len(events)-1: # last event
                    events[i].notes.append(notes[note_idx])
                else:
                    events[i+1].notes.append(notes[note_idx])
                note_idx += 1
            else:
                break

    # update note onset
    for bar in song.bars:
        for i_e, event in enumerate(bar.events):
            for note in event.notes:
                note.onset = i_e

    # update note duration
    for i in range(len(events)):
        for note in events[i].notes:
            offset = note.midi.end
            """
            because we choose the event[i] with smallest distance as the container of notes,
            some offset of notes may start from event[i-1] not event[i]
            """
            for j in range(i if i == 0 else i-1, len(events)):
                if offset in events[j]:
                    if offset - events[j].start > events[j].end - offset: # include this moment
                        note.duration = j-i+1
                    else:
                        note.duration = j-i
                    if note.duration == 0:
                        note.duration = 1
                    break # leave after setting duration
            assert note.duration > 0 # check again

    #print(song) # debug
    return song

def _test_pop909(testdir: str, song_sel: int = None, track_sel=['melody', 'bridge', 'piano'], N: int = None):
    songs, err_cnt = pop909("./dataset/pop909/POP909", "./dataset/pop909_struct/POP909", song_sel=song_sel, N=N)
    print("error number:", err_cnt)

    for song in songs:
        for event in song.flatten_events():
            tmp = []
            for note in event.notes:
                if note.track in track_sel:
                    tmp.append(note)
            event.notes = tmp

    for _, song in enumerate(songs):
        with open(f"{testdir}/{song.name}.txt", "w") as f:
            f.write(f"{song}\n")

if __name__ == '__main__':
    song_sel = 14
    #song_sel = None
    #track_sel = ['melody', 'bridge', 'piano']
    track_sel = ['melody', 'bridge', 'piano']
    _test_pop909("/screamlab/home/tanch/structural_expansion/pop909_test", song_sel=song_sel, track_sel=track_sel)
