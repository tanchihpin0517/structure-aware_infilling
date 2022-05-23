import os
import sys
import pickle
import music

def main(data_file, save_file):
    songs = []
    sel_track = ["melody", "bridge"]
    with open(data_file, "rb") as f:
        songs = pickle.load(f)

    for song in songs:
        for bar in song.bars:
            for event in bar.events:
                tmp = []
                for note in event.notes:
                    if note.track.lower() in sel_track:
                        tmp.append(note)
                event.notes = tmp

    testing_data = []
    count = 0
    for song in songs:
        structs, struct_map = extract_struct(song)
        song.save(f"./dataset/testing_data_origin/{count}_origin.midi")

        appear = set([None])
        for sid, (struct, s_start, s_end) in enumerate(song.struct_indices):
            if struct == None:
                continue
            # skip first appearing
            if struct not in appear:
                appear.add(struct)
                continue
            # find struct with length <= 4
            if s_end-s_start > 4:
                continue

            c1 = song.struct_indices[sid-1][0] == song.struct_indices[sid][0] and \
                 song.struct_indices[sid][0] != song.struct_indices[sid+1][0]
            c2 = song.struct_indices[sid-1][0] != song.struct_indices[sid][0] and \
                 song.struct_indices[sid][0] == song.struct_indices[sid+1][0]

            if not (c1 or c2):
                continue

            data = {}
            for bar in song.bars[s_start:s_end]:
                assert bar.struct_id == struct_map[struct]

            context_len = 6
            data['past'] = song.bars[max(0,s_start-context_len):s_start]
            data['target'] = song.bars[s_start:s_end]
            data['future'] = song.bars[s_end:s_end+context_len]
            data['struct'] = structs[struct_map[struct]]
            data['original_song'] = song

            testing_data.append(data)

            seg = music.Song.copy(song, with_content=False)
            seg.bars = data['past'] + data['target'] + data['future']
            seg.save(f"./dataset/testing_data_origin/{count}_seg.midi", time_strip=True)

            count += 1

    print(len(testing_data))
    with open(save_file, 'wb') as f:
        pickle.dump(testing_data, f)


def extract_struct(song):
    appear = set([None])
    structs = []
    struct_map = {}
    struct_count = 0
    for struct, s_start, s_end in song.struct_indices:
        if struct not in appear:
            appear.add(struct)
            structs.append(song.bars[s_start:s_end])
            struct_map[struct] = struct_count
            struct_count += 1
    return structs, struct_map


if __name__ == '__main__':
    main("./dataset/pop909.pickle.testing", "./dataset/testing_data.pickle")
