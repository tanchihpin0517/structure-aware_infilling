import os
from typing import List
from music import Song
from collections.abc import Iterable
import copy
import torch
import numpy as np
from collections import OrderedDict

class Tokenizer:

    PAD = "PAD"
    MASK = "MASK"
    BOS = "BOS"
    EOS = "EOS"
    BAR = "Bar"
    TEMPO = "Tempo"
    POSITION = "Position"
    PITCH = "Pitch"
    VELOCITY = "Velocity"
    DURATION = "Duration"
    STRUCT = "Struct"
    NONE = "None"

    def bar(self, key):
        return f"{self.BAR}({key})"

    def tempo(self, key):
        return f"{self.TEMPO}({key})"

    def pos(self, key):
        return f"{self.POSITION}({key})"

    def pitch(self, key):
        return f"{self.PITCH}({key})"

    def vel(self, key):
        return f"{self.VELOCITY}({key})"

    def dur(self, key):
        return f"{self.DURATION}({key})"

    def struct(self, key):
        return f"{self.STRUCT}({key})"

    def __init__(self, vocab_file=None, ignore_idx=-100, use_cp=True, verbose=True):

        self.token_to_id_tabel = {}
        self.id_to_token_tabel = []
        self.use_cp = use_cp
        self.class_tabel = OrderedDict()
        self.ignore_idx = -100

        self.max_dur = 16
        self.tempo_base = 28
        self.tempo_tick_num = 47
        self.vel_base = 0
        self.vel_tick_num=33

        #if vocab_file is None:
        #    self._init_vocab()
        #else:
        #    if os.path.exists(vocab_file):
        #        if verbose:
        #            print("Vocabulary file already exists: " + vocab_file)
        #        with open(vocab_file, "r") as f:
        #            for i, token in enumerate(map(lambda t: t.strip(), f)):
        #                self.token_to_id_tabel[token] = i
        #                self.id_to_token_tabel.append(token)
        #    else:
        #        if verbose:
        #            print("Vocabulary file doesn't exist. Create a new one: " + vocab_file)
        #        self._init_vocab()
        #        self.save_vocab(vocab_file)
        self._init_vocab()
        self.save_vocab(vocab_file)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.token_to_id_tabel[key]
        else:
            try:
                idx = int(key)
                return self.id_to_token_tabel[idx]
            except ValueError:
                raise TypeError(f"key must be str or indice, not: {type(key)}")

    def save_vocab(self, vocab_file):
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.id_to_token_tabel))

    def vocab_size(self):
        return len(self.id_to_token_tabel)

    def _init_vocab(self):
        """
        CP:
            begin: (bar(bos), temp(bos), pos(bos), pitch(bos), vel(bos), dur(bos), struct(bos))
                    ---> only use bar(key) to check bos and eos
            ...
            k: (bar(0), temp, pos, pitch, vel, dur, struct)
            k+1: (bar(0), temp, pos, pitch, vel, dur, struct)
            ...
            k+l: (bar(1), temp(bar), pos(bar), pitch(bar), vel(bar), dur(bar), struct(N))
                 ---> check bar and struct (if exist) first, and if find bar(1), ignore remaining attributes
            ...
            end: (bar(eos), temp(eos), pos(eos), pitch(eos), vel(eos), dur(eos), struct(eos))
        REMI:
            begin: bos
            ...
            k-2: struct
            k-1: bar(1)
            k: tempo
            k+1: pos
            k+2: pitch
            k+3: vel
            k+4: dur
            ...
            end: eos
        """
        tokens = [
            self.PAD,
            self.MASK,
            self.BOS,
            self.EOS,
        ]
        tokens = tokens + ["RESERVED"] * (10 - len(tokens))

        bar, tempo, pos, pitch, vel, dur, struct = [[] for _ in range(7)]
        offset = len(tokens)

        bar.extend([self.bar(t) for t in [self.BOS, self.EOS]])
        bar.extend([self.bar(b) for b in range(2)])
        self.class_tabel[self.BAR] = (offset, offset+len(bar))
        tokens.extend(bar)
        offset = len(tokens)

        tempo.extend([self.tempo(self.BAR)])
        #tempo.extend([self.tempo(t) for t in [self.BAR, self.BOS, self.EOS]])
        tempo.extend([self.tempo(t) for t in range(28, 28+47*4, 4)])
        self.class_tabel[self.TEMPO] = (offset, offset+len(tempo))
        tokens.extend(tempo)
        offset = len(tokens)

        pos.extend([self.pos(self.BAR)])
        #pos.extend([self.pos(p) for p in [self.BAR, self.BOS, self.EOS]])
        pos.extend([self.pos(p) for p in range(16)])
        self.class_tabel[self.POSITION] = (offset, offset+len(pos))
        tokens.extend(pos)
        offset = len(tokens)

        pitch.extend([self.pitch(self.BAR)])
        #pitch.extend([self.pitch(p) for p in [self.BAR, self.BOS, self.EOS]])
        pitch.extend([self.pitch(p) for p in range(22, 22+86)])
        self.class_tabel[self.PITCH] = (offset, offset+len(pitch))
        tokens.extend(pitch)
        offset = len(tokens)

        vel.extend([self.vel(self.BAR)])
        #vel.extend([self.vel(v) for v in [self.BAR, self.BOS, self.EOS]])
        vel.extend([self.vel(v) for v in range(0, 33*4, 4)])
        self.class_tabel[self.VELOCITY] = (offset, offset+len(vel))
        tokens.extend(vel)
        offset = len(tokens)

        dur.extend([self.dur(self.BAR)])
        #dur.extend([self.dur(d) for d in [self.BAR, self.BOS, self.EOS]])
        dur.extend([self.dur(d) for d in range(1, self.max_dur+1)])
        self.class_tabel[self.DURATION] = (offset, offset+len(dur))
        tokens.extend(dur)
        offset = len(tokens)

        struct.extend([self.struct(self.NONE)])
        #struct.extend([self.struct(l) for l in [self.BAR, self.BOS, self.EOS]])
        struct.extend([self.struct(l) for l in range(16)])
        self.class_tabel[self.STRUCT] = (offset, offset+len(struct))
        tokens.extend(struct)
        offset = len(tokens)

        for i, token in enumerate(tokens):
            self.token_to_id_tabel[token] = i
            self.id_to_token_tabel.append(token)

    def encode(self, song: Song) -> list:
        # remove empty bars in the begin and end of songs
        start, end = 0, len(song.bars)
        for bar in song.bars:
            if not bar.empty():
                break
            start += 1
        for bar in reversed(song.bars):
            if not bar.empty():
                break
            end -= 1

        song_copy = Song.copy(song)
        song_copy.bars = song.bars[start:end]
        song = song_copy

        if self.use_cp:
            tokens = [[
                self.bar(self.BOS),
                self.tempo(self.BAR),
                self.pos(self.BAR),
                self.pitch(self.BAR),
                self.vel(self.BAR),
                self.dur(self.BAR),
                self.struct(self.NONE)
            ]]
            struct_count = 0
            struct_map = OrderedDict()
            last_struct = None

            for bar in song.bars:
                # struct
                if bar.struct is not None and bar.struct != last_struct:
                    last_struct = bar.struct
                    if bar.struct not in struct_map:
                        struct_map[bar.struct] = self.struct(struct_count)
                        struct_count += 1

                if bar.struct is None:
                    struct = self.struct(self.NONE)
                else:
                    struct = struct_map[bar.struct]

                # bar
                tokens.append([self.bar(1),
                               self.tempo(self.BAR),
                               self.pos(self.BAR),
                               self.pitch(self.BAR),
                               self.vel(self.BAR),
                               self.dur(self.BAR),
                               struct
                ])

                # note
                for event in bar.events:
                    tempo = self._fit_range(event.tempo, self.tempo_base, self.tempo_tick_num, 4)
                    for note in event.notes:
                        pos = note.onset
                        pitch = note.pitch
                        vel = self._fit_range(note.velocity, self.vel_base, self.vel_tick_num, 4)
                        dur = note.duration
                        if dur > self.max_dur:
                            dur = self.max_dur
                        tokens.append([self.bar(0),
                                       self.tempo(tempo),
                                       self.pos(pos),
                                       self.pitch(pitch),
                                       self.vel(vel),
                                       self.dur(dur),
                                       struct
                        ])

            tokens.append([self.bar(self.EOS),
                           self.tempo(self.BAR),
                           self.pos(self.BAR),
                           self.pitch(self.BAR),
                           self.vel(self.BAR),
                           self.dur(self.BAR),
                           struct_map[last_struct]
            ])

            return tokens
        else:
            tokens = [self.bar(self.BOS)]
            struct_count = 0
            struct_map = {}
            last_struct = None

            for bar in song.bars:
                # struct
                if bar.struct is not None and bar.struct != last_struct:
                    last_struct = bar.struct
                    if bar.struct not in struct_map:
                        struct_map[bar.struct] = self.struct(struct_count)
                        struct_count += 1
                    tokens.append(struct_map[bar.struct])

                # bar
                tokens.append(self.bar(1))

                # note
                for event in bar.events:
                    tempo = self._fit_range(event.tempo, self.tempo_base, self.tempo_tick_num, 4)
                    for note in event.notes:
                        pos = note.onset
                        pitch = note.pitch
                        vel = self._fit_range(note.velocity, self.vel_base, self.vel_tick_num, 4)
                        dur = note.duration
                        if dur > self.max_dur:
                            dur = self.max_dur
                        tokens.extend([self.tempo(tempo), self.pos(pos), self.pitch(pitch), self.vel(vel), self.dur(dur)])
            tokens.append(self.bar(self.EOS))

            return tokens

    def _fit_range(self, val, start, tick_num, step):
        if val < start:
            val = start
        if val > start + (tick_num-1)*step:
            val = start + (tick_num-1) * step
        val = start + ((val-start)//step)*step
        return val

    def token_to_id(self, token):
        token = copy.deepcopy(token)
        if self._is_array(token):
            self._token_to_id_array(token)
        else:
            token = self[token]
        return token

    def _token_to_id_array(self, token):
        for i, item in enumerate(token):
            if self._is_array(item):
                self._token_to_id_array(item)
            else: # token is the last dimension
                token[i] = self[item]

    def id_to_token(self, tid):
        assert isinstance(tid, (list, torch.Tensor, np.ndarray))

        if isinstance(tid, (torch.Tensor, np.ndarray)):
            tid = tid.tolist()
        else:
            tid = copy.deepcopy(tid)

        if self._is_array(tid):
            self._id_to_token_array(tid)
        else:
            tid = self[tid]
        return tid

    def _id_to_token_array(self, tid):
        for i, item in enumerate(tid):
            if self._is_array(item):
                self._id_to_token_array(item)
            else: # tid is the last dimension
                tid[i] = self[item]

    def _is_array(self, obj):
        if isinstance(obj, list):
            return True
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return len(obj.shape) > 0
        else:
            return False

    def class_ranges(self):
        return list(self.class_tabel.values())

    def pad(self, songs, val, max_seq_len):
        tokens = []
        masks = []
        for song in songs:
            if self.use_cp:
                tokens.append(song[:max_seq_len] + [[val]*len(self.class_tabel) for i in range(max_seq_len-len(song))])
            else:
                tokens.append(song[:max_seq_len] + [val for i in range(max_seq_len-len(song))])
            masks.append([1]*len(song[:max_seq_len]) + [0 for i in range(max_seq_len-len(song))])
        tokens = torch.LongTensor(tokens)
        #masks = (tokens != 0).to(torch.float)
        masks = torch.FloatTensor(masks)

        return tokens, masks

    def get_labels(self, segs: torch.LongTensor):
        """
        This function change PAD token to ignore_idx,
        and shift each class token based on class_tabel
        """
        labels = segs.clone().detach()
        labels[labels == self[self.PAD]] = self.ignore_idx

        if self.use_cp:
            """
            labels: (Batch, Sequence, Class)
            """
            for i, (k, v) in enumerate(self.class_tabel.items()):
                labels[:, :, i][labels[:, :, i] != self.ignore_idx] -= v[0]

        if self.use_cp:
            return torch.permute(labels, (2, 0, 1)) # move the class dim to the first
        else:
            return labels[None] # extend 1 dim
        return labels

if __name__ == "__main__":
    pass
    #vocab = Vocabulary("dataset/vocab.txt")
    tknz = Tokenizer()
    print(tknz["PAD"])
    print(tknz[10])
    print(tknz.class_tabel)
