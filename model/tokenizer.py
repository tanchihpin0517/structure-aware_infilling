import os
from typing import List, Tuple
from music import Song, Bar, Event, Note
from collections.abc import Iterable
import copy
import torch
import numpy as np
from collections import OrderedDict
import utils
from copy import deepcopy

class Tokenizer:

    PAD = "PAD"
    MASK = "MASK"
    BOS = "BOS"
    EOS = "EOS"
    BOP = "BOP"
    EOP = "EOP"
    BAR = "Bar"
    TEMPO = "Tempo"
    POSITION = "Position"
    PITCH = "Pitch"
    VELOCITY = "Velocity"
    DURATION = "Duration"
    STRUCT = "Struct"
    NONE = "None"
    NONE_ID = -1

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

    def eop_id(self):
        return self[self.EOP]

    def eos_id(self):
        return self.token_to_id(self.bar(self.EOS))

    def bar_id(self, bar_num, struct=None):
        return self.token_to_id(self.bar(bar_num if self.use_bar_cd else self.bar(1)))

    def __init__(self, vocab_file=None, ignore_idx=-100, use_bar_cd=True, verbose=True):
        self.token_to_id_tabel = {}
        self.id_to_token_tabel = []
        self.use_bar_cd = use_bar_cd
        self.class_tabel = OrderedDict()
        self.ignore_idx = -100

        self.max_dur = 16
        self.tempo_base = 28
        self.tempo_tick_num = 47
        self.vel_base = 0
        self.vel_tick_num=33
        self.struct_num = 8
        self.max_bar_num = 32

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
        REMI:
            begin: bos
            ...
            k-2: bar(1)
            k-1: struct
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
            #self.MASK, # not used
            #self.BOS, # not used (bar(BOS) instead)
            #self.EOS, # not used (bar(EOS) instead)
            self.BOP,
            self.EOP,
        ]
        tokens = tokens + ["RESERVED"] * (10 - len(tokens))

        bar, tempo, pos, pitch, vel, dur, struct = [[] for _ in range(7)]
        offset = len(tokens)

        """
        we didn't add bar(BOP) and bar(EOP) because those two tokens are not generated by model
        """
        bar.extend([self.bar(t) for t in [self.BOS, self.EOS]])
        bar.extend([self.bar(b) for b in range(self.max_bar_num+1)]) # bar count-down technic
        self.class_tabel[self.BAR] = (offset, offset+len(bar))
        tokens.extend(bar)
        offset = len(tokens)

        tempo.extend([self.tempo(self.BAR)])
        tempo.extend([self.tempo(t) for t in range(28, 28+47*4, 4)])
        self.class_tabel[self.TEMPO] = (offset, offset+len(tempo))
        tokens.extend(tempo)
        offset = len(tokens)

        pos.extend([self.pos(self.BAR)])
        pos.extend([self.pos(p) for p in range(16)])
        self.class_tabel[self.POSITION] = (offset, offset+len(pos))
        tokens.extend(pos)
        offset = len(tokens)

        pitch.extend([self.pitch(self.BAR)])
        pitch.extend([self.pitch(p) for p in range(22, 22+86)])
        self.class_tabel[self.PITCH] = (offset, offset+len(pitch))
        tokens.extend(pitch)
        offset = len(tokens)

        dur.extend([self.dur(self.BAR)])
        dur.extend([self.dur(d) for d in range(1, self.max_dur+1)])
        self.class_tabel[self.DURATION] = (offset, offset+len(dur))
        tokens.extend(dur)
        offset = len(tokens)

        struct.extend([self.struct(self.NONE)])
        struct.extend([self.struct(l) for l in range(self.struct_num)])
        self.class_tabel[self.STRUCT] = (offset, offset+len(struct))
        tokens.extend(struct)
        offset = len(tokens)

        for i, token in enumerate(tokens):
            self.token_to_id_tabel[token] = i
            self.id_to_token_tabel.append(token)

    def encode(self, song: Song, with_eos=True):
        bar_id = []
        struct_id = []
        struct_index = []
        bar_count = 0
        struct_range = []
        s_range_start = 0

        tokens = [self.bar(self.BOS)]
        bar_id.append(bar_count) # bos is not a new bar
        sid = self.NONE_ID
        struct_id.append(self.NONE_ID)
        struct_index.append(0)

        struct_count = 0
        struct_map = OrderedDict()

        for s_label, s_start, s_end in song.struct_indices:
            s_len = s_end - s_start

            struct_range.append((sid, s_range_start, len(struct_index)))
            s_range_start = len(struct_index)

            if s_label is None:
                struct = self.struct(self.NONE)
                sid = self.NONE_ID
            else:
                if s_label not in struct_map:
                    struct_map[s_label] = struct_count
                    struct_count += 1
                struct = self.struct(struct_map[s_label])
                sid = struct_map[s_label]
            sidx = struct_index[-1] + 1

            for bar_i, bar in enumerate(song.bars[s_start:s_end]):
                # bar
                tokens.append(self.bar(s_len-bar_i) if self.use_bar_cd else self.bar(1))
                bar_id.append(bar_count)
                bar_count += 1
                struct_id.append(sid)
                struct_index.append(sidx)

                tokens.append(struct)
                bar_id.append(bar_count)
                struct_id.append(sid)
                struct_index.append(sidx)

                # note
                for event in bar.events:
                    tempo = self._fit_range(event.tempo, self.tempo_base, self.tempo_tick_num, 4)
                    if len(event.notes) > 0:
                        pos = event.notes[0].onset
                        tokens.extend([
                            self.tempo(tempo),
                            self.pos(pos),
                        ])
                        bar_id.extend([bar_count] * 2)
                        struct_id.extend([sid] * 2)
                        struct_index.extend([sidx] * 2)

                    for note in event.notes:
                        pitch = note.pitch
                        #vel = self._fit_range(note.velocity, self.vel_base, self.vel_tick_num, 4)
                        dur = note.duration
                        if dur > self.max_dur:
                            dur = self.max_dur

                        tokens.extend([
                            self.pitch(pitch),
                            self.dur(dur),
                        ])
                        bar_id.extend([bar_count] * 2)
                        struct_id.extend([sid] * 2)
                        struct_index.extend([sidx] * 2)

        struct_range.append((sid, s_range_start, len(struct_index)))

        if with_eos:
            tokens.append(self.bar(self.EOS))
            bar_id.append(bar_count)
            struct_id.append(self.NONE_ID)
            struct_index.append(struct_index[-1] + 1)

        del struct_range[0] # remove range of BOS

        assert len(tokens) == len(bar_id) == len(struct_id) == len(struct_index)
        assert len(song.struct_indices) == len(struct_range)
        return tokens, bar_id, struct_id, struct_index, struct_range

    def decode(self, tokens, empty_song: Song) -> Song:
        assert len(empty_song.bars) == 0
        song = empty_song
        event_time = round((60 / song.bpm) / song.beat_division, 8)
        time_offset = 0.0

        for token in tokens:
            tag, val = self._get_tag_and_val(token)
            if val == self.BOS or val == self.EOS:
                continue
            elif tag == self.BAR:
                song.bars.append(Bar(struct=None))
                for _ in range(song.beat_per_bar*song.beat_division):
                    start = time_offset
                    end = time_offset + event_time
                    song.bars[-1].events.append(Event(start=start, end=end))
                    time_offset += event_time
                song.bars[-1].start = song.bars[-1].events[0].start
                song.bars[-1].end = song.bars[-1].events[-1].end
            elif tag == self.TEMPO:
                tempo = val
            elif tag == self.POSITION:
                pos = val
            elif tag == self.PITCH:
                pitch = val
            elif tag == self.VELOCITY:
                vel = val
            elif tag == self.DURATION:
                try:
                    dur = val
                    #note = Note(pitch=pitch, velocity=vel, onset=pos, duration=dur)
                    note = Note(pitch=pitch, onset=pos, duration=dur)
                    song.bars[-1].events[pos].tempo = tempo
                    song.bars[-1].events[pos].notes.append(note)
                except UnboundLocalError as e:
                    print(f"song {song.name}:", e)
            elif tag == self.STRUCT:
                struct = val
                if struct != self.NONE:
                    song.bars[-1].struct = str(struct)

        return song

    def _get_tag_and_val(self, token):
        tag = token.split("(", 1)[0]
        val = token.split("(", 1)[1].rsplit(")", 1)[0]
        try:
            val = int(val)
            return tag, val
        except ValueError:
            return tag, val

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
        assert isinstance(tid, (int, list, torch.Tensor, np.ndarray)), f"{type(tid)} is not allowed."

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

    def pad(self, inputs, val, max_seq_len, gen_mask=True):
        """
        attention mask: 0 -> not attend, 1 -> attend
        """
        inputs = copy.deepcopy(inputs)
        outputs = []
        masks = [] if gen_mask else None

        use_last = True if val == "last" else False

        for tokens in inputs:
            val = tokens[-1] if use_last else val
            outputs.append(tokens[:max_seq_len] + [val for i in range(max_seq_len-len(tokens))])
            if gen_mask:
                masks.append([1]*len(tokens[:max_seq_len]) + [0 for i in range(max_seq_len-len(tokens))])
        outputs = torch.LongTensor(outputs)
        masks = torch.FloatTensor(masks) if gen_mask else None

        return outputs, masks

    def get_labels(self, segs: torch.LongTensor, ignore_labels=None):
        """
        This function change PAD token to ignore_idx,
        and shift each class token based on class_tabel
        """
        labels = segs.clone().detach()
        labels[labels == self[self.PAD]] = self.ignore_idx
        labels[labels == self[self.EOP]] = self.ignore_idx # for segments

        if ignore_labels is not None:
            labels[ignore_labels > 0] = self.ignore_idx

        return labels[None] # extend 1 dim

    def sample(self, pred_scores):
        pred_ids = utils.nucleus(pred_scores[0])
        return pred_ids

    def is_eos(self, token):
        t = token

        if t == self.bar(self.EOS): # str
            return True
        elif self[t] == self.bar(self.EOS):
            return True
        else:
            return False

    def is_bar(self, token):
        t = token

        if t in [self.bar(i) for i in range(self.max_bar_num)]: # str
            return True
        elif self[t] in [self.bar(i) for i in range(self.max_bar_num)]:
            return True
        else:
            return False

    def is_legal(self, token):
        return True # always legal while not using cp

    def extract_struct(self, song_ids, struct_ids, struct_indices, max_struct_len=512, mask_first_time=False, struct_ratio=None):
        song_ids = deepcopy(song_ids)
        struct_ids = deepcopy(struct_ids)
        struct_indices = deepcopy(struct_indices)

        dim = (len(song_ids), self.struct_num, max_struct_len)
        struct_seqs = torch.zeros(dim).long()
        struct_seq_masks = torch.zeros((len(song_ids), self.struct_num, max_struct_len))
        """
        mask:
            0 => attend
            1 => not attend
        struct_masks: mask to indicate whether doing cross attention to struct sequence for each input token
        """
        struct_masks = []
        struct_lens = []
        print("struct ratio:", struct_ratio)

        for i in range(len(song_ids)):
            song_id = song_ids[i]
            struct_id = struct_ids[i]
            struct_index = struct_indices[i]
            assert len(song_id) == len(struct_id) == len(struct_index)

            s_start = 0
            appear = set()
            mask = []
            while s_start < len(struct_index):
                s_end = s_start
                while s_end < len(struct_index) and struct_index[s_end] == struct_index[s_start]:
                    s_end += 1
                next_start = s_end

                if struct_ratio:
                    s_end = s_start + int((s_end-s_start) * struct_ratio)

                sid = struct_id[s_start]
                slen = s_end - s_start


                if mask_first_time:
                    """
                    input tokens will not attend to struct sequnces which appear at the first time
                    """
                    if sid == self.NONE_ID or sid not in appear:
                        mask.extend([1] * (next_start - s_start))
                    else:
                        mask.extend([0] * (next_start - s_start))
                else:
                    if sid == self.NONE_ID:
                        mask.extend([1] * (next_start - s_start))
                    else:
                        mask.extend([0] * (next_start - s_start))


                if sid != self.NONE_ID and sid not in appear:
                    appear.add(sid)
                    pad = 0
                    seq = torch.LongTensor(song_id[s_start:s_end] + [pad]*(max_struct_len-slen))
                    seq_mask = torch.FloatTensor([0]*(min(slen, max_struct_len)) + [1]*(max_struct_len-slen))

                    struct_seqs[i][sid] = seq[:max_struct_len]
                    struct_seq_masks[i][sid] = seq_mask
                    struct_lens.append(slen)

                s_start = next_start
            assert len(song_id) == len(mask)
            struct_masks.append(mask)

        struct_lens = np.array(struct_lens)
        print("mean of struct length:", struct_lens.mean())
        print("mean + 2*std of struct length:", struct_lens.mean() + 2*struct_lens.std())
        print("max of struct length:", struct_lens.max())

        return struct_seqs, struct_seq_masks, struct_masks

if __name__ == "__main__":
    pass
    #vocab = Vocabulary("dataset/vocab.txt")
    tknz = Tokenizer()
    print(tknz["PAD"])
    print(tknz[10])
    print(tknz.class_tabel)
