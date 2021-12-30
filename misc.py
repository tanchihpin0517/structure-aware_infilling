import sys

class Note:
    def __init__(self, midi, onset=None, duration=None, track=None):
        self.midi = midi
        self.pitch = midi.pitch
        self.velocity = midi.velocity
        self.onset = onset
        self.duration = duration
        self.track = track

    def __repr__(self):
        return f"Note(pitch={self.pitch},duration={self.duration},track={self.track})"

class Event:
    def __init__(self, time=None, struct=None, chord=None, bar=None, notes=None):
        self.time = time
        self.struct = struct
        self.chord = chord
        self.bar = bar
        if notes is None:
            self.notes = []
        else:
            self.notes = notes

    def __repr__(self):
        return f"time={'%.2f' % self.time}, label={self.struct}, chord={self.chord}, bar={self.bar}, {self.notes}"

class Song(list):
    def __init__(self, name=None, beat_per_bar=None, beat_division=None, bpm=None, info=None, content=None):
        super().__init__()
        self.name = name
        self.beat_per_bar = beat_per_bar
        self.beat_division = beat_division
        self.bpm = bpm
        if info is not None:
            self.info_copy(info)
        if content is not None:
            self.extend(content)

    def info_copy(self, song):
        self.name = song.name
        self.beat_per_bar = song.beat_per_bar
        self.beat_division = song.beat_division
        self.bpm = song.bpm

class Vocabulary:
    def __init__(self, vocab_file=None):
        self.token_to_id = {}
        self.id_to_token = []

        if vocab_file is None:
            self._init_vocab()
        else:
            with open(vocab_file, "r") as f:
                for i, token in enumerate(map(lambda t: t.strip(), f)):
                    self.token_to_id[token] = i
                    self.id_to_token.append(token)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.token_to_id[key]
        elif isinstance(key, int):
            return self.id_to_token[key]
        else:
            raise TypeError(f"key must be str or int, not: {type(key)}")

    def save(self, vocab_file):
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.id_to_token))

    def size(self):
        return len(self.id_to_token)

    def _init_vocab(self):
        tokens = [
            "PAD",
            "MASK",
            "BOS",
            "EOS",
            "RESERVED",
            "RESERVED",
            "RESERVED",
            "RESERVED",
            "RESERVED",
            "RESERVED",
        ]
        tokens.extend([f"Bar({b})" for b in range(2)])
        tokens.extend([f"Tempo({t})" for t in range(28, 216, 4)])
        tokens.extend([f"Position({p})" for p in range(16)])
        tokens.extend([f"Pitch({p})" for p in range(22, 108)])
        tokens.extend([f"Velocity({v})" for v in range(0, 132, 4)])
        tokens.extend([f"Duration({d})" for d in range(1,32+1)])
        tokens.extend([f"Label({l})" for l in range(16)])

        for i, token in enumerate(tokens):
            self.token_to_id[token] = i
            self.id_to_token.append(token)

class Checkpoint:
    def __init__(self, epoch, config, model_state_dict, optim_state_dict, loss, vocab):
        self.epoch = epoch
        self.config = config
        self.model_state_dict = model_state_dict
        self.optim_state_dict = optim_state_dict
        self.loss = loss
        self.vocab = vocab

if __name__ == "__main__":
    pass
    #vocab = Vocabulary("dataset/vocab.txt")
    vocab = Vocabulary()
    print(vocab["PAD"])
    print(vocab[10])
