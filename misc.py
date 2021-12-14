import sys

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
    def __init__(self, name=None, beat_per_bar=None, beat_division=None):
        super().__init__()
        self.name = name
        self.beat_per_bar = beat_per_bar
        self.beat_division = beat_division
