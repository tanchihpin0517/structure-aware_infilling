from dataclasses import dataclass, field
import pretty_midi as pmidi
from typing import List, Optional
import dataclasses
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
import colorsys

@dataclass
class Note:
    pitch: int = -1
    velocity: int = -1
    onset: int = -1
    duration: int = -1

@dataclass
class POP909Note(Note):
    track: str = None
    midi: pmidi.Note = None

    def __post_init__(self):
        self.pitch = self.midi.pitch
        self.velocity = self.midi.velocity

@dataclass
class Event:
    start: float = None
    end: float = None
    chord: str = None
    tempo: int = None
    notes: List[Note] = field(default_factory=list)

    def __repr__(self):
        s = []
        s.append(f"Event(start={self.start}, end={self.end}, tempo={self.tempo}, chord={self.chord})")
        for n in self.notes:
            s.append(" "*8 + str(n))
        return "\n".join(s)

    def __contains__(self, obj):
        if isinstance(obj, float): # time
            return self.start <= obj < self.end
        else:
            raise Exception(f"{type(obj)} is not supported for operator 'in' of Event")


@dataclass
class Bar:
    struct: str = None
    events: List[Event] = field(default_factory=list)
    start: float = None
    end: float = None

    def __repr__(self):
        s = []
        s.append(f"Bar(struct={self.struct}, start={self.start}, end={self.end})")
        for e in self.events:
            s.append(" "*4 + str(e))
        return "\n".join(s)

    def __contains__(self, obj):
        if isinstance(obj, float): # time
            return self.start <= obj < self.end
        else:
            raise Exception(f"{type(obj)} is not supported for operator 'in' of Bar")

    def empty(self):
        if self.struct is not None:
            return False
        for event in self.events:
            if len(event.notes) > 0:
                return False
        return True

@dataclass
class Song:
    name: str
    beat_per_bar: int
    beat_division: int
    bpm: int
    bars: List[Bar] = field(default_factory=list)

    @staticmethod
    def copy(song, with_content=True):
        r = dataclasses.replace(song)
        if with_content:
            r.bars = deepcopy(song.bars)
        else:
            r.bars = []
        return r

    def __repr__(self):
        s = []
        for bar in self.bars:
            s.append(str(bar))
        return "\n".join(s)

    def flatten_events(self):
        events = []
        for bar in self.bars:
            for event in bar.events:
                events.append(event)
        return events

    def flatten_notes(self):
        notes = []
        for bar in self.bars:
            for event in bar.events:
                for note in event.notes:
                    notes.append(note)
        return notes

    def clip(self, start, end):
        r = Song.copy(self, with_content=False)
        r.bars = deepcopy(self.bars[start:end])
        return r

    def save(self, file):
        midi_data = pmidi.PrettyMIDI(initial_tempo=self.bpm)
        inst = pmidi.Instrument(program=0)
        for event in self.flatten_events():
            event_time = event.end - event.start
            for note in event.notes:
                midi_note = pmidi.Note(
                    #velocity=note.velocity,
                    velocity=100,
                    pitch=note.pitch,
                    start=event.start,
                    end=event.start+note.duration*event_time
                )
                inst.notes.append(midi_note)
        midi_data.instruments.append(inst)
        midi_data.write(file)

    def save_fig(self, file, style='paper'):
        start_time = int(self.bars[0].start)
        end_time = int(self.bars[-1].end)

        #fig, ax = plt.subplots(figsize=((end_time - start_time)*0.03, 5))
        fig, ax = plt.subplots(figsize=((end_time - start_time)*0.5, 5))

        if style == 'dark':
            ax.set_facecolor((0,0,0))
        if style == 'paper':
            ax.set_facecolor((1,1,1))

        padY=0.02
        padX=0.02

        last_struct = None
        for bar in self.bars:
            if bar.struct != last_struct:
                last_struct = bar.struct
                ax.vlines(bar.start, 50, 100, 'red')
                ax.text(bar.start+1, 50, f"{bar.struct}({round(bar.start, 1)})", color='red', fontsize = 20)

            for event in bar.events:
                event_start = event.start
                event_time = event.end - event.start

                for note in event.notes:
                    onset = max(start_time, event_start)
                    offset = min(end_time, event_start+event_time*note.duration)
                    #color = colorsys.hsv_to_rgb(note[4]/10,.9,.9)
                    color = self._note_color(note.pitch, style)
                    ax.broken_barh([(onset+padX/2,offset-onset-padX/2)],(note.pitch-0.5+padY/2,1-padY),facecolors=[color])

        plt.savefig(file)

    def _note_color(self, p, style='dark'):
        t = p*0.08333
        if style == 'dark':
            return colorsys.hsv_to_rgb(t, 1, 1)
        if style == 'paper':
            return colorsys.hsv_to_rgb(t, 1, 0.8)
