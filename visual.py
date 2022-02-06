'''
Usage: 

import visual
score = visual.load_score(f'./gen_tokens/0.1/003.json')
score.show(0,256,style='dark')
'''

import matplotlib.pyplot as plt
import numpy as np
import colorsys

def note_color(p,v,style = 'dark'):
    t=p*0.08333
    if style == 'dark':
        return colorsys.hsv_to_rgb(t,1,v/128.0)
    if style == 'paper':
        return colorsys.hsv_to_rgb(t,1,v/128.0*0.8)

import json
from main import load_vocab

def load_score(ids_file_name):
    vocab = load_vocab('./dataset/vocab.txt')
    with open(ids_file_name) as f:
        ids = json.load(f)
    score=Score()
    get_number_in_parentheses = lambda token : int(token.split('(')[1].split(')')[0])
    bar_start_time = 0
    current_label=-2

    label_next = False
    bog_next = False

    for id in ids:
        
        token = vocab.id_to_token[id]
        if token == 'EOS': break
        if 'Label' in token:
            current_label = get_number_in_parentheses(token)
            label_next = True
        if 'BOG' in token:
            bog_next = True

        if 'Bar' in token:
            bar_start_time += 16
        
        if 'Position' in token:
            start = bar_start_time + get_number_in_parentheses(token)
            if label_next:
                score.events.append(Event(start,current_label))
                label_next = False

            if bog_next:
                score.events.append(Event(start,'BOG','green'))
                bog_next = False

        if 'Pitch' in token:
            pitch = get_number_in_parentheses(token)

        if 'Duration' in token:
            end = start + get_number_in_parentheses(token)
            score.notes.append(Note(start,end,pitch))

    score.calculate_duration()
    print(f'Length: {score.duration} ({score.duration//16} bars)')
    return score

from dataclasses import dataclass
from typing import Union

@dataclass
class Event:
    time : float
    label : str = ''
    color : Union[str,tuple] = 'red'

@dataclass
class Note:
    onset : float
    offset : float
    pitch : float
    velocity : float = 127

class Score():

    def __init__(self):
        self.notes : list[Note] = [] # onset, offset, pitch, velocity
        self.events : list[Event] = []
        self.duration = 0
    
    def calculate_duration(self):
        for note in self.notes:
            if note.offset>self.duration:
                self.duration = note.offset
    
    def show(self,start_time=0,end_time=np.inf,style='dark'):
        
        self.calculate_duration()
        if(end_time>self.duration): end_time = self.duration
        
        fig, ax = plt.subplots(figsize=((end_time- start_time)*0.03,5))

        if style == 'dark':
            ax.set_facecolor((0,0,0))
        if style == 'paper':
            ax.set_facecolor((1,1,1))
        
        padY=0.02
        padX=0.02
        
        for note in self.notes:
            if note.onset>end_time:
                break
            if note.offset<start_time:
                continue
            onset = max(start_time,note.onset)
            offset = min(end_time,note.offset)
            #color = colorsys.hsv_to_rgb(note[4]/10,.9,.9)
            color = note_color(note.pitch,note.velocity,style)
            ax.broken_barh([(onset+padX/2,offset-onset-padX/2)],(note.pitch-0.5+padY/2,1-padY),facecolors=[color])

        for event in self.events:
            time = event.time
            if time>=start_time and time<=end_time:
                ax.vlines(time,50,100,event.color)
                ax.text(time+2,50,event.label,color=event.color,fontsize = 20)