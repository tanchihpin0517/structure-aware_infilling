import os
import sys
import argparse
import re
import shutil
from template import Template

parser = argparse.ArgumentParser()
parser.add_argument('-songs_dir', type=str, default='./assets/songs')
parser.add_argument('-templates_dir', type=str, default='./templates')
parser.add_argument('-index_file', type=str, default='./index.html')
args = parser.parse_args()
print(args.songs_dir)
print(args.templates_dir)
print(args.index_file)

def main():
    template = Template(os.path.join(args.templates_dir, "index.html"))
    songs = []
    losses = []
    for song_dir in os.listdir(args.songs_dir):
        loss = song_dir
        song_dir = os.path.join(args.songs_dir, song_dir)
        if os.path.isdir(song_dir):
            losses.append(loss)
            for song in os.listdir(song_dir):
                idx = song.split(".")[0]
                if idx not in songs:
                    songs.append(idx)
    songs.sort()
    #songs = songs[:4]
    losses.sort()
    losses = losses[:4]

    table = {"losses":losses, "songs":songs, "github":False}
    content = template.render(table)
    with open(args.index_file, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
