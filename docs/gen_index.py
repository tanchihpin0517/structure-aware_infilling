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

    models = ["Ours", "VLI", "Hsu", "Real"]
    songs = list(range(1,15+1))

    table = {
        "songs":songs,
        "models":models,
        "github":True,
        "gh_url":"https://cdn.jsdelivr.net/gh/tanchihpin0517/structure-aware_infilling",
        "repo":"https://github.com/tanchihpin0517/structure-aware_infilling",
        "arxiv":"",
    }
    content = template.render(table)
    with open(args.index_file, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
