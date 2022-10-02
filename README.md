# structural_expansion

The official implementation of ISMIR 2022 paper: [Melody Infilling with User-Provided Structural Context]().

## Installation
Clone datasets:
```sh
git submodule init
git submodule update
```

Install dependencies:
```sh
pip install -r requirements.txt
```

Generate training and testing data:
```sh
python main.py --preprocess --data-file "./dataset/pop909.pickle"
python prepare_experiment.py
```

## Training
From scratch:
```sh
TN_DATA_FILE="./dataset/pop909.pickle.training"
VAL_DATA_FILE="./dataset/pop909.pickle.testing"
python main.py --train --cuda \
    --infilling \
    --struct-ratio 1.0 \
    --dim-model 512 \
    --dim-inner 2048 \
    --dim-subembed 512 \
    --num-layer 6 \
    --num-head 8 \
    --seg-size 2048 \
    --mem-len 2048 \
    --max-struct-len 512 \
    --batch-size 4 \
    --epoch-num 1000 \
    --training-data $TN_DATA_FILE \
    --testing-data $VAL_DATA_FILE \
    --accm-step 1 \
    --save-path "./trained_model/loss_%d.ckpt" \
    # --with-past # set this flag if you want to consider the loss of past context
```
Resume from checkpoint:
```sh
TN_DATA_FILE="./dataset/pop909.pickle.training"
VAL_DATA_FILE="./dataset/pop909.pickle.testing"
python main.py --train --cuda \
    --infilling \
    --struct-ratio 1.0 \
    --dim-model 512 \
    --dim-inner 2048 \
    --dim-subembed 512 \
    --num-layer 6 \
    --num-head 8 \
    --seg-size 2048 \
    --mem-len 2048 \
    --max-struct-len 512 \
    --batch-size 4 \
    --epoch-num 1000 \
    --training-data $TN_DATA_FILE \
    --testing-data $VAL_DATA_FILE \
    --accm-step 1 \
    --save-path "./trained_model/loss_%d.ckpt" \
    --ckpt-path "./trained_model/$CHECKPOINT_FILE" \ # the checkpoint you want to continue from
    # --with-past # set this flag if you want to consider the loss of past context
```

## Test
Generate the result with the experiment setting of this paper:
```sh
CKPT="./trained_model/$CHECKPOINT_FILE" # select checkpoint
DATA_FILE="./dataset/testing_data.pickle"
python main.py --experiment --cuda \
    --infilling \
    --struct-ratio 1.0 \
    --seg-size 2048 \
    --gen-num 16 \
    --max-gen-len 4096 \
    --ckpt-path $CKPT \
    --data-file $DATA_FILE \
    --save-path "./experiment_result"
```

Generate with the custom song:
```sh
CKPT="./trained_model/$CHECKPOINT_FILE"
python main.py --generate --cuda \
    --infilling \
    --struct-ratio 1.0 \
    --seg-size 2048 \
    --gen-num 16 \
    --max-gen-len 4096 \
    --ckpt-path $CKPT \
    --song-file "./$SONG_FILE" \
    --save-path "./custom_result"
```
The 'song file' is provided by the user.
The content of the song file looks like:
```
Line 1: BPM BPB(beat per bar)
Repeat for each structure section {
    Label [S or T: optional]
    Bar 1: Note(tempo position pitch duration), Note(tempo position pitch duration), ...
    Bar 2: Note(tempo position pitch duration), Note(tempo position pitch duration), ...
    ...
}
```
The first line of this file contains two value: BPM(beat per minut) and BPB(beat per bar), which are seperated by space.
The remaining content is composed by repeated structure blocks.

For each structure block, the first line contains the structure label and an optional tag, 'S'(source) or 'T'(target).
If 'S' is presented, it means this section will be used as the "structural context" for generation.
If 'T' is presetned, it means this section is the target which will replaced by the infilling result.

The lines following the label represent the bars of this section (one line for one bar).
Each bar contains multiple notes formatted by 4-element tuples: (tempo, position, pitch, duration)

Here is an example of the song file (see the whole content in "./custom_song.txt):
```
90 4 // BPM: 90, Time signature: 4/4
x // this is the first structure section
90 14 66 2
// <--- snipped --->
A S // 'S' is presented, which mean this section will be used as the structural context
90 0 70 1 90 0 59 2 90 0 63 3 90 0 47 5 90 1 78 1 90 2 66 1 90 2 80 1 90 2 54 3 90 3 82 1 90 3 59 2 90 4 63 1 90 4 66 1 90 6 68 6 90 6 65 4 90 6 61 5 90 8 49 5 90 10 80 1 90 10 56 2 90 11 82 1 90 12 80 1 90 12 61 1 90 12 65 1 90 12 68 2 90 14 61 1
90 0 68 1 90 0 58 2 90 0 61 2 90 0 46 5 90 2 65 1 90 2 53 3 90 3 58 1 90 4 61 1 90 4 61 1 90 4 58 1 90 4 65 1 90 6 66 5 90 6 58 4 90 6 63 4 90 8 78 1 90 8 51 6 90 9 80 1 90 10 82 1 90 10 54 4 90 12 61 1 90 12 58 1 90 12 63 2 90 12 66 2 90 13 63 1 90 14 66 1 90 14 58 1 90 15 68 1
90 0 70 1 90 0 59 2 90 0 63 2 90 0 47 4 90 2 66 1 90 2 54 3 90 3 59 1 90 4 63 1 90 4 63 1 90 4 59 1 90 4 66 1 90 6 68 3 90 6 65 4 90 6 61 4 90 8 49 5 90 10 61 1 90 10 56 3 90 12 68 1 90 12 61 1 90 12 68 2 90 12 65 3 90 14 66 7 90 14 61 1
90 0 58 2 90 0 54 3 90 0 42 15 90 2 49 3 90 4 54 1 90 4 58 2 90 4 61 2 90 6 49 3 90 7 54 1 90 8 66 2 90 8 61 3 90 8 58 3 90 8 54 3 90 10 49 3 90 12 54 2 90 12 58 3 90 14 49 1 90 15 54 1
// <--- snipped --->
A T // this section with the length of 4 bars (4 lines in the file) will be replaced by the 4-bar infilling result.
90 0 70 1 90 0 59 2 90 0 63 3 90 0 47 5 90 1 78 1 90 2 66 1 90 2 80 1 90 2 54 3 90 3 82 1 90 3 59 2 90 4 63 1 90 4 66 1 90 6 68 6 90 6 65 4 90 6 61 5 90 8 49 5 90 10 80 1 90 10 56 2 90 11 82 1 90 12 80 1 90 12 61 1 90 12 65 1 90 12 68 2 90 14 61 1
90 0 68 1 90 0 58 2 90 0 61 2 90 0 46 5 90 2 65 1 90 2 53 3 90 3 58 1 90 4 61 1 90 4 61 1 90 4 58 1 90 4 65 1 90 6 66 5 90 6 58 4 90 6 63 4 90 8 78 1 90 8 51 6 90 9 80 1 90 10 82 1 90 10 54 4 90 12 61 1 90 12 58 1 90 12 63 2 90 12 66 2 90 13 63 1 90 14 66 1 90 14 58 1 90 15 68 1
90 0 70 1 90 0 59 2 90 0 63 2 90 0 47 4 90 2 66 1 90 2 54 3 90 3 59 1 90 4 63 1 90 4 63 1 90 4 59 1 90 4 66 1 90 6 68 3 90 6 65 4 90 6 61 4 90 8 49 5 90 10 61 1 90 10 56 3 90 12 68 1 90 12 61 1 90 12 68 2 90 12 65 3 90 14 70 7 90 14 61 1
90 0 58 2 90 0 54 3 90 0 42 15 90 2 49 3 90 2 78 1 90 4 85 1 90 4 54 1 90 4 58 2 90 4 61 2 90 4 87 2 90 6 85 1 90 6 49 3 90 7 54 1 90 8 85 1 90 8 66 2 90 8 61 3 90 8 58 3 90 8 54 3 90 8 87 2 90 10 85 1 90 10 49 3 90 12 80 1 90 12 82 1 90 12 61 1 90 12 54 2 90 12 58 3 90 13 63 1 90 13 80 1 90 14 66 1 90 14 49 1 90 14 78 2 90 15 68 1 90 15 54 1
// <--- snipped --->
```
Note that the length of the infilling results equals to the length of the target section.
If you want to generate 4 bars results, you can just provide 4 empty lines in the song file.

The reason we require users to generate thier custom song file is that the songs in real world are not easy to read.
There are many things should be considered like time alignment, and we cannot handle every exceptions.
We choose to let the program reading a fixed format, leaving the convertion work for the users.