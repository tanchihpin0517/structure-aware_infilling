import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict-file', type=str, default='dictionary.pickle')
    parser.add_argument('--data-file', type=str, default='worded_data.pickle')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--save-path', type=str, default="trained-model", help='folder to save checkpoints')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--target-max-percent', type=float, default=0.25, help="Up to `seq_len *                 target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--n-step-bars', type=int, default=8, help='how many bars to step before next training   data fetching (the smaller the more training data)')
    parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')parser.add_argument('--train-epochs', type=int, default=2000, help='number of training epochs')
    parser.add_argument('--init-lr', type=float, default=1e-4, help='initial learning rate')

# for prediction phase
    parser.add_argument('--test-data-file', type=str, default='worded_data.pickle')
    parser.add_argument('--ckpt-path', type=str, default="trained-model/loss.ckpt", help='checkpoint to load.')
    parser.add_argument('--song-idx', type=int, default=170)
    return parser.parse_args()


def main():
    args = parse_args()



if __name__ == "__main__":
    main()
