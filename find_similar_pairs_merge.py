import pickle
import numpy as np
import os, csv, argparse
FASHION_DIR = "/cvgl/u/anenberg/Fashion144k_stylenet_v1"
SIMILAR_PAIR_DIR = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/similar_pairs"
import pdb



def main():
    parser = argparse.ArgumentParser(description='Merges similar pairs dictionaries')
    parser.add_argument('-k','--number', help='number of dictionaries to merge \
                        starting from 0.pkl',type=int, default=50)
    args = parser.parse_args()
    pairs = {}
    processed = []
    pkl_files = os.listdir(SIMILAR_PAIR_DIR)
    pkl_files_filt = []
    for f in pkl_files:
        f_split = f.split('.')
        if len(f_split) == 2 and f_split[0].isdigit() and f_split[1] == 'pkl':
            pkl_files_filt.append(f)
    pkl_files_filt = sorted(pkl_files_filt, key = lambda x: int(x.split('.')[0]))

    for i, f in enumerate(pkl_files_filt[:args.number]):
        with open(os.path.join(SIMILAR_PAIR_DIR, f), "rb") as o:
            fdict = pickle.load(o)
            pairs.update(fdict)
        if i % 100 == 0:
            print("joined {0}".format(f))
    with open(os.path.join(FASHION_DIR, "similar_pairs.pkl"), "wb") as f:
        pickle.dump(pairs, f, protocol=2)



if __name__ == "__main__":
    main()
