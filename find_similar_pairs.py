import pickle
import numpy as np
import os, csv, sys
from myThreadPool import ThreadPool
FASHION_DIR = "/cvgl/u/anenberg/Fashion144k_stylenet_v1"
SIMILAR_PAIR_DIR = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/similar_pairs"
kThreads = 100
step_size = 1
similar_thresh = 0.75

def r_metric(i,j,labels):
    """
    Intersection over union of the labels.
    """
    return float(sum(labels[i] & labels[j])) / sum(labels[i] | labels[j])

def find_similar(labels,ts,start_idx,end_idx,out_file):
    #ts=0.75, start_idx=0, end_idx=None
    print('starting [{0}:{1})'.format(start_idx,end_idx))
    pairs = {}
    if end_idx is None:
        end_idx = labels.shape[0]
    for i in range(start_idx, end_idx):
        for j in range(i+1,labels.shape[0]):
            similarity = r_metric(i,j,labels)
            if similarity > ts:
                pairs[(i,j)] = similarity
            if j % 100 == 0:
                print('running {0} in [{1}:{2})...{3:.2f}'.format(i,start_idx,end_idx,100.0*j/labels.shape[0]))
    with open(out_file, 'wb') as handle:
        pickle.dump(pairs, handle)
    print('finished [{0}:{1}), saving...{2}'.format(start_idx,end_idx,os.path.basename(out_file)))


if __name__ == "__main__":
    color_mat = np.load(os.path.join(FASHION_DIR,'feat/feat_col.npy'))
    single_mat = np.load(os.path.join(FASHION_DIR,'feat/feat_sin.npy'))
    labels = np.hstack([color_mat, single_mat])

    if not os.path.exists(SIMILAR_PAIR_DIR):
        os.mkdir(SIMILAR_PAIR_DIR)

    pool = ThreadPool(kThreads)
    for start_idx in range(0,labels.shape[0],step_size):
        out_file = os.path.join(
                    SIMILAR_PAIR_DIR,
                    "{0}_{1}.pkl".format(
                        start_idx,
                        start_idx+step_size)
                    )
        pool.add_task(
            find_similar,
            labels,
            similar_thresh,
            start_idx,
            start_idx+step_size,
            out_file
            )
    pool.wait_completion()
