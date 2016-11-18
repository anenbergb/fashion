import pickle
import numpy as np
import os, csv, argparse
import concurrent.futures
FASHION_DIR = "/cvgl/u/anenberg/Fashion144k_stylenet_v1"
SIMILAR_PAIR_DIR = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/similar_pairs"
step_size = 1
similar_thresh = 0.75



def r_metric(i,j,labels):
    """
    Intersection over union of the labels.
    """
    return 1.0*sum(labels[i] & labels[j]) / sum(labels[i] | labels[j])

def find_sim_dis_ims(anchor, labels, ts=0.75, td=0.1, max_it = 100):
    similar = None
    dissimilar = None
    sample_idxs = np.random.choice(np.arange(labels.shape[0]), max_it, replace=False)
    for idx in sample_idxs:
        if dissimilar is not None and similar is not None:
            break
        if idx == anchor:
            continue
        similarity = r_metric(anchor,idx,labels)
        if similarity > ts and similar is None:
            similar = idx
        elif dissimilar < td and dissimilar is None:
            dissimilar = idx
    return (similar, dissimilar)

def triplet(labels, ts=0.75, td=0.1, max_tries=10, max_it=100):
    for _ in range(max_tries):
        anchor = np.random.randint(labels.shape[0])
        similar, dissimilar = find_sim_dis_ims(anchor, labels, ts=ts, td=td, max_it=max_it)
        if similar is not None and dissimilar is not None:
            return (dissimilar, anchor, similar)


def find_similar(
    labels,
    ts,
    idx):
    pairs = {}
    for j in range(idx+1,labels.shape[0]):
        similarity = r_metric(idx,j,labels)
        if similarity > ts:
            pairs[(idx,j)] = similarity
        if j % 10000 == 0:
            print('running {0}...{1:.2f}'.format(idx,100.0*j/labels.shape[0]))
    return pairs

def main():
    color_mat = np.load(os.path.join(FASHION_DIR,'feat/feat_col.npy'))
    single_mat = np.load(os.path.join(FASHION_DIR,'feat/feat_sin.npy'))
    labels = np.hstack([color_mat, single_mat])
    parser = argparse.ArgumentParser(description='Computes similarity between image pairs')
    parser.add_argument('-s','--start', help='start index', type=int, default=0)
    parser.add_argument('-e','--end', help='end index', type=int, default=labels.shape[0])
    parser.add_argument('-t','--threads', help='number of threads',type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(SIMILAR_PAIR_DIR):
        os.mkdir(SIMILAR_PAIR_DIR)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = {}
        for start_idx in range(args.start,args.end):
            futures[executor.submit(find_similar, labels, similar_thresh, start_idx)] = start_idx
        for future in concurrent.futures.as_completed(futures):
            start_idx = futures[future]
            try:
                pairs = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (start_idx, exc))
            else:
                out_file = os.path.join(
                    SIMILAR_PAIR_DIR,
                    "{0}.pkl".format(start_idx))
                with open(out_file, 'wb') as handle:
                    pickle.dump(pairs, handle)
                print('saving {0}'.format(out_file))

if __name__ == "__main__":
    main()