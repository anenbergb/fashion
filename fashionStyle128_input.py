"""
Fashion 144k dataset input module.
The dataset can be downloaded from this page:
http://hi.cs.waseda.ac.jp/~esimo/en/research/stylenet/
"""

# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import pdb
import os, pickle, csv
import numpy as np
from scipy import misc

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import textwrap as tw




class DataSetClass():
    """
    self.similar_pairs : pairs of images whose rscore is above 0.75
    """
    def __init__(self, dataset_path, similar_pairs_pkl, td=0.1, max_tries=50):
        self.dataset_path = dataset_path
        self.color_mat = np.load(os.path.join(dataset_path,'feat/feat_col.npy'))
        self.single_mat = np.load(os.path.join(dataset_path,'feat/feat_sin.npy'))
        self.labels = np.hstack([self.color_mat, self.single_mat])
        with open(os.path.join(dataset_path, similar_pairs_pkl), 'rb') as f:
            self.similar_pairs = pickle.load(f)
            self.sim_pairs_list = self.similar_pairs.keys()

        with open(os.path.join(dataset_path,'photos.txt'), 'rb') as f:
            reader = csv.reader(f)
            self.image_paths = [os.path.join(dataset_path, p[0]) for p in reader]
        with open(os.path.join(dataset_path,'feat/colours.txt'), 'rb') as f:
            reader = csv.reader(f)
            self.colors = [p[0] for p in reader]
        with open(os.path.join(dataset_path,'feat/singles.txt'), 'rb') as f:
            reader = csv.reader(f)
            self.singles = [p[0] for p in reader]

        self.pair_indices = range(len(self.sim_pairs_list))
        random.shuffle(self.pair_indices)
        self.pair_index = 0

        self.td = td
        self.max_tries = max_tries

        self.im_height = 384
        self.im_width = 256

    def r_metric(self, i, j):
        """
        Intersection over union of the labels.
        """
        return 1.0*sum(self.labels[i] & self.labels[j]) / sum(self.labels[i] | self.labels[j])

    def triplet(self, similar_pair):
        anchor, similar = similar_pair
        sample_idxs = np.random.choice(np.arange(self.labels.shape[0]), self.max_tries, replace=False)
        dissimilar = None
        for i in sample_idxs:
            if i != anchor and i != similar:
                similarity = self.r_metric(anchor, i)
                if similarity < self.td:
                    dissimilar = i
                    break
        if dissimilar is None:
            return None
        else:
            return (anchor, similar, dissimilar)

    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def load_images(self, image_indices):
        images = np.zeros((len(image_indices), self.im_height, self.im_width, 3))
        for i,j in enumerate(image_indices):
            img = misc.imread(self.image_paths[j])
            if img.ndim == 2:
                img = self.to_rgb(img)
            images[i,:,:,:] = img
        images_float = images.astype(np.float32)
        return images_float

    def get_triplet_batch(self, batch_size=10):
        """
        Returns a batch of images of shape (3 * batch_size, height, width, 3)
        Batch is partitioned into [anchor, similar images, dissimilar images]
        """
        j = self.pair_index
        n = len(self.pair_indices)
        if j + batch_size <= n:
            batch = [self.sim_pairs_list[i] for i in self.pair_indices[j:j+batch_size]]
        else:
            b1 = [self.sim_pairs_list[i] for i in self.pair_indices[j:n]]
            b2 = [self.sim_pairs_list[i] for i in self.pair_indices[0:n-j]]
            batch = b1 + b2
        self.pair_index += batch_size

        anchors = []
        similars = []
        dissimilars = []
        for idx1, idx2 in batch:
            similar_pair = (idx1, idx2) if random.randint(0,1) == 0 else (idx2, idx1)
            triplet_ = self.triplet(similar_pair)
            assert(triplet_ is not None)
            anchors.append(triplet_[0])
            similars.append(triplet_[1])
            dissimilars.append(triplet_[2])

        assert(len(anchors) == len(similars) and len(anchors) == len(dissimilars) )

        ax = self.load_images(anchors)
        px = self.load_images(similars)
        nx = self.load_images(dissimilars)

        images = np.vstack([ax, px, nx])
        indices = np.hstack([anchors, similars, dissimilars])
        return (images, indices)

    def tags(self, idx):
        colors = [self.colors[i] for i,x in enumerate(self.color_mat[idx]) if x>0]
        singles = [self.singles[i] for i,x in enumerate(self.single_mat[idx]) if x>0]
        return colors + singles

    def show_triplet(self, images, indices, save_path):
        """
        images is a numpy array of shape (3, im_height, im_width, 3)
        indices is a numpy array of shape (3)

        the first dimension is [anchor, similar, dissimilar]
        """
        images = images.astype('uint8')
        anchor_im = images[0]
        pos_im = images[1]
        neg_im = images[2]
        anchor_idx = indices[0]
        pos_idx = indices[1]
        neg_idx = indices[2]

        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(neg_im)
        plt.title('Dissimilar r={0:.2f}'.format(self.r_metric(anchor_idx, neg_idx)), fontsize=16)
        #plt.xlabel("colors: Heather-Gray-Boots, White-Shirt, White-Sunglasses.\nsingles: Boots, Heather Gray, Shirt, Sunglasses, White")
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        #plt.figtext(.02, .02, 



        plt.subplot(132)
        plt.imshow(anchor_im)
        plt.title('Anchor', fontsize=16)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.subplot(133)
        plt.imshow(pos_im)
        plt.title('Similar r={0:.2f}'.format(self.r_metric(anchor_idx, pos_idx)),fontsize=16)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

        comment = "Anchor : {0}\n".format(', '.join(self.tags(anchor_idx)))
        comment += "Similar : {0}\n".format(', '.join(self.tags(pos_idx)))
        comment += "Dissimilar : {0}\n".format(', '.join(self.tags(neg_idx)))

        comment = tw.fill(tw.dedent(comment.rstrip() ), width=80)
        plt.figtext(0, 0.1, comment, horizontalalignment='left',
            fontsize=12, multialignment='left')

        plt.savefig(save_path)
        plt.close(fig)





def select_triplet(batch, idx):
    """
    Suppose the batch is a matrix whose first dimension of
    shape 3*batch_size, where it is partitioned into segments
    [anchor, similar, dissimilar] segments.
    """
    assert(batch.shape[0] % 3 == 0)
    batch_size = batch.shape[0] / 3
    return batch[np.array([idx, idx+batch_size, idx+2*batch_size])]


def example():
    dataset_path = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/"
    similar_pairs_file  = "similar_pairs.pkl2"

    dataSet = DataSetClass(dataset_path, similar_pairs_file)
    batch, indices = dataSet.get_triplet_batch(1)
    bs = batch.shape
    print("batch_shape (3*batch_size, height, width, 3) : ({0},{1},{2},{3})".format(bs[0], bs[1], bs[2], bs[3]))
    print(batch[0])
    im_1 = select_triplet(batch, 0)
    idx_1 = select_triplet(indices, 0)
    pdb.set_trace()

    #dataSet.show_triplet(im_1, idx_1, "./figures/trip1.png")

if __name__ == '__main__':
    example()























