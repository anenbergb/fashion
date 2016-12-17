"""
Fashion 144k dataset input module.
The dataset can be downloaded from this page:
http://hi.cs.waseda.ac.jp/~esimo/en/research/stylenet/
"""

import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import pdb
import os, pickle, csv
import numpy as np
from scipy import misc
import threading

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import textwrap as tw
import time




class DataSetClass():
    """
    self.similar_pairs : pairs of images whose rscore is above 0.75

    only consider the self.single_mat (123) tags when training the classification network.
    """
    def __init__(self, dataset_path, similar_pairs_pkl, td=0.1, max_tries=300):
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

        """ Training, validation, testing splits """
        self.testids = np.load(os.path.join(dataset_path,'testids.npy'))
        self.trainids = np.load(os.path.join(dataset_path,'trainids.npy'))
        self.validids = np.load(os.path.join(dataset_path,'validids.npy'))

        self.sim_train_pairs, self.sim_valid_pairs, self.sim_test_pairs, self.sim_test_or_valid_pairs = self.pair_splits()
        self.sim_train_pairs_list = self.sim_train_pairs.keys()
        self.sim_valid_pairs_list = self.sim_valid_pairs.keys()
        self.sim_test_pairs_list = self.sim_test_pairs.keys()
        self.sim_test_or_valid_pairs_list = self.sim_test_or_valid_pairs.keys()



        """ load image mean and std """
        im_stats = np.load(os.path.join(dataset_path, "stats/stats80554.npz"))
        self.mean = im_stats["mean"]
        self.channel_mean = im_stats["mean_channels"]
        self.channel_std = im_stats["channel_std"]


        """ indices to track training data when sampling pair minibatches """
        self.train_pair_indices = range(len(self.sim_train_pairs_list))
        random.shuffle(self.train_pair_indices)
        self.train_pair_index = 0


        """indices to track sampling of training images """
        self.train_indices = range(len(self.trainids))
        random.shuffle(self.train_indices)
        self.train_index = 0

        """ Set other dataset properties """
        self.td = td
        self.max_tries = max_tries

        self.im_height = 384
        self.im_width = 256
        self.kclasses = self.single_mat.shape[1]

    def pair_splits(self):
        train_pairs = {}
        valid_pairs = {}
        test_pairs = {}
        test_or_valid_pairs = {}
        
        def in_split(idx):
            if idx in self.validids:
                return "valid"
            elif idx in self.testids:
                return "test"
            else:
                return "train"
        
        for k,v in self.similar_pairs.items():
            split1 = in_split(k[0])
            split2 = in_split(k[1])
            if split1 != "train" and split2 != "train":
                test_or_valid_pairs[k] = v
            if split1 == split2:
                if split1 == "train":
                    train_pairs[k] = v
                elif split1 == "test":
                    test_pairs[k] = v
                else:
                    assert(split1 == "valid")
                    valid_pairs[k] = v
        return train_pairs, valid_pairs, test_pairs, test_or_valid_pairs


    def r_metric(self, i, j):
        """
        Intersection over union of the labels.
        """
        return 1.0*sum(self.labels[i] & self.labels[j]) / sum(self.labels[i] | self.labels[j])

    def triplet(self, similar_pair):
        """
        Samples a triplet pair from the training data set.
        """
        anchor, similar = similar_pair
        sample_idxs = np.random.choice(self.trainids, self.max_tries, replace=False)
        #sample_idxs = np.random.choice(np.arange(self.labels.shape[0]), self.max_tries, replace=False)
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

    def load_images(self, image_indices, normalize=True):
        """
        Normalizes the images by subtracting the channel_mean from each channel (RGB) and
        dividing each channel (RGB) by the channel's standard deviation. 
        """
        images = np.zeros((len(image_indices), self.im_height, self.im_width, 3))
        for i,j in enumerate(image_indices):
            img = misc.imread(self.image_paths[j])
            if img.ndim == 2:
                img = self.to_rgb(img)
            elif img.ndim == 3 and img.shape[2]>3:
                img = img[:,:,0:3]
            images[i,:,:,:] = img
        images_float = images.astype(np.float32)
        if normalize:
            images_float = (images_float - self.channel_mean) / self.channel_std
        return images_float

    def get_triplet_batch(self, batch_size=10, all_labels=False):
        """
        Returns a batch of images of shape (3 * batch_size, height, width, 3)
        Batch is partitioned into [anchor, similar images, dissimilar images]
        """
        j = self.train_pair_index
        n = len(self.train_pair_indices)
        if j + batch_size <= n:
            batch = [self.sim_train_pairs_list[i] for i in self.train_pair_indices[j:j+batch_size]]
            self.train_pair_index += batch_size
        else:
            b1 = [self.sim_train_pairs_list[i] for i in self.train_pair_indices[j:n]]
            b2 = [self.sim_train_pairs_list[i] for i in self.train_pair_indices[0:batch_size - (n-j)]]
            batch = b1 + b2

        if self.train_pair_index >= n:
            #reshuffle the indices since this is a new epoch.
            random.shuffle(self.train_pair_indices)
            self.train_pair_index = 0
        
        assert(len(batch) == batch_size)

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
        if all_labels:
            labels = self.labels[indices].astype(np.float32) # [colors, singles]
        else:
            labels = self.single_mat[indices].astype(np.float32)

        return (images, labels)

    def get_image_batch(self, batch_size=10, split='train'):
        """
        generates batches of images from either the 'train', 'test', or 'eval' split.
        """
        j = self.train_index
        n = len(self.train_indices)

        if j + batch_size <= n:
            batch = [self.trainids[i] for i in self.train_indices[j:j+batch_size]]
            self.train_index += batch_size
        else:
            b1 = [self.trainids[i] for i in self.train_indices[j:n]]
            b2 = [self.trainids[i] for i in self.train_indices[0:batch_size - (n-j)]]
            batch = b1 + b2

        if self.train_index >= n:
            #reshuffle the indices since this is a new epoch.
            random.shuffle(self.train_indices)
            self.train_index = 0

        images = self.load_images(batch)
        labels = self.single_mat[np.asarray(batch)].astype(np.float32)
        return images, labels


    def triplet_iterator(self, batch_size=10, all_labels=False):
        while True:
            images, labels = self.get_triplet_batch(batch_size=batch_size, all_labels=all_labels)
            yield images, labels

    def image_iterator(self, batch_size=10, split='train'):
        while True:
            images, labels = self.get_image_batch(batch_size=batch_size, split='train')
            yield images, labels

    def sample_k_pairs(self, nrof_similar=10, nrof_dissimilar=10, split='train'):
        """
        Samples a fixed number of similar pairs and dissimilar pairs from the
        specified dataset split train/valid/test/test_or_valid

        returns two lists:
            ids = [first index | second index].

            first half of the ids correspond to the
            first id in the pair, the second half correspond to the second id in the
            pair

            labels = 0/1 where 1 is similar, 0 is dissimilar.
        """
        if split=='train':
            sim_pairs_list = self.sim_train_pairs_list
            all_ids = self.trainids
        elif split=='valid':
            sim_pairs_list = self.sim_valid_pairs_list
            all_ids = self.validids
        elif split=='test':
            sim_pairs_list = self.sim_test_pairs_list
            all_ids = self.testids
        else:
            assert(split=='test_or_valid')
            sim_pairs_list = self.sim_test_or_valid_pairs_list
            all_ids = np.hstack([self.testids, self.validids])

        if nrof_similar < len(sim_pairs_list):
            indices = np.arange(len(sim_pairs_list))
            sample_indices = np.random.choice(indices, nrof_similar, replace=False)
            sim_pairs_list = [sim_pairs_list[i] for i in sample_indices]

        # generate dissimilar pairs list
        dis_pairs_list = []
        all_ids1 = np.copy(all_ids)
        all_ids2 = np.copy(all_ids)
        random.shuffle(all_ids1)
        random.shuffle(all_ids2)
        count = 0
        for id1, id2 in zip(all_ids1, all_ids2):
            if count < nrof_dissimilar and id1 != id2:
                if self.r_metric(id1, id2) < self.td:
                    dis_pairs_list.append((id1,id2))
                    count += 1
        assert(count == nrof_dissimilar) # make sure to sample nrof_dissimilar pairs

        labels = [1]*len(sim_pairs_list)
        labels += [0]*len(dis_pairs_list)

        ids = sim_pairs_list + dis_pairs_list
        return ids, labels

    def sample_k_ids(self, k=100, split='train'):
        """
        Samples the ids of k images from the specified split.
        split is one of train/valid/test/test_or_valid
        """
        if split=='train':
            all_ids = self.trainids
        elif split=='valid':
            all_ids = self.validids
        elif split=='test':
            all_ids = self.testids
        else:
            assert(split=='test_or_valid')
            all_ids = np.hstack([self.testids, self.validids])

        if k < len(all_ids):
            indices = np.arange(len(all_ids))
            sample_indices = np.random.choice(indices, k, replace=False)
            all_ids = [all_ids[i] for i in sample_indices]

        return all_ids

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


class TripletRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of triplets.
    """
    def __init__(self, dataset, batch_size, prefix='triplet_', all_labels=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataX = tf.placeholder(dtype=tf.float32, shape=(None, dataset.im_height, dataset.im_width, 3), name=prefix+'input')
        
        if all_labels:
            klabels = dataset.labels.shape[1]
        else:
            klabels = dataset.kclasses
        self.dataY = tf.placeholder(dtype=tf.float32, shape=[None, klabels], name=prefix+'labels')
        # The actual queue of data. The queue contains a vector for
        self.queue = tf.FIFOQueue(
                        capacity=2000,
                        dtypes=[tf.float32, tf.float32],
                        shapes=[[dataset.im_height, dataset.im_width, 3], [klabels]]
        )

        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])
        self.all_labels = all_labels

    def get_inputs(self):
        """
        Return's tensors containing a batch of triplets
        """
        return self.queue.dequeue_many(3*self.batch_size)

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in self.dataset.triplet_iterator(batch_size=self.batch_size, all_labels=self.all_labels):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


class ImageRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of images sampled from the training set
    """
    def __init__(self, dataset, batch_size, prefix=''):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataX = tf.placeholder(dtype=tf.float32, shape=(None, dataset.im_height, dataset.im_width, 3), name=prefix+'input')
        self.dataY = tf.placeholder(dtype=tf.float32, shape=[None, dataset.kclasses], name=prefix+'labels')
        # The actual queue of data. The queue contains a vector for
        self.queue = tf.FIFOQueue(
                        capacity=2000,
                        dtypes=[tf.float32, tf.float32],
                        shapes=[[dataset.im_height, dataset.im_width, 3], [dataset.kclasses]]
        )

        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

    def get_inputs(self):
        """
        Return's tensors containing a batch of triplets
        """
        return self.queue.dequeue_many(self.batch_size)

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in self.dataset.image_iterator(batch_size=self.batch_size):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads



def _resize(image, height, width):
  """Resize images to the aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  new_height = tf.convert_to_tensor(height, dtype=tf.int32)
  new_width = tf.convert_to_tensor(width, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def _mean_image_subtraction(image, means, stds):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.

       images_float = (images_float - self.channel_mean) / self.channel_std
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')
  if len(stds) != num_channels:
    raise ValueError('len(stds) must match the number of channels')

  channels = tf.split(2, num_channels, image)
  for i in range(num_channels):
    channels[i] -= means[i]
    channels[i] /= stds[i]
  return tf.concat(2, channels)

def preprocess_tensor(image, height, width, channel_mean, channel_std):
    im = _resize(image, height, width)
    im.set_shape([height, width, 3])
    im = tf.to_float(im)
    im = _mean_image_subtraction(im, channel_mean, channel_std)
    return im






def show_triplet(images, labels, dataset, save_path):
    """
    images is a numpy array of shape (3, im_height, im_width, 3)
    labels is a numpy array of shape (3, 123)

    the first dimension is [anchor, similar, dissimilar]
    """
    def r_metric(label1, label2):
        """
        label1 & label 2 are np.arrays of shape (123,)
        """
        return 1.0*sum(label1 & label2) / sum(label1 | label2)

    def label_names(dataset, label):
        colors = [dataset.colors[i] for i,x in enumerate(label[:dataset.color_mat.shape[1]]) if x>0]
        singles = [dataset.singles[i] for i,x in enumerate(label[dataset.color_mat.shape[1]:]) if x>0]
        return colors + singles

    def tags(self, idx):
        singles = [self.singles[i] for i,x in enumerate(self.single_mat[idx]) if x>0]


    images = (images * dataset.channel_std) + dataset.channel_mean
    images = images.astype('uint8')
    anchor_im = images[0]
    pos_im = images[1]
    neg_im = images[2]
    

    anchor_label = labels[0]
    pos_label = labels[1]
    neg_label = labels[2]

    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(neg_im)
    plt.title('Dissimilar r={0:.2f}'.format(r_metric(anchor_label, neg_label)), fontsize=16)
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
    plt.title('Similar r={0:.2f}'.format(r_metric(anchor_label, pos_label)),fontsize=16)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    comment = "Anchor : {0}\n".format(', '.join(label_names(dataset, anchor_label)))
    comment += "Similar : {0}\n".format(', '.join(label_names(dataset, pos_label)))
    comment += "Dissimilar : {0}\n".format(', '.join(label_names(dataset, neg_label)))

    comment = tw.fill(tw.dedent(comment.rstrip() ), width=80)
    plt.figtext(0, 0.1, comment, horizontalalignment='left',
        fontsize=12, multialignment='left')

    plt.savefig(save_path)
    plt.close(fig)



def tripletRunnerTest():
    pdb.set_trace()
    def select_triplet(batch, idx):
        """
        Suppose the batch is a matrix whose first dimension of
        shape 3*batch_size, where it is partitioned into segments
        [anchor, similar, dissimilar] segments.
        """
        assert(batch.shape[0] % 3 == 0)
        batch_size = batch.shape[0] / 3
        return batch[np.array([idx, idx+batch_size, idx+2*batch_size])]

    def compute_r_metric(batch):
        """
        Suppose the batch is a matrix whose first dimension of
        shape 3*batch_size, where it is partitioned into segments
        [anchor, similar, dissimilar] segments.
        """
        assert(batch.shape[0] % 3 == 0)
        batch_size = batch.shape[0] / 3
        anchor = batch[:batch_size]
        similar = batch[batch_size:2*batch_size]
        dissimilar = batch[2*batch_size:3*batch_size]
        r_metrics = []
        for i in range(batch_size):
            sim_r_metric = 1.0*sum(anchor[i] & similar[i]) / sum(anchor[i] | similar[i])
            dis_r_metric = 1.0*sum(anchor[i] & dissimilar[i]) / sum(anchor[i] | dissimilar[i])
            r_metrics.append((sim_r_metric, dis_r_metric))
        return r_metrics

    with tf.device("/cpu:0"):
        tf.set_random_seed(669)
        print("Starting triplet Runner Test")
        data_dir = '/cvgl/u/anenberg/Fashion144k_stylenet_v1/'
        dataset = DataSetClass(data_dir, "similar_pairs.pkl2")
        print("dataset.train_pair_index %d" % dataset.train_pair_index)
        batch_size = 1
        dataset.train_pair_index = 61900
        runner = TripletRunner(dataset, batch_size, prefix='', all_labels=True)
        image_batch, label_batch = runner.get_inputs()
        print("dataset.train_pair_index %d" % dataset.train_pair_index)




        config=tf.ConfigProto( log_device_placement=True,
                allow_soft_placement=True,
                inter_op_parallelism_threads=4,
                intra_op_parallelism_threads=8)
        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        batches = 0
        save_root = '/afs/cs.stanford.edu/u/anenberg/scr/CS331B/fashion/debug/runner_12_14'
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        runner.start_threads(sess)
        print("dataset.train_pair_index %d" % dataset.train_pair_index)

        def save_image(images, labels, i):
            images = select_triplet(images, i)
            labels = select_triplet(labels, i)
            save_suffix = "triplet_{0}".format(batches*batch_size+i)
            save_path = os.path.join(save_root, save_suffix)
            show_triplet(images, labels, dataset, save_path)
            print("[Saving @ {0}".format(save_suffix))


        while True:
            start_time = time.time()
            image_batch_np, label_batch_np = sess.run([image_batch, label_batch])
            assert(image_batch_np.shape[0] == 3*batch_size and image_batch_np.shape[0] == 3*batch_size)
            label_batch_np = label_batch_np.astype(np.int)
            r_metrics = compute_r_metric(label_batch_np)
            duration = time.time() - start_time
            #if batches % 10 == 0:
            #    print("[%d][%d/%d]\tR_SIM %.3f\tR_DIS %.3f\tTime %.3f"%(batches, 0, batch_size, r_metrics[0][0], r_metrics[0][1], duration))
            for i, (rsim, rdis) in enumerate(r_metrics):
                if (batch_size*batches + i) % 10 == 0:
                    print("[%d][%d/%d][%d/%d]\tdataset[%d/%d]\tR_SIM %.3f\tR_DIS %.3f\tTime %.3f"%(batches, i, batch_size, batch_size*batches + i, len(dataset.train_pair_indices), dataset.train_pair_index, len(dataset.train_pair_indices), rsim, rdis, duration))
                if rsim < 0.75 or rdis > 0.1:
                    print("[%d][%d/%d]\tR_SIM %.3f\tR_DIS %.3f\tTime %.3f\t[WARNING] detected invalid triplet" % (batches, i, batch_size, rsim, rdis, duration))
                    save_image(image_batch_np, label_batch_np, i)

            if batches % 10000 == 0:
                save_image(image_batch_np, label_batch_np,random.randint(0,batch_size-1))
            batches += 1

        coord.request_stop()
        coord.join(threads)
        sess.close()



if __name__ == '__main__':
    print("hello world")
    tripletRunnerTest()
