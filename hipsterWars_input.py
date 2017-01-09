"""
HipsterWars dataset at
http://www.cs.unc.edu/~hadi/hipsterwars/
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

from collections import defaultdict

class DataSetClass():
    """
    HipsterWars dataset 

    5 classes.
    """
    def __init__(self, dataset_path, stats_file='/cvgl/u/anenberg/Fashion144k_stylenet_v1/stats/stats80554.npz'):
        self.dataset_path = dataset_path

        category = []
        image_id = []
        style_skill = []
        with open(os.path.join(dataset_path, "skills.txt"), 'rb') as f:
            reader = csv.reader(f)
            for l in reader:
                category.append(l[0])
                image_id.append(l[1])
                style_skill.append(float(l[2]))

        category2label = {'Hipster':0, 'Goth':1, 'Preppy':2, 'Pinup':3, 'Bohemian':4}
        image_path = [os.path.join(dataset_path,'images',id)+".jpg" for id in image_id]
        category_id = [category2label[cat] for cat in category]


        im_stats = np.load(stats_file)
        self.mean = im_stats["mean"]
        self.channel_mean = im_stats["mean_channels"]
        self.channel_std = im_stats["channel_std"]

        self.im_height = 600
        self.im_width = 400

        self.num_classes = len(category2label)
        self.nrof_images = len(image_path)

        self.category = category
        self.image_id = image_id
        self.style_skill = style_skill
        self.category2label = category2label
        self.label2category = ['Hipster', 'Goth', 'Preppy', 'Pinup', 'Bohemian']
        self.image_path = image_path
        self.category_id = np.array(category_id)

    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def imread(self, i, rescale=None, normalize=True, dtype=np.float32):
        """
        Loads the i-th image in the hipster wars dataset and returns the label.
        Rescales the image to the same dimensions as 
        """
        assert(i < len(self.image_id))
        im = misc.imread(self.image_path[i]).astype(dtype)
        if rescale is not None:
            im = misc.imresize(im, rescale)
        if normalize:
            im = (im - self.channel_mean) / self.channel_std
        return im, self.category_id[i]

    def load_images(self, image_indices, normalize=True, rescale=None):
        """
        Normalizes the images by subtracting the channel_mean from each channel (RGB) and
        dividing each channel (RGB) by the channel's standard deviation. 
        """
        if rescale is None:
            images = np.zeros((len(image_indices), self.im_height, self.im_width, 3))
        else:
            images = np.zeros((len(image_indices), rescale[0], rescale[1], 3))
        for i,j in enumerate(image_indices):
            img = misc.imread(self.image_path[j])
            if img.ndim == 2:
                img = self.to_rgb(img)
            elif img.ndim == 3 and img.shape[2]>3:
                img = img[:,:,0:3]
            if rescale is not None:
                img = misc.imresize(img, rescale)
            images[i,:,:,:] = img
        images_float = images.astype(np.float32)
        if normalize:
            images_float = (images_float - self.channel_mean) / self.channel_std

        return images_float, np.asarray(self.category_id[np.array(image_indices)])

    def sample_k(self, k=100):
        """
        Samples k images from the dataset. Samples an equal number of images from
        the categories.
        """
        category_id_dict = defaultdict(list)
        for idx, category_id in enumerate(self.category_id):
            category_id_dict[category_id].append(idx)
        assert(len(category_id_dict) == len(self.category2label))

        nrof_sampled = 0
        count = 0
        sample_ids = []
        for category_id, idxs in category_id_dict.iteritems():
            nrof_images_per_class = (k - nrof_sampled) / (len(category_id_dict) - count)
            nrof_images_remainder = (k - nrof_sampled) % (len(category_id_dict) - count)
            if nrof_images_remainder > 0:
                nrof_images_per_class += 1

            indices = np.arange(len(idxs))
            if nrof_images_per_class < len(idxs):
                indices = np.random.choice(indices, nrof_images_per_class, replace=False)
            sample_ids += [idxs[i] for i in indices]
            nrof_sampled += nrof_images_per_class
            count += 1

        return sample_ids



    # def images_to_sprite(self, image_indices):
    #     """Creates the sprite image along with any necessary padding

    #     Args:
    #       data: NxHxW[x3] tensor containing the images.

    #     Returns:
    #       data: Properly shaped HxWx3 image with any necessary padding.
    #     """
    #     if len(data.shape) == 3:
    #         data = np.tile(data[...,np.newaxis], (1,1,1,3))
    #     data = data.astype(np.float32)
    #     min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    #     data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    #     max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    #     data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    #     # Inverting the colors seems to look better for MNIST
    #     data = 1 - data

    #     n = int(np.ceil(np.sqrt(data.shape[0])))
    #     padding = ((0, n ** 2 - data.shape[0]), (0, 0),
    #             (0, 0)) + ((0, 0),) * (data.ndim - 3)
    #     data = np.pad(data, padding, mode='constant',
    #             constant_values=0)
    #     # Tile the individual thumbnails into an image.
    #     data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
    #             + tuple(range(4, data.ndim + 1)))
    #     data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #     data = (data * 255).astype(np.uint8)
    #     return data


    def images_to_sprite(self, image_indices, h=150, w=100):
        nrof_images = len(image_indices)
        row_count = int(np.ceil(np.sqrt(nrof_images)))
        #pdb.set_trace()
        G = np.zeros((row_count*h, row_count*w,3), dtype='uint8')
        for i, im_idx in enumerate(image_indices):
            #location
            a = i / row_count
            b = i % row_count
            im, _ = self.imread(im_idx, rescale=(h,w,3), normalize=False, dtype=np.uint8)
            G[a*h:(a+1)*h, b*w:(b+1)*w, :] = im
        return G




def preprocessBatchRunner(dataset, fn_preprocess, batch_size=50, num_threads=4):
    """
    Returns a queue runner that returns batches of images drawn from the dataset
    and preprocessed by the fn_preprocess.
    """
    # convert string into tensors
    all_filepaths = tf.convert_to_tensor(dataset.image_path, dtype=dtypes.string)
    all_labels = ops.convert_to_tensor(dataset.category_id, dtype=dtypes.int32)
    #all_labels = ops.convert_to_tensor(np.arange(dataset.nrof_images), dtype=dtypes.int32)

    all_filepath_queue  = tf.train.slice_input_producer([all_filepaths, all_labels],
                                                        #num_epochs=1,
                                                        shuffle=False)
    file_content = tf.read_file(all_filepath_queue[0])
    image = tf.image.decode_jpeg(file_content, channels=3)
    label = all_filepath_queue[1]
    image_preprocessed = fn_preprocess(image)
    # The image could have been resized, scaled, or cropped
    # collect batches of images
    image_batch, label_batch  = tf.train.batch([image_preprocessed, label],
                                batch_size=batch_size,
                                num_threads=num_threads,
                                allow_smaller_final_batch=True
                                )
    return image_batch, label_batch




  



if __name__ == '__main__':
    ds = DataSetClass('/cvgl/u/anenberg/hipsterwars_v1.0')
    im = ds.imread(0, rescale=True)
    pdb.set_trace()






















