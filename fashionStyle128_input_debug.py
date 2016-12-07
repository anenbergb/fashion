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

import fashionStyle128_input

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

    dataSet = fashionStyle128_input.DataSetClass(dataset_path, similar_pairs_file)
    batch, indices = dataSet.get_triplet_batch(1)
    bs = batch.shape
    print("batch_shape (3*batch_size, height, width, 3) : ({0},{1},{2},{3})".format(bs[0], bs[1], bs[2], bs[3]))
    print(batch[0])
    im_1 = select_triplet(batch, 0)
    idx_1 = select_triplet(indices, 0)
    pdb.set_trace()

    #dataSet.show_triplet(im_1, idx_1, "./figures/trip1.png")

def example2():
    dataset_path = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/"
    similar_pairs_file  = "similar_pairs.pkl2"
    dataSet = fashionStyle128_input.DataSetClass(dataset_path, similar_pairs_file)
    with tf.device("/cpu:0"):
        runner = fashionStyle128_input.TripletRunner(dataSet, 3)
        batch = runner.get_inputs()


    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    runner.start_threads(sess)

    while True:
        pdb.set_trace()
        batch_ = sess.run([batch])
        print(batch_[0].shape)

"""Example 3"""
class miniNet(object):
    def __init__(self, images):
        self.v
        self.images = 


def example3():
    dataset_path = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/"
    similar_pairs_file  = "similar_pairs.pkl2"
    dataSet = fashionStyle128_input.DataSetClass(dataset_path, similar_pairs_file)
    with tf.device("/cpu:0"):
        runner = fashionStyle128_input.TripletRunner(dataSet, 3)
        batch = runner.get_inputs()


    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    runner.start_threads(sess)

    while True:
        pdb.set_trace()
        batch_ = sess.run([batch])
        print(batch_[0].shape)




if __name__ == '__main__':


