
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import argparse
import os
import sys
import pdb
import numpy as np
from datetime import datetime
import pickle


import tensorflow.contrib.slim as slim
import hipsterWars_input
sys.path.append('/afs/cs.stanford.edu/u/anenberg/scr/CS331B/models/slim/')
from nets import nets_factory
from nets import inception_resnet_v2 as incep
from nets import vgg
from preprocessing import vgg_preprocessing
import math
from matplotlib import pyplot as plt

import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import fashionStyle128_model
import fashionStyle128_input

def main(args):

  subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
  save_dir = os.path.join(args.save_dir, args.model_name, subdir)
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


  if args.model_name == 'vgg_16':
    image_size = vgg.vgg_16.default_image_size #224
    logit_loader = load_vgg16
    fn_preprocess = lambda im: vgg_preprocessing.preprocess_image(im,
                                                         image_size,
                                                         image_size,
                                                         is_training=False,
                                                        resize_side_min=image_size+1)
    embedding_size = 4096
    means = (123.68, 116.78, 103.94) #RGB
    stds = None
  
  elif args.model_name == 'fashion128':

    im_stats = np.load(args.fashion144k_stats)
    means = im_stats["mean_channels"]
    stds = im_stats["channel_std"]
    image_size = (384, 256)
    logit_loader = lambda s, i: load_fashion128_v12(s,i, args.checkpoint_path)
    fn_preprocess = lambda im: fashionStyle128_input.preprocess_tensor(
                                                  im,
                                                  image_size[0],
                                                  image_size[1],
                                                  means,
                                                  stds)
    embedding_size = 128


  dataset = hipsterWars_input.DataSetClass(args.dataset_dir)
  image_batch, label_batch  = hipsterWars_input.preprocessBatchRunner(dataset,
                                                        fn_preprocess,
                                                        batch_size=args.batch_size,
                                                        num_threads=args.num_threads)
  
  config=tf.ConfigProto( #log_device_placement=True,
                     allow_soft_placement=True,
                     intra_op_parallelism_threads=4)
  sess = tf.Session(config=config)
  logits = logit_loader(sess, image_batch)
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  nrof_images = dataset.nrof_images
  nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
  emb_array = np.zeros((nrof_images, embedding_size))
  label_array = np.zeros(nrof_images)
  for i in range(nrof_batches):
    start_index = i*args.batch_size
    end_index = min((i+1)*args.batch_size, nrof_images)
    print("[{0} - {1}]".format(start_index, end_index))
    np_image_batch, np_label_batch, logits_out = sess.run([image_batch, label_batch, logits])
    real_batch_sz = end_index - start_index
    emb_array[start_index:end_index,:] = logits_out[:real_batch_sz]
    label_array[start_index:end_index] = np_label_batch[:real_batch_sz]
    if i % 10 == 0:
      show_images(np_image_batch, means=means, stds=stds, save_path=os.path.join(save_dir, "batch{0}".format(i)))
  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()


  train_svm_classifer(emb_array, label_array, save_dir, dataset.category2label.keys())
  print("Output saved to {0}".format(save_dir))



def load_vgg16(sess, input_tensor, checkpoint_path='/cvgl/u/anenberg/CS331B/pretrain/vgg_16.ckpt'):
  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(input_tensor,
                          num_classes=None,
                          is_training=False)
  saver = tf.train.Saver()
  saver.restore(sess, checkpoint_path)
  return logits

def load_fashion128(sess, input_tensor, checkpoint_path):
  model = fashionStyle128_model.Style128Net(None, input_tensor, None, 'forward_reload')
  model.build_graph()
  saver = tf.train.Saver()
  saver.restore(sess, checkpoint_path)
  return model.embeddings

def load_fashion128_v12(sess, input_tensor, checkpoint_path):
  model = fashionStyle128_model.Style128Net(None, input_tensor, None, 'forward_reload')
  model.build_graph()
  pdb.set_trace()
  saver = tf.train.import_meta_graph(checkpoint_path+".meta")
  saver.restore(sess, checkpoint_path)

  return model.embeddings

  





def show_images(batch, means=None, stds=None, save_path='/afs/cs.stanford.edu/u/anenberg/scr/CS331B/fashion/debug/out.jpg'):
    """
    Show 25 images from the batch.
    Need to add back the means to the corresponding channels. RGB
    """
    if stds is not None:
      batch[:,:,:,0] *= stds[0]
      batch[:,:,:,1] *= stds[1]
      batch[:,:,:,2] *= stds[2]

    if means is not None:
      batch[:,:,:,0] += means[0]
      batch[:,:,:,1] += means[1]
      batch[:,:,:,2] += means[2]

    batch_split_ = np.split(batch, batch.shape[0], axis=0)
    batch_split = [np.squeeze(x,axis=0) for x in batch_split_]
    def make_grid(images,height,width):
      """
      Assume all images are of the same shape.
      """
      assert(len(images) >= height*width)
      shape = images[0].shape
      grid = np.zeros((height*shape[0], width*shape[1],shape[2]))
      for i in range(height):
          for j in range(width):
              grid[i*shape[0]:(i+1)*shape[0],j*shape[1]:(j+1)*shape[1],:] = images[i*width + j]
      return grid
    grid = make_grid(batch_split,5,5)
    fig = plt.figure()
    plt.imshow(grid.astype(np.uint8))
    plt.savefig(save_path)




def train_svm_classifer(features, labels, save_dir, label_names):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
 
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.1)
 
    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]
  
    #param = [{"kernel": ["linear"], "C": [1]}]
    # request probability estimation
    # loss defaults to squared_hinge, penalty defaults to l2
    # multiclass = ovr (one verse rest)
    #svm = LinearSVC()
    svm = SVC(probability=True)
 
    # 5-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
            cv=5, n_jobs=5, verbose=2)
 
    clf.fit(X_train, y_train)
 
    with open(os.path.join(save_dir, 'best_svm.pkl'), 'wb') as f:
      pickle.dump(clf.best_estimator_, f)

    f1=open(os.path.join(save_dir, 'train_svm.txt'), 'w+')

    print("\nBest parameters set:")
    f1.write("\nBest parameters set:\n")
    print(clf.best_params_)
    f1.write(str(clf.best_params_)+"\n")
    y_predict=clf.predict(X_test)
 
    labels=sorted(list(set(labels)))

    print("\nConfusion matrix:")
    f1.write("\nConfusion matrix:\n")
    l = "Labels: {0}".format(",".join([str(x) for x in labels]))
    print(l)
    f1.write(l+"\n")
    l = "Label names: {0}\n".format(",".join(label_names))
    print(l)
    f1.write(l+"\n")
    l = confusion_matrix(y_test, y_predict, labels=labels)
    print(l)
    f1.write(str(l)+"\n")
    l = "\nClassification report:"
    print(l)
    f1.write(l+"\n")
    l = classification_report(y_test, y_predict)
    print(l)
    f1.write(l+"\n")
    l = "Accuracy score: {0}\n".format(accuracy_score(y_test, y_predict))
    print(l)
    f1.write(l+"\n")




"""
model names:
resnet_v1_101
/cvgl/u/anenberg/CS331B/pretrain/resnet_v1_101.ckpt
"""

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
      
    parser.add_argument('--dataset_dir', type=str,
      help='Path to the data directory containing the hipster wars dataset.',
      default='/cvgl/u/anenberg/hipsterwars_v1.0')
    parser.add_argument('--model_name', type=str,
      help='Name of the pretrained model.',
      default='vgg_16')


    parser.add_argument('--batch_size', type=int, 
      help='Number of images in a batch to process with pretrained network.',
      default=50)
    parser.add_argument('--num_threads', type=int, 
      help='Number of threads to run the input queue runner.',
      default=4)
    parser.add_argument('--save_dir', type=str, 
      help='Directory to save the svm trained on the hipster wars dataset', 
      default='/afs/cs.stanford.edu/u/anenberg/scr/CS331B/fashion/results/hipsterWars')

    parser.add_argument('--checkpoint_path', type=str,
      help='Path to the checkpoint file to restore weights from',
      default='/cvgl/u/anenberg/CS331B/models')

    parser.add_argument('--fashion144k_stats', type=str,
      help='Path to the npz file of the stats (mean, std) of fashion144k_stylnet.',
      default='/cvgl/u/anenberg/Fashion144k_stylenet_v1/stats/stats80554.npz')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




