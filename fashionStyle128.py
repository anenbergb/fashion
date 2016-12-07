from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy import interpolate
import math
import six
import pdb
from scipy.special import expit




def evaluate_embedding(pairs, sess, model, dataset, batch_size):

    pairs1 = []
    pairs2 = []
    for id1, id2 in pairs:
        pairs1.append(id1)
        pairs2.append(id2)
    # first half of ids are the first id in the pair, second half are the
    # second id in the pair
    ids_unzip = pairs1 + pairs2
    nrof_images = len(ids_unzip)
    embedding_size = model.embeddings.get_shape()[1]
    #embedding_size should be 128
    emb_array = np.zeros((nrof_images, embedding_size))

    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    for i in range(nrof_batches):
      start_index = i*batch_size
      end_index = min((i+1)*batch_size, nrof_images)
      ids_batch = ids_unzip[start_index:end_index]
      images = dataset.load_images(ids_batch)
      feed_dict = {model.images:images}
      emb_array[start_index:end_index,:] = sess.run(model.embeddings, feed_dict=feed_dict)

    # embedding1 is first id in the pair, embedding2 is second id in pair.
    embeddings1 = emb_array[:int(nrof_images/2)]
    embeddings2 = emb_array[int(nrof_images/2):]
    return embeddings1, embeddings2

def compute_embedding_dist(embeddings1, embeddings2):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])    
    diff = np.subtract(embeddings1, embeddings2)
    return np.sum(np.square(diff),1)


def calculate_roc(thresholds, train_dist, test_dist, train_actual_issim, test_actual_issim):
    nrof_thresholds = len(thresholds)
    tpr_train = np.zeros((nrof_thresholds))
    fpr_train = np.zeros((nrof_thresholds))
    acc_train = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        tpr_train[threshold_idx], fpr_train[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(threshold, train_dist, train_actual_issim)
    best_threshold_index = np.argmax(acc_train)
    train_roc = (tpr_train[best_threshold_index], fpr_train[best_threshold_index], acc_train[best_threshold_index])
    test_roc = calculate_accuracy(thresholds[best_threshold_index], test_dist, test_actual_issim)
    #pdb.set_trace()
    return train_roc, test_roc

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

  
def calculate_val(thresholds, train_dist, test_dist, train_actual_issim, test_actual_issim, far_target):
    nrof_thresholds = len(thresholds)
    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(threshold, train_dist, train_actual_issim)
    if np.max(far_train)>=far_target:
        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = np.argmax(far_train)
    
    train_val_far = calculate_val_far(threshold, train_dist, train_actual_issim)
    test_val_far = calculate_val_far(threshold, test_dist, test_actual_issim)
    #pdb.set_trace()
    return train_val_far, test_val_far  


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def add_to_collections(names, value):
  """Stores `value` in the collections given by `names`.
  Note that collections are not sets, so it is possible to add a value to
  a collection several times. This function makes sure that duplicates in
  `names` are ignored, but it will not check for pre-existing membership of
  `value` in any of the collections in `names`.
  `names` can be any iterable, but if `names` is a string, it is treated as a
  single collection name.
  Args:
    names: The keys for the collections to add to. The `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collections.
  """
  # Make sure names are unique, but treat strings as a single collection name
  names = (names,) if isinstance(names, six.string_types) else set(names)
  for name in names:
    tf.add_to_collection(name, value)


def evaluate_attribute_predictions(ids, sess, model, dataset, batch_size):

    nrof_images = len(ids)
    prediction_size = model.predictions.get_shape()[1]
    #prediction_size should be 123
    assert(prediction_size==dataset.kclasses)
    prediction_array = np.zeros((nrof_images, prediction_size))

    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    for i in range(nrof_batches):
      start_index = i*batch_size
      end_index = min((i+1)*batch_size, nrof_images)
      ids_batch = ids[start_index:end_index]
      images = dataset.load_images(ids_batch)
      feed_dict = {model.images:images}
      prediction_array[start_index:end_index,:] = sess.run(model.predictions, feed_dict=feed_dict)

    labels_array = dataset.single_mat[np.asarray(ids)].astype(np.float32)
    assert(prediction_array.shape == labels_array.shape)
    return prediction_array, labels_array

def compute_scaled_predictions(raw_predictions):
  """
  raw_predictions is a (num_images, num_classes)

  Use the logistic function to rescale the
  raw predictions to between 0-1.
  """
  return expit(raw_predictions)



def binary_stats(y_true, y_pred, normalize=True, sample_weight=None):
  """
  y_true is (num_images, num_classes) 0/1 if class is present
  y_pred is (num_images, num_classes) 0/1 if class is present
      that had been previously computed using a fixed threshold.

  Computes typical multi-label classification metrics of Hamming score,
  Precision, Recall, and F1
  https://en.wikipedia.org/wiki/Multi-label_classification
  """
  hamming_list = []
  precision_list = []
  recall_list = []
  f1_list = []
  for i in range(y_true.shape[0]):
      set_true = set( np.where(y_true[i])[0] )
      set_pred = set( np.where(y_pred[i])[0] )
      intersection = len(set_true.intersection(set_pred))
      if len(set_true) == 0 and len(set_pred) == 0:
          hamming = 1
          precision = 1
          recall = 1
          f1 = 1
      elif len(set_pred) == 0 or len(set_pred) == 0:
          hamming = intersection/float( len(set_true.union(set_pred)) )
          precision = 0.0
          recall = 0.0
          f1 = 0.0
      else:
          hamming = intersection/float( len(set_true.union(set_pred)) )
          precision = intersection/float(len(set_pred))
          recall = intersection/float(len(set_true))
          if precision + recall == 0.0:
            f1 = 0.0
          else:
            f1 = 2.0*(precision*recall) / (precision + recall)

      hamming_list.append(hamming)
      precision_list.append(precision)
      recall_list.append(recall)
      f1_list.append(f1)

  return np.mean(hamming_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)


def compute_top_k_recall(y_true, y_pred, k=10):
  """
  y_true is (num_images, num_classes) 0/1 if class is present
  y_pred is (num_images, num_classes) floating point score \in range 0-1
  """
  assert(y_true.shape == y_pred.shape)
  k = min(k, y_pred.shape[1])
  out = np.zeros((k, y_pred.shape[0]))
  for i in range(y_pred.shape[0]):
    r1 = zip(y_pred[i], range(y_pred.shape[1]))
    r1_sorted = sorted(r1, key= lambda x: x[0], reverse=True)
    true_positives = [1 if y_true[i,idx] == 1 else 0 for _, idx in r1_sorted]
    true_pos_cumsum = np.cumsum(true_positives)
    for j in range(k):
      out[j,i] = true_pos_cumsum[j] / float(j + 1)

  out_mean = out.mean(axis=1)
  return out_mean








