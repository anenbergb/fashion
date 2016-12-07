from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse, os, pickle
import numpy as np
import fashionStyle128_model
import fashionStyle128_input
import fashionStyle128
import tensorflow as tf
from datetime import datetime
import random
import pdb



IM_HEIGHT = 384
IM_WIDTH = 256
KTHREADS = 8
val_train_nrof_similar = 100
val_train_nrof_dissimilar = 100
val_test_nrof_similar = 100
val_test_nrof_dissimilar = 100



def train(args, hps):
  """Creating output directories"""
  subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
  #save event logs (Tensorboard) to log_dir
  log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
  if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
      os.makedirs(log_dir)
  #save checkpoints in model_dir
  model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
  if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
      os.makedirs(model_dir)

  np.random.seed(seed=args.seed)
  print('Model directory: %s' % model_dir)
  print('Log directory: %s' % log_dir)

  """Creating dataset and training queue runner"""
  dataset = fashionStyle128_input.DataSetClass(args.data_dir, "similar_pairs.pkl2")

  tf.set_random_seed(args.seed)

  """Pretrain the feature extraction network by training classification task"""
  pretrain(args, hps, dataset, log_dir, model_dir)


  """ Create subdirectories to store the training log files and models """
  log_dir = os.path.join(log_dir, "train")
  model_dir = os.path.join(model_dir, "train")
  if not os.path.isdir(log_dir):
      os.makedirs(log_dir)
  if not os.path.isdir(model_dir):
      os.makedirs(model_dir)


  X = tf.Variable(10, name='veryimportant')

  runner = fashionStyle128_input.TripletRunner(dataset, args.batch_size, prefix='triplet_')
  image_batch, label_batch = runner.get_inputs()

  """Building the joint training model."""
  model = fashionStyle128_model.Style128Net(hps, image_batch, label_batch, 'joint')
  model.build_graph()

  """ Build evaluation (embedding) model """
  eval_images_placeholder = tf.placeholder(tf.float32, shape=(None, IM_HEIGHT, IM_WIDTH, 3), name='input')
  eval_model = fashionStyle128_model.Style128Net(hps, eval_images_placeholder, None, 'embedding')
  eval_model.build_graph()



  sv = tf.train.Supervisor(logdir=model_dir,
                           is_chief=True,
                           summary_op=None,
                           save_summaries_secs=60,
                           save_model_secs=500,
                           global_step=model.global_step)
  config=tf.ConfigProto( #log_device_placement=True,
                       allow_soft_placement=True,
                       intra_op_parallelism_threads=KTHREADS)
  sess = sv.prepare_or_wait_for_session(config=config)
  summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)


  tf.train.start_queue_runners(sess=sess) #not sure
  runner.start_threads(sess)

  print("Beginning joint training...")
  epoch = 0
  while not sv.should_stop() and epoch < args.max_nrof_epochs:
    step = train_one_epoch(args, sess, model, dataset, epoch, summary_writer)

    """ Validate the model """
    v = validate_step(args, sess, eval_model, dataset)
    tpr_train = v[0]
    fpr_train = v[1]
    acc_train = v[2]
    tpr_test = v[3]
    fpr_test = v[4]
    acc_test = v[5]
    val_train = v[6]
    far_train = v[7]
    val_test = v[8]
    far_test = v[9]
    print('[Train] Accuracy: %1.3f, TPR: %1.3f, FPR: %1.3f' % (acc_train, tpr_train, fpr_train))
    print('[Test] Accuracy: %1.3f, TPR: %1.3f, FPR: %1.3f' % (acc_test, tpr_test, fpr_test))
    print('[Train] Validation rate: %2.5f @ FAR=%2.5f' % (val_train, far_train))
    print('[Test] Validation rate: %2.5f @ FAR=%2.5f' % (val_test, far_test))
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    summary.value.add(tag='validation/train/accuracy', simple_value=acc_train)
    summary.value.add(tag='validation/train/tp_rate', simple_value=tpr_train)
    summary.value.add(tag='validation/train/fp_rate', simple_value=fpr_train)
    summary.value.add(tag='validation/train/val_rate', simple_value=val_train)
    summary.value.add(tag='validation/train/far_rate', simple_value=far_train)

    summary.value.add(tag='validation/test/accuracy', simple_value=acc_test)
    summary.value.add(tag='validation/test/tp_rate', simple_value=tpr_test)
    summary.value.add(tag='validation/test/fp_rate', simple_value=fpr_test)
    summary.value.add(tag='validation/test/val_rate', simple_value=val_test)
    summary.value.add(tag='validation/test/far_rate', simple_value=far_test)
    summary_writer.add_summary(summary, step)

    step = sess.run(model.global_step, feed_dict=None)
    epoch = step // args.epoch_size

  sv.Stop()

def pretrain(args, hps, dataset, log_dir_root, model_dir_root):

  log_dir = os.path.join(log_dir_root, "pretrain")
  if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
      os.makedirs(log_dir)
  model_dir = os.path.join(model_dir_root, "pretrain")
  if not os.path.isdir(model_dir):  # Create the log directory if it doesn't exist
      os.makedirs(model_dir) 


  runner = fashionStyle128_input.ImageRunner(dataset, args.pretrain_batch_size, prefix='pretrain_')
  image_batch, label_batch = runner.get_inputs()
  model = fashionStyle128_model.Style128Net(hps, image_batch, label_batch, 'classification')
  model.build_graph()

  pdb.set_trace()

  sv = tf.train.Supervisor(logdir=model_dir, #directory to save model to.
                           is_chief=True,
                           summary_op=None,
                           save_summaries_secs=60,
                           save_model_secs=500,
                           global_step=model.global_step)
  config=tf.ConfigProto( #log_device_placement=True,
                       allow_soft_placement=True,
                       intra_op_parallelism_threads=KTHREADS)
  sess = sv.prepare_or_wait_for_session(config=config)
  summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)
  tf.train.start_queue_runners(sess=sess)
  runner.start_threads(sess)

  print("Beginning pretraining...")
  epoch = 0
  while not sv.should_stop() and epoch < args.nrof_pretrain_epochs:
    step = train_one_epoch(args, sess, model, dataset, epoch, summary_writer)
    step = sess.run(model.global_step, feed_dict=None)
    epoch = step // args.epoch_size

  sv.Stop()







def train_one_epoch(args, sess, model, dataset, epoch, summary_writer):
  batch_number = 0
  while batch_number < args.epoch_size:
    start_time = time.time()
    feed_dict = {model.learning_rate_placeholder: args.learning_rate}
    (_, summaries, loss, step) = sess.run(
      [model.train_op, model.summaries, model.loss,
      model.global_step],
      feed_dict=feed_dict)
    duration = time.time() - start_time
    if step % 10 == 0:
      summary_writer.add_summary(summaries, step)
      tf.logging.info('loss: %.3f\n' % (loss))
      summary_writer.flush()
    print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
          (epoch, batch_number+1, args.epoch_size, duration, loss))
    
    batch_number += 1
  return step

def validate_step(args, sess, model, dataset):
  """
  Run validation on the training and validation similar pairs.
  """
  batch_size = 3*args.batch_size #batch size is the number of triples per batch.


  train_pairs, train_actual_issimilar = dataset.sample_k_pairs(nrof_similar=args.val_train_nrof_similar, nrof_dissimilar=args.val_train_nrof_dissimilar, split='train')
  train_embedding1, train_embedding2 = fashionStyle128.evaluate_embedding(train_pairs, sess, model, dataset, batch_size)
  train_dist = fashionStyle128.compute_embedding_dist(train_embedding1, train_embedding2)

  test_pairs, test_actual_issimilar = dataset.sample_k_pairs(nrof_similar=args.val_test_nrof_similar, nrof_dissimilar=args.val_test_nrof_dissimilar, split='test_or_valid')
  test_embedding1, test_embedding2 = fashionStyle128.evaluate_embedding(test_pairs, sess, model, dataset, batch_size)
  test_dist = fashionStyle128.compute_embedding_dist(test_embedding1, test_embedding2)

  """ Calculate evaluation metrics """
  thresholds = np.arange(0, 4, 0.01)
  train_roc, test_roc = fashionStyle128.calculate_roc(
    thresholds,
    train_dist,
    test_dist,
    np.asarray(train_actual_issimilar),
    np.asarray(test_actual_issimilar))

  thresholds = np.arange(0, 4, 0.001)
  far_target = args.far_target
  train_val_far, test_val_far = fashionStyle128.calculate_val(
    thresholds,
    train_dist,
    test_dist,
    np.asarray(train_actual_issimilar),
    np.asarray(test_actual_issimilar),
    far_target)


  tpr_train, fpr_train, acc_train = train_roc
  tpr_test, fpr_test, acc_test = test_roc
  val_train, far_train = train_val_far
  val_test, far_test = test_val_far

  return tpr_train, fpr_train, acc_train, tpr_test, fpr_test, acc_test, val_train, far_train, val_test, far_test
  






def evaluate():
    print("[TODO] implement evaluate.")


def main(args):
  if args.gpu == -1:
    dev = '/cpu:0'
  elif args.gpu == 0:
    dev = '/gpu:0'
  elif args.gpu == 1:
    dev = '/gpu:1'
  elif args.gpu == 2:
    dev = '/gpu:2'
  elif args.gpu == 3:
    dev = '/gpu:3'
  else:
    raise ValueError('Only supports gpus 0, 1, 2, or 3')

  hps = fashionStyle128_model.HParams(batch_size=args.batch_size,
                             epoch_size=args.epoch_size,
                             loss=args.loss,
                             alpha=args.alpha,
                             optimizer=args.optimizer,
                             learning_rate=args.learning_rate,
                             learning_rate_decay_epochs=args.learning_rate_decay_epochs,
                             learning_rate_decay_factor=args.learning_rate_decay_factor,
                             moving_average_decay=args.moving_average_decay)

  with tf.device(dev):
    if args.mode == 'train':
      train(args, hps)
    elif args.mode == 'eval':
      evaluate(args, hps)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
    
  parser.add_argument('--mode', type=str, choices=['train', 'eval'], 
    help='train or eval.', default='train')

  parser.add_argument('--logs_base_dir', type=str, 
    help='Directory where to write event logs.',
    default='/cvgl/u/anenberg/CS331B/logs')
  parser.add_argument('--models_base_dir', type=str,
    help='Directory where to write trained models and checkpoints.',
    default='/cvgl/u/anenberg/CS331B/models')

  parser.add_argument('--data_dir', type=str,
    help='Path to the data directory containing meta-data and photos subfolder.',
    default='/cvgl/u/anenberg/Fashion144k_stylenet_v1/')

  parser.add_argument('--train_dir', type=str,
    help='Directory to keep training outputs.',
    default='/cvgl/u/anenberg/CS331B/run1/train')
  parser.add_argument('--eval_dir', type=str,
    help='Directory to keep eval outputs.',
    default='/cvgl/u/anenberg/CS331B/run1/eval')

  parser.add_argument('--loss', type=str, choices=['RANKING', 'TRIPLET'],
        help='The loss function to use.', default='RANKING')
  parser.add_argument('--alpha', type=float,
      help='Positive to negative triplet distance margin.', default=0.2)



  parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
  parser.add_argument('--epoch_size', type=int,
      help='Number of batches per epoch.', default=1000)
  parser.add_argument('--batch_size', type=int,
      help='Number of images to process in a batch.', default=10)
  parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM', 'SGD'],
      help='The optimization algorithm to use', default='ADADELTA')
  parser.add_argument('--learning_rate', type=float,
      help='Initial learning rate. If set to a negative value a learning rate ' +
      'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.001)
  parser.add_argument('--learning_rate_decay_epochs', type=int,
      help='Number of epochs between learning rate decay.', default=100)
  parser.add_argument('--learning_rate_decay_factor', type=float,
      help='Learning rate decay factor.', default=1.0)
  parser.add_argument('--moving_average_decay', type=float,
      help='Exponential decay for tracking of training parameters.', default=0.9999)
  parser.add_argument('--seed', type=int,
      help='Random seed.', default=666)
  parser.add_argument('--gpu', type=int,
      help='Id of the single gpu to run on.', default=-1)

  parser.add_argument('--trainval_size', type=int,
      help='Number of similar pairs sampled from the training dataset to validate on.', default=100)
  parser.add_argument('--far_target', type=float,
      help='The target false accept rate used to set the threshold for validation experiment.',
      default=1e-3)
  parser.add_argument('--val_train_nrof_similar', type=float,
      help='The number of similar pairs from the training dataset to use to tune ' +
      'the similarity threshold during validation.',
      default=200)
  parser.add_argument('--val_train_nrof_dissimilar', type=float,
      help='The number of dissimilar pairs from the training dataset to use to tune ' +
      'the similarity threshold during validation.',
      default=200)
  parser.add_argument('--val_test_nrof_similar', type=float,
      help='The number of similar pairs from the test dataset to evaluate on.',
      default=100)
  parser.add_argument('--val_test_nrof_dissimilar', type=float,
      help='The number of dissimilar pairs from the test dataset to evaluate on.',
      default=100)

  parser.add_argument('--nrof_pretrain_epochs', type=int, 
    help='Number of epochs to train the classification network for ' +
    'before joint training', default=500)
  parser.add_argument('--pretrain_batch_size', type=int,
      help='Number of images to process in a batch when pretraining the ' +
      'feature extraction network on the classification task.' , default=30)


  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))