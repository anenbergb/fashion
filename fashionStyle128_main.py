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

  """Creating dataset """
  dataset = fashionStyle128_input.DataSetClass(args.data_dir, "similar_pairs.pkl2")

  tf.set_random_seed(args.seed)
  learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

  """ Create pretraining model and queue runner """
  runner_pretrain = fashionStyle128_input.ImageRunner(dataset, args.pretrain_batch_size, prefix='pretrain_')
  image_batch_pretrain, label_batch_pretrain = runner_pretrain.get_inputs()
  model_pretrain = fashionStyle128_model.Style128Net(hps, image_batch_pretrain, label_batch_pretrain, 'pretrain', learning_rate_placeholder)
  model_pretrain.build_graph()

  """Create joint training model and queue runner """
  runner_joint = fashionStyle128_input.TripletRunner(dataset, args.batch_size, prefix='triplet_')
  image_batch_joint, label_batch_joint = runner_joint.get_inputs()
  model = fashionStyle128_model.Style128Net(hps, image_batch_joint, label_batch_joint, 'joint', learning_rate_placeholder)
  model.build_graph()

  """ Build evaluation (forward prop) models for pretraining and joint training"""
  eval_images_placeholder = tf.placeholder(tf.float32, shape=(None, IM_HEIGHT, IM_WIDTH, 3), name='eval_input')

  eval_model_pretrain = fashionStyle128_model.Style128Net(hps, eval_images_placeholder, None, 'pretrain_forward')
  eval_model_pretrain.build_graph()

  eval_model = fashionStyle128_model.Style128Net(hps, eval_images_placeholder, None, 'joint_forward')
  eval_model.build_graph()


  global_step = tf.Variable(0, name='global_step', trainable=False)
  inc_global_step_op = tf.assign_add(global_step, 1)

  sv = tf.train.Supervisor(logdir=model_dir,
                           is_chief=True,
                           summary_op=None,
                           save_summaries_secs=60,
                           save_model_secs=500,
                           global_step=global_step)
  config=tf.ConfigProto( #log_device_placement=True,
                       allow_soft_placement=True,
                       intra_op_parallelism_threads=KTHREADS)
  sess = sv.prepare_or_wait_for_session(config=config)
  summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)


  tf.train.start_queue_runners(sess=sess) #not sure
  runner_pretrain.start_threads(sess)
  runner_joint.start_threads(sess)

  epoch_pretrain = 0
  epoch_joint = 0
  while not sv.should_stop() and epoch_pretrain + epoch_joint < args.max_nrof_epochs + args.nrof_pretrain_epochs:
    if epoch_pretrain < args.nrof_pretrain_epochs:
      if epoch_pretrain == 0:
        print("Beginning pretraining...")
      step, step_cumulative = train_one_epoch(args, sess, model_pretrain, dataset, epoch_pretrain, summary_writer, inc_global_step_op, epoch_size= args.epoch_size_pretrain, prefix='Pretrain ')
      
      print("Validation (embedding):")
      validate_embedding_step(args, sess, eval_model_pretrain, dataset, summary_writer, step, prefix='pretrain')
      print("Validation (attribute classification):")
      validate_attribute_prediction_step(args, sess, eval_model_pretrain, dataset, summary_writer, step, prefix='pretrain')
      epoch_pretrain = step // args.epoch_size_pretrain
    else:
      if epoch_joint == 0:
        print("Beginning joint training...")

      step, step_cumulative = train_one_epoch(args, sess, model, dataset, epoch_joint, summary_writer, inc_global_step_op, epoch_size = args.epoch_size, prefix='Joint ')

      print("Validation (embedding):")
      validate_embedding_step(args, sess, eval_model, dataset, summary_writer, step, prefix='joint')
      print("Validation (attribute classification):")
      validate_attribute_prediction_step(args, sess, eval_model, dataset, summary_writer, step, prefix='joint')
      epoch_joint = step // args.epoch_size

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    step_pretrain = sess.run(model_pretrain.global_step, feed_dict=None)
    step_joint = sess.run(model.global_step, feed_dict=None)
    step_cumulative = sess.run(global_step)
    assert(step_pretrain + step_joint == step_cumulative)

  sv.Stop()



def train_one_epoch(args, sess, model, dataset, epoch, summary_writer, inc_global_step, epoch_size = 10, prefix=''):
  batch_number = 0
  while batch_number < epoch_size:
    start_time = time.time()
    feed_dict = {model.learning_rate_placeholder: args.learning_rate}
    (_, summaries, loss, step, step_cumulative) = sess.run(
      [model.train_op, model.summaries, model.loss,
      model.global_step, inc_global_step],
      feed_dict=feed_dict)
    duration = time.time() - start_time
    if step % 10 == 0:
      summary_writer.add_summary(summaries, step) #step_cumulative
      tf.logging.info('loss: %.3f\n' % (loss))
      summary_writer.flush()
    print(prefix+'Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
          (epoch, batch_number+1, epoch_size, duration, loss))
    batch_number += 1
  return step, step_cumulative

def validate_embedding_step(args, sess, model, dataset, summary_writer, step, prefix='train'):
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
  thresholds = np.arange(0, 10, 0.01)
  train_roc, test_roc, threshold_roc = fashionStyle128.calculate_roc(
    thresholds,
    train_dist,
    test_dist,
    np.asarray(train_actual_issimilar),
    np.asarray(test_actual_issimilar))

  thresholds = np.arange(0, 10, 0.001)
  far_target = args.far_target
  train_val_far, test_val_far, threshold_val = fashionStyle128.calculate_val(
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



  print('[Train] Accuracy: %1.3f, TPR: %1.3f, FPR: %1.3f @ threshold %1.3f' % (acc_train, tpr_train, fpr_train, threshold_roc))
  print('[Test] Accuracy: %1.3f, TPR: %1.3f, FPR: %1.3f @ threshold %1.3f' % (acc_test, tpr_test, fpr_test, threshold_roc))
  print('[Train] Validation rate: %2.5f @ FAR=%2.5f @ threshold %1.3f' % (val_train, far_train, threshold_val))
  print('[Test] Validation rate: %2.5f @ FAR=%2.5f @ threshold %1.3f' % (val_test, far_test, threshold_val))
  # Add validation loss and accuracy to summary
  summary = tf.Summary()
  summary.value.add(tag=prefix +'_validation/embedding/train/accuracy', simple_value=acc_train)
  summary.value.add(tag=prefix +'_validation/embedding/train/true_positive_rate', simple_value=tpr_train)
  summary.value.add(tag=prefix +'_validation/embedding/train/false_positive_rate', simple_value=fpr_train)
  summary.value.add(tag=prefix +'_validation/embedding/train/VAL_validation_rate', simple_value=val_train)
  summary.value.add(tag=prefix +'_validation/embedding/train/FAR_false_accept_rate', simple_value=far_train)
  summary.value.add(tag=prefix +'_validation/embedding/test/accuracy', simple_value=acc_test)
  summary.value.add(tag=prefix +'_validation/embedding/test/true_positive_rate', simple_value=tpr_test)
  summary.value.add(tag=prefix +'_validation/embedding/test/false_positive_rate', simple_value=fpr_test)
  summary.value.add(tag=prefix +'_validation/embedding/test/VAL_validation_rate', simple_value=val_test)
  summary.value.add(tag=prefix +'_validation/embedding/test/FAR_false_accept_rate', simple_value=far_test)
  summary_writer.add_summary(summary, step)
  

def validate_attribute_prediction_step(args, sess, model, dataset, summary_writer, step, prefix='train'):
  """
  Calculate the attribute prediction performance on training and validation dataset splits.
  """
  batch_size = args.pretrain_batch_size
  threshold = args.attribute_threshold

  train_ids = dataset.sample_k_ids(k=args.nrof_val_train, split='train')
  test_ids = dataset.sample_k_ids(k=args.nrof_val_test, split='test_or_valid')

  train_predictions_raw, train_labels = fashionStyle128.evaluate_attribute_predictions(train_ids, sess, model, dataset, batch_size)
  test_predictions_raw, test_labels = fashionStyle128.evaluate_attribute_predictions(test_ids, sess, model, dataset, batch_size)

  train_predictions = fashionStyle128.compute_scaled_predictions(train_predictions_raw)
  test_predictions = fashionStyle128.compute_scaled_predictions(test_predictions_raw)
  train_predictions01 = np.greater(train_predictions, threshold).astype(np.int)
  test_predictions01 = np.greater(test_predictions, threshold).astype(np.int)
  train_labels = train_labels.astype(np.int)
  test_labels = test_labels.astype(np.int)


  train_hamming_score, train_precision, train_recall, train_f1 = fashionStyle128.binary_stats(train_labels, train_predictions01)
  test_hamming_score, test_precision, test_recall, test_f1 = fashionStyle128.binary_stats(test_labels, test_predictions01)

  k = 10
  train_top_k_recall = fashionStyle128.compute_top_k_recall(train_labels, train_predictions, k=k)
  test_top_k_recall = fashionStyle128.compute_top_k_recall(test_labels, test_predictions, k=k)


  """ Print results and save to logs """
  print('[Train] Hamming score: %1.3f, Precision %1.3f, Recall %1.3f, F1 %1.3f at threshold %1.2f' % (train_hamming_score, train_precision, train_recall, train_f1, threshold))
  print('[Test] Hamming score: %1.3f, Precision %1.3f, Recall %1.3f, F1 %1.3f at threshold %1.2f' % (test_hamming_score, test_precision, test_recall, test_f1, threshold))
  print('Top k Recall. \t1\t2\t3\t4\t5\t6\t7\t8\t9\t10')
  print('[Train]\t\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f'%tuple(train_top_k_recall.tolist()[:k]))
  print('[Test]\t\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f'%tuple(test_top_k_recall.tolist()[:k]))

  summary = tf.Summary()
  summary.value.add(tag=prefix + '_validation/attribute/train/hamming_score', simple_value=train_hamming_score)
  summary.value.add(tag=prefix + '_validation/attribute/train/precision', simple_value=train_precision)
  summary.value.add(tag=prefix + '_validation/attribute/train/recall', simple_value=train_recall)
  summary.value.add(tag=prefix + '_validation/attribute/train/F1', simple_value=train_f1)
  summary.value.add(tag=prefix + '_validation/attribute/test/hamming_score', simple_value=test_hamming_score)
  summary.value.add(tag=prefix + '_validation/attribute/test/precision', simple_value=test_precision)
  summary.value.add(tag=prefix + '_validation/attribute/test/recall', simple_value=test_recall)
  summary.value.add(tag=prefix + '_validation/attribute/test/F1', simple_value=test_f1)

  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/1', simple_value=train_top_k_recall[0])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/2', simple_value=train_top_k_recall[1])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/3', simple_value=train_top_k_recall[2])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/4', simple_value=train_top_k_recall[3])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/5', simple_value=train_top_k_recall[4])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/6', simple_value=train_top_k_recall[5])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/7', simple_value=train_top_k_recall[6])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/8', simple_value=train_top_k_recall[7])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/9', simple_value=train_top_k_recall[8])
  summary.value.add(tag=prefix + '_validation/attribute/train/top_recall/10', simple_value=train_top_k_recall[9])

  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/1', simple_value=test_top_k_recall[0])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/2', simple_value=test_top_k_recall[1])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/3', simple_value=test_top_k_recall[2])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/4', simple_value=test_top_k_recall[3])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/5', simple_value=test_top_k_recall[4])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/6', simple_value=test_top_k_recall[5])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/7', simple_value=test_top_k_recall[6])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/8', simple_value=test_top_k_recall[7])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/9', simple_value=test_top_k_recall[8])
  summary.value.add(tag=prefix + '_validation/attribute/test/top_recall/10', simple_value=test_top_k_recall[9])

  summary_writer.add_summary(summary, step)



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

  if args.loss == 'RANKING':
    joint_loss_weight = args.joint_loss_weight_rank
  else:
    assert(args.loss == 'TRIPLET')
    joint_loss_weight = args.joint_loss_weight_triplet


  hps = fashionStyle128_model.HParams(batch_size=args.batch_size,
                             epoch_size=args.epoch_size,
                             loss=args.loss,
                             alpha=args.alpha,
                             optimizer=args.optimizer,
                             learning_rate=args.learning_rate,
                             learning_rate_decay_epochs=args.learning_rate_decay_epochs,
                             learning_rate_decay_factor=args.learning_rate_decay_factor,
                             moving_average_decay=args.moving_average_decay,
                             joint_loss_weight=joint_loss_weight
                             )

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


  parser.add_argument('--joint_loss_weight_rank', type=float,
      help='Parameter to weight the classification loss relative to the embedding loss.' +
      'total loss = alpha * (classification loss) + (1- alpha) * (ranking embedding loss)',
      default=0.1)
  parser.add_argument('--joint_loss_weight_triplet', type=float,
      help='Parameter to weight the classification loss relative to the embedding loss.' +
      'total loss = alpha * (classification loss) + (1- alpha) * (triplet embedding loss)',
      default=0.1)

  parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
  parser.add_argument('--epoch_size', type=int,
      help='Number of batches per epoch.', default=1000)
  parser.add_argument('--epoch_size_pretrain', type=int,
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


  parser.add_argument('--nrof_val_train', type=float,
      help='The number of images from the training dataset to evaluate on.',
      default=100)
  parser.add_argument('--nrof_val_test', type=float,
      help='The number of images from the test dataset to evaluate on.',
      default=100)
  parser.add_argument('--attribute_threshold', type=float,
      help='Threshold above which the score for an attribute is interpreted as positive.',
      default=0.6)

  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))