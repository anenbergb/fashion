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

from tensorflow.contrib.tensorboard.plugins import projector
import hipsterWars_input
from scipy import misc


IM_HEIGHT = 384
IM_WIDTH = 256
KTHREADS = 8
val_train_nrof_similar = 100
val_train_nrof_dissimilar = 100
val_test_nrof_similar = 100
val_test_nrof_dissimilar = 100



def train(args, hps):


  if len(args.checkpoint) > 0:
    print("Restoring model from checkpoint at %s.\nSkipping pretraining"%(args.checkpoint))
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), args.checkpoint)
    assert(os.path.isdir(model_dir))

  else:
    """Creating output directories"""
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    #save checkpoints in model_dir
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

  np.random.seed(seed=args.seed)
  print('Model directory: %s' % model_dir)

  """Creating dataset """
  dataset = fashionStyle128_input.DataSetClass(args.data_dir, args.similar_pairs_pkl, stats_file=args.dataset_stats_file)

  """Creating Hipster Wars dataset """
  dataset_hipster = hipsterWars_input.DataSetClass(args.data_dir_hipster, stats_file=args.dataset_stats_file)
  hipster_embeddings_var = tf.Variable(
    initial_value=np.zeros((args.nrof_images_hipster, 128)),
    validate_shape=False,
    dtype=tf.float32,
    trainable=False,
    name='embedding_hipster')
  hipster_embedding_placeholder = tf.placeholder(tf.float32, shape=(args.nrof_images_hipster, 128))
  hipster_embed_assign_op = hipster_embeddings_var.assign(hipster_embedding_placeholder)
  hipster_saver = tf.train.Saver([hipster_embeddings_var])
  hipster_output_path = os.path.join(model_dir, 'embed')
  os.makedirs(hipster_output_path)



  tf.set_random_seed(args.seed)
  learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
  eval_images_placeholder = tf.placeholder(tf.float32, shape=(None, IM_HEIGHT, IM_WIDTH, 3), name='eval_input')



  if len(args.checkpoint) == 0: #Train from scratch.
    """ Create pretraining model and queue runner """
    runner_pretrain = fashionStyle128_input.ImageRunner(dataset, args.pretrain_batch_size, prefix='pretrain_')
    image_batch_pretrain, label_batch_pretrain = runner_pretrain.get_inputs()
    model_pretrain = fashionStyle128_model.Style128Net(hps, image_batch_pretrain, label_batch_pretrain, 'pretrain', learning_rate_placeholder)
    model_pretrain.build_graph()
    eval_model_pretrain = fashionStyle128_model.Style128Net(hps, eval_images_placeholder, None, 'pretrain_forward')
    eval_model_pretrain.build_graph()


  """Create joint training model and queue runner """
  runner_joint = fashionStyle128_input.TripletRunner(dataset, args.batch_size, prefix='triplet_')
  image_batch_joint, label_batch_joint = runner_joint.get_inputs()
  model = fashionStyle128_model.Style128Net(hps, image_batch_joint, label_batch_joint, 'joint', learning_rate_placeholder)
  model.build_graph(restore_checkpoint = len(args.checkpoint) > 0)
  # restore from checkpoint = True will not share weights with pretrain model since pretrain model doesn't exist.

  """ Build evaluation (forward prop) models for pretraining and joint training"""
  eval_model = fashionStyle128_model.Style128Net(hps, eval_images_placeholder, None, 'joint_forward')
  eval_model.build_graph()






  saver = tf.train.Saver()
  global_step = tf.Variable(0, name='global_step', trainable=False)
  inc_global_step_op = tf.assign_add(global_step, 1)

  sv = tf.train.Supervisor(logdir=model_dir, #checkpoint model and save log events for tensorboard.
                           is_chief=True,
                           summary_op=None,
                           save_model_secs=200,
                           global_step=global_step,
                           saver=saver)
  config=tf.ConfigProto( #log_device_placement=True,
                       allow_soft_placement=True,
                       intra_op_parallelism_threads=KTHREADS)
  sess = sv.prepare_or_wait_for_session(config=config)

  """ Hipster Wars evaluation summary writer"""
  hipster_summary_writer = tf.summary.FileWriter(hipster_output_path, sess.graph)



  tf.train.start_queue_runners(sess=sess) #not sure
  runner_joint.start_threads(sess)
  epoch_pretrain = args.nrof_pretrain_epochs
  epoch_joint = 0

  if len(args.checkpoint) == 0:
    runner_pretrain.start_threads(sess)
    epoch_pretrain = 0



  while not sv.should_stop() and epoch_pretrain + epoch_joint < args.max_nrof_epochs + args.nrof_pretrain_epochs:
    if epoch_pretrain < args.nrof_pretrain_epochs:
      if epoch_pretrain == 0:
        print("Beginning pretraining...")
      step, step_cumulative = train_one_epoch(sv, args, sess, model_pretrain, dataset, epoch_pretrain, inc_global_step_op, epoch_size= args.epoch_size_pretrain, prefix='Pretrain ')
      
      print("Validation (embedding):")
      validate_embedding_step(sv, args, sess, eval_model_pretrain, dataset, step, prefix='pretrain')
      print("Validation (attribute classification):")
      validate_attribute_prediction_step(sv, args, sess, eval_model_pretrain, dataset, step, prefix='pretrain')
      epoch_pretrain = step // args.epoch_size_pretrain
    else:
      if epoch_joint == 0:
        print("Beginning joint training...")

      step, step_cumulative = train_one_epoch(sv, args, sess, model, dataset, epoch_joint, inc_global_step_op, epoch_size = args.epoch_size, prefix='Joint ')

      print("Validation (embedding):")
      validate_embedding_step(sv, args, sess, eval_model, dataset, step, prefix='joint')
      print("Validation (attribute classification):")
      validate_attribute_prediction_step(sv, args, sess, eval_model, dataset, step, prefix='joint')
      print("Hipster Wars embedding calculation:")
      validate_HipsterWars_step(sv, args, sess, eval_model, dataset_hipster, step, model_dir, hipster_embeddings_var, hipster_saver, hipster_output_path, hipster_summary_writer, hipster_embedding_placeholder, hipster_embed_assign_op)
      epoch_joint = step // args.epoch_size

    print('Model directory: %s' % model_dir)

    if len(args.checkpoint) == 0: #training from scratch, so model_pretrain exists
      step_pretrain = sess.run(model_pretrain.global_step, feed_dict=None)
      step_joint = sess.run(model.global_step, feed_dict=None)
      step_cumulative = sess.run(global_step)
      assert(step_pretrain + step_joint == step_cumulative)
    

  sv.Stop()



def train_one_epoch(sv, args, sess, model, dataset, epoch, inc_global_step, epoch_size = 10, prefix=''):
  batch_number = 0
  while batch_number < epoch_size:
    start_time = time.time()
    feed_dict = {model.learning_rate_placeholder: args.learning_rate}
    (_, loss, step, step_cumulative) = sess.run(
      [model.train_op, model.loss,
      model.global_step, inc_global_step],
      feed_dict=feed_dict)
    if step % 10 == 0:
      summaries = sess.run(model.summaries, feed_dict=feed_dict)
      sv.summary_computed(sess, summaries)
    duration = time.time() - start_time


    print(prefix+'Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
          (epoch, batch_number+1, epoch_size, duration, loss))
    batch_number += 1
  return step, step_cumulative


def validate_embedding_step(sv, args, sess, model, dataset, step, prefix='train'):
  """
  Run validation to determine quality of learned embedding on the training and validation similar pairs.
  """
  batch_size = 3*args.batch_size #batch size is the number of triples per batch.


  train_pairs, train_actual_issimilar = dataset.sample_k_pairs(nrof_similar=args.val_train_nrof_similar, nrof_dissimilar=args.val_train_nrof_dissimilar, split='train')
  train_embedding1, train_embedding2 = fashionStyle128.evaluate_embedding_pairs(train_pairs, sess, model, dataset, batch_size)
  train_dist = fashionStyle128.compute_embedding_dist(train_embedding1, train_embedding2)

  test_pairs, test_actual_issimilar = dataset.sample_k_pairs(nrof_similar=args.val_test_nrof_similar, nrof_dissimilar=args.val_test_nrof_dissimilar, split='test_or_valid')
  test_embedding1, test_embedding2 = fashionStyle128.evaluate_embedding_pairs(test_pairs, sess, model, dataset, batch_size)
  test_dist = fashionStyle128.compute_embedding_dist(test_embedding1, test_embedding2)

  """ Calculate evaluation metrics """
  thresholds = np.arange(0, 20, 0.01)
  train_roc, test_roc, threshold_roc = fashionStyle128.calculate_roc(
    thresholds,
    train_dist,
    test_dist,
    np.asarray(train_actual_issimilar),
    np.asarray(test_actual_issimilar))

  thresholds = np.arange(0, 20, 0.001)
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
  sv.summary_computed(sess, summary)
  

def validate_attribute_prediction_step(sv, args, sess, model, dataset, step, prefix='train'):
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

  sv.summary_computed(sess, summary)



def validate_HipsterWars_step(sv, args, sess, model, dataset, step, model_dir, embed_var, saver, output_path, summary_writer, placeholder, assign_op):
  """
  Run the model on the Hipster Wars dataset to get an embedding vector per HipsterWar image.
  Evaluate whether images of the same style (Bohemian, goth, etc.) have small euclidean distance.


  dataset : refers to the hipster wars dataset.
  """
  batch_size = 3*args.batch_size #batch size is the number of triples per batch.
  max_nrof_images_hipster = 2916
  """
  Maximum size sprite currenty supported is 8192px X 8192px.
  Each thumbnail image is assumed to be of size 150 x 100.
  Sprite should have the same number of rows and columns with thumbnails stored in row-first order.
  Thus, the sprite will be 54 x 54 in dimension maximum which equates to 8100px x 8100px.
  """
  nrof_images = min(max_nrof_images_hipster, args.nrof_images_hipster)
  ids = dataset.sample_k(k=nrof_images)
  emb_array, labels_array = fashionStyle128.evaluate_embedding(ids, sess, model, dataset, batch_size, normalize=True, rescale=(384, 256))
  
  #emb_tensor = tf.Variable(emb_array, name='embedding_hipster', trainable=False)
  #sess.run(emb_tensor.initializer)
  #sess.run(nop, feed_dict = {placeholder : emb_array})
  #emb_tensor = tf.convert_to_tensor(emb_array, name='emb_array_convert_to_tensor')

  sess.run(assign_op, feed_dict={placeholder : emb_array})
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  embedding.tensor_name = embed_var.name


  embedding.metadata_path = os.path.join(output_path, 'labels.tsv')
  sprite_png = os.path.join(output_path, 'sprite.png')
  embedding.sprite.image_path = sprite_png
  embedding.sprite.single_image_dim.extend([args.emb_thumbnail_w, args.emb_thumbnail_h])
  projector.visualize_embeddings(summary_writer, config)
  saver.save(sess, os.path.join(output_path, 'model.ckpt'), sv.global_step)

  ## Make sprite and labels.
  #images = np.array(all_images).reshape(
  #        -1, thumbnail_size, thumbnail_size).astype(np.float32)
  #sprite = images_to_sprite(images)
  #scipy.misc.imsave(os.path.join(output_path, 'sprite.png'), sprite)
  #all_labels = np.array(all_labels).flatten()

  with open(os.path.join(output_path, 'labels.tsv'), 'w') as f:
    f.write('Category\n')
    for category_id in labels_array[:-1]:
      f.write('%s\n' % (dataset.label2category[np.int(category_id)]))
    f.write('%s' % (dataset.label2category[np.int(labels_array[-1])]))

  # Create sprite and save it to disk.
  sprite = dataset.images_to_sprite(ids, h=args.emb_thumbnail_h, w=args.emb_thumbnail_w)
  misc.imsave(sprite_png, sprite)




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

  parser.add_argument('--models_base_dir', type=str,
    help='Directory where to write trained models, checkpoints, and event logs.',
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

  parser.add_argument('--checkpoint', type=str,
    help='Path to checkpoint folder to restore model weights from.',
    default='')

  parser.add_argument('--data_dir_hipster', type=str,
    help='Path to the data directory containing the hipster wars dataset.',
    default='/cvgl/u/anenberg/hipsterwars_v1.0')
  parser.add_argument('--nrof_images_hipster', type=int,
      help='Number of images to sample from Hipster Wars dataset to ' +
      'run embeddinging on.' , default=500)
  parser.add_argument('--emb_thumbnail_w', type=int,
      help='Size of the thumbnail width to display in the Tensorboard embedding visualization', 
      default=100)
  parser.add_argument('--emb_thumbnail_h', type=int,
      help='Size of the thumbnail width to display in the Tensorboard embedding visualization', 
      default=150)


  parser.add_argument('--dataset_stats_file', type=str,
    help='Path to the .npz file containing the image means, channel means, and channel stds ' +
          'for the training dataset (fashion 144k)',
    default='/cvgl/u/anenberg/Fashion144k_stylenet_v1/stats/stats80554.npz')

  parser.add_argument('--similar_pairs_pkl', type=str,
  	help='Path to the .pkl file containing similar image pairs from the Fashion144k dataset',
  	default='/cvgl/u/anenberg/Fashion144k_stylenet_v1/similar_pairs.pkl2')



  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))