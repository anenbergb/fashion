4import sys
import time
import argparse, os, pickle
import numpy as np
import fashionStyle128_model
import fashionStyle128_input
import tensorflow as tf
from datetime import datetime

import pdb



IM_HEIGHT = 384
IM_WIDTH = 256




def train(args, hps):
  """Creating output directories"""
  subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
  #save event logs to log_dir
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

  """Creating dataset"""
  dataSet = fashionStyle128_input.DataSetClass(args.data_dir, "similar_pairs.pkl2")



  """Building the model."""
  # Placeholder for input images
  tf.set_random_seed(args.seed)
  images_placeholder = tf.placeholder(tf.float32, shape=(None, IM_HEIGHT, IM_WIDTH, 3), name='input')
  model = fashionStyle128_model.Style128Net(hps, images_placeholder, args.mode)
  model.build_graph()



  sv = tf.train.Supervisor(logdir=model_dir,
                           is_chief=True,
                           summary_op=None,
                           save_summaries_secs=60,
                           save_model_secs=300,
                           global_step=model.global_step)
  config=tf.ConfigProto(log_device_placement=True)
  sess = sv.prepare_or_wait_for_session(config=config)
  summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)


  tf.train.start_queue_runners(sess=sess) #not sure

  epoch = 0
  while not sv.should_stop() and epoch < args.max_nrof_epochs:
    train_one_epoch(args, sess, model, dataSet, epoch, summary_writer)
    step = sess.run(model.global_step, feed_dict=None)
    epoch = step // args.epoch_size

  sv.Stop()

def train_one_epoch(args, sess, model, dataset, epoch, summary_writer):
  batch_number = 0
  while batch_number < args.epoch_size:
    start_time = time.time()
    batch, indices = dataset.get_triplet_batch(args.batch_size)
    load_time = time.time() - start_time
    print('Loaded %d image triplets in %.2f seconds' % (batch.shape[0] / 3, load_time))
    feed_dict = {model.images_placeholder: batch, model.learning_rate_placeholder: args.learning_rate}
    (_, summaries, loss, step) = sess.run(
      [model.train_op, model.summaries, model.loss,
      model.global_step],
      feed_dict=feed_dict)

    if step % 100 == 0:
      summary_writer.add_summary(summaries, step)
      tf.logging.info('loss: %.3f\n' % (loss))
      summary_writer.flush()
    duration = time.time() - start_time
    print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
          (epoch, batch_number+1, args.epoch_size, duration, loss))
    
    batch_number += 1
  return step

def evaluate():
    print("[TODO] implement evaluate.")


def main(args):
  if args.num_gpus == 0:
    dev = '/cpu:0'
  elif args.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

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
      'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
  parser.add_argument('--learning_rate_decay_epochs', type=int,
      help='Number of epochs between learning rate decay.', default=100)
  parser.add_argument('--learning_rate_decay_factor', type=float,
      help='Learning rate decay factor.', default=1.0)
  parser.add_argument('--moving_average_decay', type=float,
      help='Exponential decay for tracking of training parameters.', default=0.9999)
  parser.add_argument('--seed', type=int,
      help='Random seed.', default=666)
  parser.add_argument('--num_gpus', type=int,
      help='Number of gpus used for training. (0 or 1).', default=0)

  return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))