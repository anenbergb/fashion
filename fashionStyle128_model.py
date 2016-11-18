"""
Fashion style model
Implements:
http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import math_ops

HParams = namedtuple('HParams',
                     'batch_size, epoch_size, loss, alpha, '
                     'optimizer, learning_rate, learning_rate_decay_epochs, '
                     'learning_rate_decay_factor, moving_average_decay')


class Style128Net(object):

  def __init__(self, hps, images, mode):
    """Style128Net constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self.images_placeholder = images
    self.mode = mode

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._set_learning_rate()
      self._build_train_op()
    # Build the summary operation based on the TF collection of Summaries.
    self.summaries = tf.merge_all_summaries()

  def _build_model(self):
    """Build the core model within the graph.
      convolutional layers are followed by relu activations
      convolutional layers have 1x1 stride, and zero padding.
    """
    with tf.name_scope('input'):
      x = self.images_placeholder
    
    net = slim.conv2d(x, 64, [3, 3], scope='conv3_1')
    #self.conv3_1 = net
    net = slim.conv2d(net, 64, [3, 3], scope='conv3_2')
    #self.conv3_2 = net
    net = slim.dropout(net, 0.25, scope='dropout1')
    #self.dropout1 = net

    net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool1')
    #self.pool1 = net
    net = slim.batch_norm(net, scope='batch_norm1')
    #self.batch_norm1 = net
    net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
    #self.conv3_3 = net
    net = slim.conv2d(net, 128, [3, 3], scope='conv3_4')
    #self.conv3_4 = net
    net = slim.dropout(net, 0.25, scope='dropout2')
    #self.dropout2 = net

    net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool2')
    #self.pool2 = net
    net = slim.batch_norm(net, scope='batch_norm2')
    #self.batch_norm2 = net
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_5')
    #self.conv3_5 = net
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_6')
    #self.conv3_6 = net
    net = slim.dropout(net, 0.25, scope='dropout3')
    #self.dropout3 = net

    net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool3')
    #self.pool3 = net
    net = slim.batch_norm(net, scope='batch_norm3')
    #self.batch_norm3 = net
    net = slim.conv2d(net, 128, [3, 3], scope='conv3_7')
    #self.conv3_7 = net
    #embeddings = slim.fully_connected(net, 128, activation_fn=None, scope='fc')
    net = slim.conv2d(net, 128, [6, 4], padding='VALID', activation_fn=None, scope='fc')
    #self.fc = net

    embeddings = tf.squeeze(net, [1, 2], name='fc/squeezed')

    anchor, positive, negative = tf.split(0, 3, embeddings)
    if self.hps.loss == 'RANKING':
      self.loss = self.ranking_loss(anchor, positive, negative)
    elif self.hps.loss == 'TRIPLET':
      self.loss = self.triplet_loss(anchor, positive, negative, self.hps.alpha)
    else:
      raise ValueError('Invalid loss')

    tf.scalar_summary('loss', self.loss)

    ###REMOVE:
    # self.loss2 = self.triplet_loss(anchor, positive, negative, self.hps.alpha)
    # self.anchor = anchor
    # self.positive = positive
    # self.negative = negative
    # self.embeddings = embeddings

  def ranking_loss(self, anchor, positive, negative):
    with tf.variable_scope('ranking_loss'):
      #anchor.get_shape().assert_is_compatible_with(positive.get_shape())
      #anchor.get_shape().assert_is_compatible_with(negative.get_shape())
      #anchor = math_ops.to_float(anchor)
      #positive = math_ops.to_float(positive)
      #negative = math_ops.to_float(negative)

      pos_dist_exp = tf.exp(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)))  # Summing over distances in each batch
      neg_dist_exp = tf.exp(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)))


      d_positive = tf.truediv(pos_dist_exp, pos_dist_exp + neg_dist_exp, name='pos_softmax')
      #d_negative = tf.truediv(neg_dist_exp, pos_dist_exp + neg_dist_exp, name='neg_softmax')
      #loss = tf.scalar_mul(0.5, tf.add(tf.square(d_positive), tf.square(tf.sub(1.0, d_negative))))
      pre_loss = tf.square(d_positive)
      loss = tf.reduce_mean(pre_loss, 0)
    return loss

  def triplet_loss(self, anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)  # Summing over distances in each batch
        neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.sub(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss


  def _set_learning_rate(self):

    # Placeholder for the learning rate
    self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

    self.learning_rate = tf.train.exponential_decay(self.learning_rate_placeholder,
      self.global_step,
      self.hps.learning_rate_decay_epochs*self.hps.epoch_size,
      self.hps.learning_rate_decay_factor, staircase=True)
    tf.scalar_summary('learning_rate', self.learning_rate)



  def _add_loss_summaries(self):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')

    loss_averages_op = loss_averages.apply(losses + [self.loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [self.loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

  def _build_train_op(self, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = self._add_loss_summaries()
    trainable_variables = tf.trainable_variables() #tf.all_variables()

    # Compute gradients. grads is only ran after loss_averages_op is ran.
    with tf.control_dependencies([loss_averages_op]):
        if self.hps.optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.hps.optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.9, epsilon=1e-6)
        elif self.hps.optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif self.hps.optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif self.hps.optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
        elif self.hps.optimizer=='SGD':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate, use_locking=False)
        else:
            raise ValueError('Invalid optimization algorithm')
        # List of (gradient, variable) pairs 
        grads = opt.compute_gradients(self.loss, trainable_variables)
        
    # Apply gradients. Increment global_step.
    apply_gradient_op = opt.apply_gradients(grads,
      global_step=self.global_step,
      name='train_step')
    
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        self.hps.moving_average_decay, self.global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #Does nothing. Only useful as a placeholder for control edges.
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        self.train_op = tf.no_op(name='train')


