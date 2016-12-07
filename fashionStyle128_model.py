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

import fashionStyle128 as util
import pdb

HParams = namedtuple('HParams',
                     'batch_size, epoch_size, loss, alpha, '
                     'optimizer, learning_rate, learning_rate_decay_epochs, '
                     'learning_rate_decay_factor, moving_average_decay')


class Style128Net(object):

  def __init__(self, hps, images, labels, mode, learning_rate=None):
    """Style128Net constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batch of labels [batch_size, 123]. Each label is a 123-dim
             vector of 0/1 values for singles tag.
      mode: One of 'pretrain', 'joint', 'pretrain_forward', 'joint_forward'
    """
    self.hps = hps
    self.images = images
    self.labels = labels
    self.mode = mode
    self.learning_rate_placeholder = learning_rate
    self.kclasses = 123
      

  def build_graph(self):
    """Build a whole graph for the model."""
    classification_network_scopes = {
      'pretrain' : 'cl_pretrain',
      'joint' : 'cl_joint'
    }

    if self.mode == 'pretrain':
      self.summary_collection = 'SUMMARY_cl'
      self.global_step = tf.Variable(0, name='global_step_cl', trainable=False)
      self._build_embedding_network(reuse=None)
      self._build_classification_net_input(mode='single')
      self.classification_network_scope = classification_network_scopes[self.mode]
      self._build_classification_network_pretrain(scope=self.classification_network_scope, reuse=None)
      
      loss = self._build_classification_loss()
      self.loss = loss

      self.loss_collection = 'cl_losses'
      tf.add_to_collection(self.loss_collection, loss)
      self._set_learning_rate()
      self._build_train_op()

      #record images
      util.add_to_collections(self.summary_collection, tf.image_summary(self.mode+'/'+'fashion', self.images, max_images=100))
      # Build the summary operation based on the TF collection of Summaries.
      self.summaries = tf.merge_all_summaries(key=self.summary_collection)

    elif self.mode == 'joint':
      self.summary_collection = 'SUMMARY_joint'

      self.global_step = tf.Variable(0, name='global_step_joint', trainable=False)
      #assumes that the network has been pre-trained in classification mode
      self._build_embedding_network(reuse=True) 
      embedding_loss = self._build_embedding_loss()

      self._build_classification_net_input(mode='triplet')
      #reset the classification network.
      self.classification_network_scope = classification_network_scopes[self.mode]
      self._build_classification_network(scope=self.classification_network_scope, reuse=None)
      class_loss = self._build_classification_loss()
      
      #total loss is the sum of the losses.
      self.loss = tf.add(embedding_loss, class_loss, name='total_loss')

      self.loss_collection = 'joint_losses'
      tf.add_to_collection(self.loss_collection, class_loss)
      #tf.scalar_summary('cl_loss', class_loss)
      tf.add_to_collection(self.loss_collection, embedding_loss)
      #tf.scalar_summary('embedding_loss', embedding_loss)
      tf.add_to_collection(self.loss_collection, self.loss)
      #tf.scalar_summary('total_loss', self.loss)

      self._set_learning_rate()
      self._build_train_op()


      #record images
      util.add_to_collections(self.summary_collection, tf.image_summary(self.mode+'/'+'fashion', self.images, max_images=100))
      # Build the summary operation based on the TF collection of Summaries.
      self.summaries = tf.merge_all_summaries(key=self.summary_collection)

    elif self.mode == 'pretrain_forward':
      #Reuse the weights when evaluating during training.
      self._build_embedding_network(reuse=True) 
      self._build_classification_net_input(mode='single')
      self.classification_network_scope = classification_network_scopes['pretrain']
      self._build_classification_network_pretrain(scope=self.classification_network_scope, reuse=True)

    else:
      assert(self.mode == 'joint_forward')
      self._build_embedding_network(reuse=True) 
      self._build_classification_net_input(mode='single')
      self.classification_network_scope = classification_network_scopes['joint']
      self._build_classification_network(scope=self.classification_network_scope, reuse=True)
    



  def _build_embedding_network(self, reuse=None):
    """Build the core model within the graph.
      convolutional layers are followed by relu activations
      convolutional layers have 1x1 stride, and zero padding.
    """
    with tf.name_scope('input'):
      x = self.images
    with tf.variable_scope('embedding'):
      net = slim.conv2d(x, 64, [3, 3], scope='conv3_1', reuse=reuse)
      #self.conv3_1 = net
      net = slim.conv2d(net, 64, [3, 3], scope='conv3_2', reuse=reuse)
      #self.conv3_2 = net
      net = slim.dropout(net, 0.25, scope='dropout1')
      #self.dropout1 = net

      net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool1')
      #self.pool1 = net
      net = slim.batch_norm(net, scope='batch_norm1', reuse=reuse)
      #self.batch_norm1 = net
      net = slim.conv2d(net, 128, [3, 3], scope='conv3_3', reuse=reuse)
      #self.conv3_3 = net
      net = slim.conv2d(net, 128, [3, 3], scope='conv3_4', reuse=reuse)
      #self.conv3_4 = net
      net = slim.dropout(net, 0.25, scope='dropout2')
      #self.dropout2 = net

      net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool2')
      #self.pool2 = net
      net = slim.batch_norm(net, scope='batch_norm2', reuse=reuse)
      #self.batch_norm2 = net
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_5', reuse=reuse)
      #self.conv3_5 = net
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_6', reuse=reuse)
      #self.conv3_6 = net
      net = slim.dropout(net, 0.25, scope='dropout3')
      #self.dropout3 = net

      net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool3')
      #self.pool3 = net
      net = slim.batch_norm(net, scope='batch_norm3', reuse=reuse)
      #self.batch_norm3 = net
      net = slim.conv2d(net, 128, [3, 3], scope='conv3_7', reuse=reuse)
      #self.conv3_7 = net
      #embeddings = slim.fully_connected(net, 128, activation_fn=None, scope='fc')
      net = slim.conv2d(net, 128, [6, 4], padding='VALID', activation_fn=None, scope='fc', reuse=reuse)
      #self.fc = net
      self.embeddings = tf.squeeze(net, [1, 2], name='fc/squeezed')

  def _build_embedding_loss(self):
    """
    Assumes that self.embeddings has been set.
    Assumes input is a triplet.
    """
    anchor, positive, negative = tf.split(0, 3, self.embeddings)
    if self.hps.loss == 'RANKING':
      loss = self.ranking_loss(anchor, positive, negative)
    elif self.hps.loss == 'TRIPLET':
      loss = self.triplet_loss(anchor, positive, negative, self.hps.alpha)
    else:
      raise ValueError('Invalid loss')


    self.embedding_loss = loss
    return loss

  def _build_classification_net_input(self, mode='single'):
    """
    Assumes that self.embeddings has been set.
    mode is either 'single' or 'triplet'
    triplet assumes that the embeddings represents a triplet batch where
    the first third is the anchor image, the second third is the positive image,
    and the last third is the negative image.
    -only the negative image serves as input to the classification network.

    single assumes that each row in the embeddings matrix corresponds to an image
    to serve as input to the classification network.
    """
    if mode=='single':
      self.classification_input = self.embeddings
      self.classification_labels = self.labels
    else:
      assert(mode=='triplet')
      _, _, self.classification_input = tf.split(0, 3, self.embeddings)
      _, _, self.classification_labels = tf.split(0,3, self.labels)


  def _build_classification_network(self, scope='classification', reuse=None):
    """
    Classification network assumes that _build_classification_network_input has
    been ran.

    joint = True will prepend "joint" to the prefix to specify that these weights
    were trained during the joint classification + embedding network process.
    This enables us to initialize a fresh set of weights.
    """
    with tf.variable_scope(scope):
      net = slim.batch_norm(
        self.classification_input,
        scope='batch_norm1',
        activation_fn=tf.nn.relu,
        reuse=reuse)
      net = slim.fully_connected(net, 128, activation_fn=None, scope='fc1', reuse=reuse)
      self.predictions = slim.fully_connected(net, self.kclasses, activation_fn=None, scope='fc2', reuse=reuse)

  def _build_classification_network_pretrain(self, scope='classification', reuse=None):
    with tf.variable_scope(scope):
      self.predictions = slim.fully_connected(self.classification_input, self.kclasses, activation_fn=None, scope='fc', reuse=reuse)

  def _build_classification_loss(self):
    with tf.variable_scope('classification_loss'):
      pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        self.predictions,
        self.classification_labels,
        name='sigmoid_cross_entropy_loss')
      self.class_pre_loss = pre_loss
      loss = tf.reduce_mean(pre_loss)
    return loss





  def ranking_loss(self, anchor, positive, negative):
    with tf.variable_scope('ranking_loss'):
      pos_dist_exp = tf.exp(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)))  # Summing over distances in each batch
      neg_dist_exp = tf.exp(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)))
      d_positive = tf.truediv(pos_dist_exp, pos_dist_exp + neg_dist_exp, name='pos_softmax')
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
    # self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

    self.learning_rate = tf.train.exponential_decay(
      self.learning_rate_placeholder,
      self.global_step,
      self.hps.learning_rate_decay_epochs*self.hps.epoch_size,
      self.hps.learning_rate_decay_factor, staircase=True)
    util.add_to_collections(self.summary_collection, tf.scalar_summary(self.mode+'/'+'learning_rate', self.learning_rate))



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
    losses = tf.get_collection(self.loss_collection)

    loss_averages_op = loss_averages.apply(losses)
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        util.add_to_collections(self.summary_collection, tf.scalar_summary(self.mode+'/'+l.op.name +' (raw)', l))
        util.add_to_collections(self.summary_collection, tf.scalar_summary(self.mode+'/'+l.op.name, loss_averages.average(l)))
  
    return loss_averages_op

  def _build_train_op(self, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = self._add_loss_summaries()
    all_trainable_variables = tf.trainable_variables() #tf.all_variables()

    # always include the embedding variables & include the appropriate classification
    # network variables
    trainable_variables = [v for v in all_trainable_variables
                            if v.name.startswith('embedding/')
                            or v.name.startswith(self.classification_network_scope+'/')]

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
        for var in trainable_variables:
            util.add_to_collections(self.summary_collection, tf.histogram_summary(self.mode+'/'+var.op.name, var))
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                util.add_to_collections(self.summary_collection, tf.histogram_summary(self.mode+'/'+var.op.name + '/gradients', grad))
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        self.hps.moving_average_decay, self.global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #Does nothing. Only useful as a placeholder for control edges.
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        self.train_op = tf.no_op(name='train')


