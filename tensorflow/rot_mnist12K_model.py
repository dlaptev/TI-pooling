import math
import tensorflow as tf

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

# xavier-like initializer
def weights_biases(kernel_shape, bias_shape):
  in_variables = 1
  for index in xrange(len(kernel_shape) - 1):
    in_variables *= kernel_shape[index]
  stdv = 1.0 / math.sqrt(in_variables)
  weights = tf.get_variable(
      'weights',
      kernel_shape,
      initializer=tf.random_uniform_initializer(-stdv, stdv))
  biases = tf.get_variable(
      'biases',
      bias_shape,
      initializer=tf.random_uniform_initializer(-stdv, stdv))
  return weights, biases

def conv_relu_maxpool(input, kernel_shape, bias_shape):
  weights, biases = weights_biases(kernel_shape, bias_shape)  
  return max_pool_2x2(tf.nn.relu(conv2d(input, weights) + biases))

def fc_relu(input, kernel_shape, bias_shape):
  weights, biases = weights_biases(kernel_shape, bias_shape)
  return tf.nn.relu(tf.matmul(input, weights) + biases)

def fc(input, kernel_shape, bias_shape):
  weights, biases = weights_biases(kernel_shape, bias_shape)
  return tf.matmul(input, weights) + biases

# x should already be reshaped as a 32x32x1 image
def single_branch(x, number_of_filters, number_of_fc_features):
  with tf.variable_scope('conv1'):
    max_pool1 = conv_relu_maxpool(x,
                                  [3,
                                   3,
                                   1,
                                   number_of_filters],
                                  [number_of_filters])
  with tf.variable_scope('conv2'):
    max_pool2 = conv_relu_maxpool(max_pool1,
                                  [3,
                                   3,
                                   number_of_filters,
                                   2 * number_of_filters],
                                  [2 * number_of_filters])
  with tf.variable_scope('conv3'):
    max_pool3 = conv_relu_maxpool(max_pool2,
                                  [3,
                                   3,
                                   2 * number_of_filters,
                                   4 * number_of_filters],
                                  [4 * number_of_filters])
    flattened_size = ((32 / 8) ** 2) * 4 * number_of_filters
    flattened = tf.reshape(max_pool3, [-1, flattened_size])
  with tf.variable_scope('fc1'):
    fc1 = fc_relu(flattened,
                  [flattened_size, number_of_fc_features],
                  [number_of_fc_features])
  return fc1

# x are batches nx32x32x1xnumber_of_transformations
def define_model(x,
                 keep_prob,
                 number_of_classes,
                 number_of_filters,
                 number_of_fc_features):
  splitted = tf.unpack(x, axis=4)
  branches = []
  with tf.variable_scope('branches') as scope:  
    for index, tensor_slice in enumerate(splitted):
      branches.append(single_branch(splitted[index],
                      number_of_filters,
                      number_of_fc_features))
      if (index == 0):
        scope.reuse_variables()
    concatenated = tf.pack(branches, axis=2)
    ti_pooled = tf.reduce_max(concatenated, reduction_indices=[2])
    drop = tf.nn.dropout(ti_pooled, keep_prob)
  with tf.variable_scope('fc2'):
    logits = fc(drop,
                [number_of_fc_features, number_of_classes],
                [number_of_classes])
  return logits
