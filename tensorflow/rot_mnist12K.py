import tools
import rot_mnist12K_model
import tensorflow as tf
import sys

NUMBER_OF_CLASSES = 10
NUMBER_OF_FILTERS = 40
NUMBER_OF_FC_FEATURES = 5120
NUMBER_OF_ROTATIONS = 1
INPUT_SIZE = 28
DESIRED_SIZE = 32
BATCH_SIZE = 64
# set up training graph
x = tf.placeholder(tf.float32, shape=[None, DESIRED_SIZE, DESIRED_SIZE, 1, NUMBER_OF_ROTATIONS])
y_gt = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])
keep_prob = tf.placeholder(tf.float32)
logits = rot_mnist12K_model.define_model(x,
                                         keep_prob,
                                         NUMBER_OF_CLASSES,
                                         NUMBER_OF_FILTERS,
                                         NUMBER_OF_FC_FEATURES,
                                         NUMBER_OF_ROTATIONS)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_gt))
train_step = tf.train.AdadeltaOptimizer().minimize(cross_entropy) #rho=0.9, epsilon=1e-06
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# run training
session = tf.Session()
session.run(tf.initialize_all_variables())
train_data_loader = tools.DataLoader('../mnist_all_rotation_normalized_float_train_valid.amat',
                                     NUMBER_OF_CLASSES,
                                     NUMBER_OF_ROTATIONS)
test_data_loader = tools.DataLoader('../mnist_all_rotation_normalized_float_test.amat',
                                     NUMBER_OF_CLASSES,
                                     NUMBER_OF_ROTATIONS)
test_size = test_data_loader.all()[1].shape[0]
test_chunk_size = 1000
assert test_size % test_chunk_size == 0
number_of_test_chunks = test_size / test_chunk_size
while (True):
  batch = train_data_loader.next_batch(BATCH_SIZE)
  if (train_data_loader.is_new_epoch()):
    train_accuracy = session.run(accuracy, feed_dict={x : batch[0],
                                                      y_gt : batch[1],
                                                      keep_prob : 1.0})
    print("completed_epochs %d, training accuracy %g" % (train_data_loader.get_completed_epochs(),
                                                         train_accuracy))
    sys.stdout.flush()
    if (train_data_loader.get_completed_epochs() % 10 == 0):
      sum = 0.0
      for chunk_index in xrange(number_of_test_chunks):
        chunk = test_data_loader.next_batch(test_chunk_size)
        sum += session.run(accuracy, feed_dict={x : chunk[0],
                                                y_gt : chunk[1],
                                                keep_prob : 1.0})
      test_accuracy = sum / number_of_test_chunks
      print("testing accuracy %g" % test_accuracy)
      sys.stdout.flush()
  session.run(train_step, feed_dict={x : batch[0],
                                     y_gt : batch[1],
                                     keep_prob : 0.5})


