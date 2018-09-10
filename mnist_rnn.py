"""Module using IndRNNCell to solve the addition problem

The addition problem is stated in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well. The network should converge
to a MSE around zero after 1500-20000 steps, depending on the number of time
steps.
"""
import tensorflow as tf
import numpy as np
import sys

from ind_rnn_cell import IndRNNCell
from ind_rnn_cell import MultiRNNCell

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 784
NUM_UNITS = 128
LEARNING_RATE_INIT = 0.0002
LEARNING_RATE_DECAY_STEPS = 600000
NUM_LAYERS = 6
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
LAST_LAYER_LOWER_BOUND = pow(0.5, 1 / TIME_STEPS)

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50
OUTPUT_SIZE = 10
NUM_EPOCHS = 10000

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST", one_hot=False, validation_size=100)
ITERATIONS_PER_EPOCH = int(mnist.train.num_examples / BATCH_SIZE)
VAL_ITERS = int(mnist.validation.num_examples / BATCH_SIZE)
TEST_ITERS = int(mnist.test.num_examples / BATCH_SIZE)

def main():
  # Placeholders for training data
  print ("here")
  sys.stdout.flush()
  inputs_ph = tf.placeholder(tf.float32, shape=(None, TIME_STEPS))
  targets_ph = tf.placeholder(tf.int64, shape=(None))
  inputs_ph1 = tf.expand_dims(inputs_ph, -1)

  in_training = tf.placeholder(tf.bool, shape=[])
  input_init = tf.random_uniform_initializer(-0.001, 0.001)

  cells = []
  for layer in range(1, NUM_LAYERS+1):
      recurrent_init_lower = 0 if layer < NUM_LAYERS else LAST_LAYER_LOWER_BOUND
      recurrent_init = tf.random_uniform_initializer(recurrent_init_lower, RECURRENT_MAX)
      single_cell = IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX, batch_norm=False, in_training=in_training, layer_idx=layer-1)
      cells.append(single_cell)
      #input_initializer=input_init,
      #recurrent_initializer=recurrent_init))
  print ("here1")
  sys.stdout.flush()

  # Build the graph
  #cell = tf.nn.rnn_cell.MultiRNNCell([
  cell = MultiRNNCell(cells, BATCH_SIZE)
  # cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) #uncomment this for LSTM runs

  output, state = tf.nn.dynamic_rnn(cell, inputs_ph1, dtype=tf.float32)

  #print ( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn/multi_rnn_cell/cell_0'))
  #print ( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cell_1'))
  #print (tf.global_variables())
  #exit()

  #print ( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn/multi_rnn_cell/cell_1' ))
  #exit()
  #is_training = True
  #output = tf.layers.batch_normalization(output, training=is_training, momentum=0)
  last = output[:, -1, :]

  weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, OUTPUT_SIZE])
  bias = tf.get_variable("softmax_bias", shape=[1],
                         initializer=tf.constant_initializer(0.1))
  prediction = tf.squeeze(tf.matmul(last, weight) + bias)
  loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=targets_ph)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), targets_ph), tf.float32))
  print ("here2")
  sys.stdout.flush()

  global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                initializer=tf.zeros_initializer)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                             LEARNING_RATE_DECAY_STEPS, 0.1,
                                             staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  optimize = optimizer.minimize(loss_op, global_step=global_step)

  # Train the model
  np.random.seed(1234)
  perm = np.random.permutation(TIME_STEPS)
  print ("here3")
  sys.stdout.flush()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
  #fout = open('ind_semi_W_ckipnorm.txt', 'w')
  #fout = open('ind_input_init.txt', 'w')
  #fout = open('ind_bn.txt', 'w')
  #fout = open('ind_bn_2init.txt', 'w')
  #fout = open('ind_bn_after.txt', 'w')
  #fout = open('ind_bn3.txt', 'w')
  #fout = open('ind_semi_W_clipl2norm_bn.txt', 'w')
  fout = open('ind_semi_W_clipcrossnorm_bn.txt', 'w')
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_EPOCHS):
        print ("epoch:", epoch)
        sys.stdout.flush()
        train_acc = []
        for iteration in range(ITERATIONS_PER_EPOCH):
            x, y = mnist.train.next_batch(BATCH_SIZE)
            loss, _, acc= sess.run([loss_op, optimize, accuracy], {inputs_ph: x[:,perm], targets_ph: y, in_training: False})
            train_acc.append(acc)
            print (iteration, ITERATIONS_PER_EPOCH)
            sys.stdout.flush()

        valid_acc = []
        for iteration in range(VAL_ITERS):
            x, y = mnist.validation.next_batch(BATCH_SIZE)
            loss, acc = sess.run([loss_op, accuracy], {inputs_ph: x[:,perm], targets_ph: y, in_training: False})
            valid_acc.append(acc)

        test_acc = []
        for iteration in range(TEST_ITERS):
            x, y = mnist.test.next_batch(BATCH_SIZE)
            loss, acc = sess.run([loss_op, accuracy], {inputs_ph: x[:,perm], targets_ph: y, in_training: False})
            test_acc.append(acc)

        print ("epoch %d, train=%f, valid=%f, test=%f" % (epoch, np.mean(train_acc), np.mean(valid_acc), np.mean(test_acc)))
        fout.write("%d %.4f %.4f %.4f\n" % (epoch, np.mean(train_acc), np.mean(valid_acc), np.mean(test_acc)))
        sys.stdout.flush()
        fout.flush()



if __name__ == "__main__":
  main()
