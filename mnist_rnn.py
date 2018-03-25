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

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50
OUTPUT_SIZE = 10
NUM_EPOCHS = 10000

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST", one_hot=False, validation_size=5000)
ITERATIONS_PER_EPOCH = int(mnist.train.num_examples / BATCH_SIZE)
VAL_ITERS = int(mnist.validation.num_examples / BATCH_SIZE)
TEST_ITERS = int(mnist.test.num_examples / BATCH_SIZE)

def main():
  # Placeholders for training data
  inputs_ph = tf.placeholder(tf.float32, shape=(None, TIME_STEPS))
  targets_ph = tf.placeholder(tf.int64, shape=(None))
  inputs_ph1 = tf.expand_dims(inputs_ph, -1)

  in_training = tf.placeholder(tf.bool, shape=[])

  # Build the graph
  #cell = tf.nn.rnn_cell.MultiRNNCell([
  cell = MultiRNNCell([
    IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX, batch_norm=False, in_training=in_training) for _ in
    range(NUM_LAYERS)
  ])
  # cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) #uncomment this for LSTM runs

  output, state = tf.nn.dynamic_rnn(cell, inputs_ph1, dtype=tf.float32)
  last = output[:, -1, :]

  weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, OUTPUT_SIZE])
  bias = tf.get_variable("softmax_bias", shape=[1],
                         initializer=tf.constant_initializer(0.1))
  prediction = tf.squeeze(tf.matmul(last, weight) + bias)
  loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=targets_ph)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), targets_ph), tf.float32))

  global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                initializer=tf.zeros_initializer)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                             LEARNING_RATE_DECAY_STEPS, 0.1,
                                             staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  optimize = optimizer.minimize(loss_op, global_step=global_step)

  # Train the model
  fout = open('ind.txt', 'w')
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_EPOCHS):
        train_acc = []
        for iteration in range(ITERATIONS_PER_EPOCH):
            x, y = mnist.train.next_batch(BATCH_SIZE)
            loss, _, acc= sess.run([loss_op, optimize, accuracy], {inputs_ph: x, targets_ph: y, in_training: False})
            train_acc.append(acc)
            #print (iteration, ITERATIONS_PER_EPOCH)
            sys.stdout.flush()

        valid_acc = []
        for iteration in range(VAL_ITERS):
            x, y = mnist.validation.next_batch(BATCH_SIZE)
            loss, acc = sess.run([loss_op, accuracy], {inputs_ph: x, targets_ph: y, in_training: False})
            valid_acc.append(acc)

        test_acc = []
        for iteration in range(TEST_ITERS):
            x, y = mnist.test.next_batch(256)
            loss, acc = sess.run([loss_op, accuracy], {inputs_ph: x, targets_ph: y, in_training: False})
            test_acc.append(acc)

        print ("epoch %d, train=%f, valid=%f, test=%f" % (epoch, np.mean(train_acc), np.mean(valid_acc), np.mean(test_acc)))
        fout.write("%d %.4f %.4f %.4f\n" % (epoch, np.mean(train_acc), np.mean(valid_acc), np.mean(test_acc)))
        sys.stdout.flush()
        fout.flush()



if __name__ == "__main__":
  main()
