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
TIME_STEPS = 50
NUM_UNITS = 2000
LEARNING_RATE_INIT = 0.0002
LEARNING_RATE_DECAY_STEPS = 600000
NUM_LAYERS = 6
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 128
NUM_EPOCHS = 10000

class IndRNNConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.0002
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 200
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.75
  lr_decay = 0.1
  batch_size = 128
  vocab_size = 10000
  #rnn_mode = BLOCK

from ptb_word_lm import *
from ptb_reader import *

ptb, vocab_size = load_ptb("/home/ziclin/data")
ITERATIONS_PER_EPOCH = int(ptb.train.num_examples / BATCH_SIZE)
VAL_ITERS = int(ptb.valid.num_examples / BATCH_SIZE)
TEST_ITERS = int(ptb.test.num_examples / BATCH_SIZE)
print (">>>>")
print ("train_iter: %d, valid_iter: %d, test_iter: %d" % (ITERATIONS_PER_EPOCH, VAL_ITERS, TEST_ITERS))

def main():
  inputs_ph = tf.placeholder(tf.int64, shape=(None, None))
  labels_ph = tf.placeholder(tf.int64, shape=(None, None))

  embedding = tf.get_variable("embedding", [vocab_size, NUM_UNITS], dtype=tf.float32)
  inputs = tf.nn.embedding_lookup(embedding, inputs_ph)
  in_training = True
  #if in_training:
  #  inputs = tf.nn.dropout(inputs, 0.75)
  
  cell = MultiRNNCell([
    IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX, batch_norm=False, in_training=in_training) for _ in
    range(NUM_LAYERS)
  ])
  # cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) #uncomment this for LSTM runs

  output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
  softmax_w = tf.get_variable("softmax_w", [NUM_UNITS, vocab_size], dtype=tf.float32)
  softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
  output = tf.reshape(output, [-1, NUM_UNITS])
  logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
  print (logits)
   # Reshape logits to be a 3-D tensor for sequence loss
  logits = tf.reshape(logits, [BATCH_SIZE, -1, vocab_size])

  # Use the contrib sequence loss and average over the batches
  loss = tf.contrib.seq2seq.sequence_loss(
      logits,
      labels_ph,
      tf.ones([BATCH_SIZE, 50], dtype=tf.float32),
      average_across_timesteps=False,
      average_across_batch=True)

  # Update the cost
  _cost = tf.reduce_sum(loss)
  _final_state = state
  #########
  if not in_training:
    return

  global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                initializer=tf.zeros_initializer)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                             LEARNING_RATE_DECAY_STEPS, 0.1,
                                             staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  optimize = optimizer.minimize(_cost, global_step=global_step)

  # Train the model

  fout = open('ptb_ind.txt', 'w')
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_EPOCHS):
      train_per = []
      for iteration in range(ITERATIONS_PER_EPOCH):
        x, y = ptb.train.next_batch(BATCH_SIZE)
        cost, _ = sess.run([_cost, optimize], feed_dict={inputs_ph: x, labels_ph: y})
        train_per.append(cost)
        if iteration % ITERATIONS_PER_EPOCH == 20:
            print ("%d/%d  %f" % (iteration, ITERATIONS_PER_EPOCH, np.mean(train_per[-20:])))
            sys.stdout.flush()

      valid_per = []
      for _ in range(VAL_ITERS):
        x, y = ptb.valid.next_batch()
        cost = sess.run(_cost, feed_dict={inputs_ph: x, labels_ph: y})
        valid_per.append(cost)

      #test_per = []
      #for _ in range(VAL_ITERS):
      #  x, y = ptb.test.next_batch()
      #  cost = sess.run(_cost, feed_dict={inputs_ph: x, labels_ph: y})
      #  test_per.append(cost)

      print ("epoch %d, train=%f, valid=%f, test=%f" % (epoch, np.mean(train_per), np.mean(valid_per), np.mean(test_per)))
      fout.write("%d %.4f %.4f %.4f\n" % (epoch, np.mean(train_per), np.mean(valid_per), np.mean(test_per)))
      sys.stdout.flush()
      fout.flush()

if __name__ == "__main__":
  main()
