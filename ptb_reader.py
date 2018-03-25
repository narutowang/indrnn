import random
import os, sys
import numpy as np
import collections
import six.moves.cPickle as pickle


PTB = collections.namedtuple("PTB", ("train", "valid", "test"))
Py3 = sys.version_info[0] == 3


class Subset(object):
    def __init__(self, samples, labels):
        self._num_examples = len(samples)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._samples = samples
        self._labels = labels

    @property
    def num_examples(self):
        return self._num_examples


    def next_batch(self, batch_size):
        """
        Return the next `batch_size` examples from this data set.
        Tuples of (sample, label, sequence_length)
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._samples[start:end], self._labels[start:end]


def _read_words(filename):
  #with tf.gfile.GFile(filename, "r") as f:
  with open(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def load_ptb(data_path, train_time_step=50, valid_time_step=50, test_time_step=2):
    train_path = os.path.join(data_path, "ptb.char.train.txt")
    valid_path = os.path.join(data_path, "ptb.char.valid.txt")
    test_path = os.path.join(data_path, "ptb.char.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    train_size = int(len(train_data) / train_time_step)
    valid_size = int(len(valid_data) / valid_time_step)
    test_size =  int(len(test_data) / test_time_step)

    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    test_data = np.array(test_data)

    train_samples = np.reshape(train_data[:(train_size*train_time_step)], [train_size, train_time_step])
    valid_samples = np.reshape(valid_data[:(valid_size*valid_time_step)], [valid_size, valid_time_step])
    test_samples = np.reshape(test_data[:(test_size*test_time_step)], [test_size, test_time_step])

    train_labels=np.reshape(train_data[1:train_size*train_time_step+1],[train_size, train_time_step])
    valid_labels=np.reshape(valid_data[1:valid_size*valid_time_step+1],[valid_size, valid_time_step])
    test_labels=np.reshape(test_data[1:test_size*test_time_step+1],[test_size, test_time_step])

    vocab_size = len(word_to_id)
    train_subset = Subset(train_samples, train_labels)
    valid_subset = Subset(valid_samples, valid_labels)
    test_subset = Subset(test_samples, test_labels)
    return PTB(train_subset, valid_subset, test_subset), vocab_size
