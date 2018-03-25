"""Module implementing the IndRNN cell"""

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import base as base_layer
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops

def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99):
    with tf.variable_scope(name_scope):
        size = inputs.get_shape().as_list()[1]

        scale = tf.get_variable(
            'scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        population_mean = tf.get_variable(
            'population_mean', [size],
            initializer=tf.zeros_initializer(), trainable=False)
        population_var = tf.get_variable(
            'population_var', [size],
            initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        # The following part is based on the implementation of :
        # https://github.com/cooijmanstim/recurrent-batch-normalization
        train_mean_op = tf.assign(
            population_mean,
            population_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            population_var, population_var * decay + batch_var * (1 - decay))

        if is_training is True:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(
                    inputs, batch_mean, batch_var, offset, scale, epsilon)
        else:
            return tf.nn.batch_normalization(
                inputs, population_mean, population_var, offset, scale,
                epsilon)

class IndRNNCell(rnn_cell_impl._LayerRNNCell):
  """Independently RNN Cell. Adapted from `rnn_cell_impl.BasicRNNCell`.

  The implementation is based on:

    https://arxiv.org/abs/1803.04831

  Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao
  "Independently Recurrent Neural Network (IndRNN): Building A Longer and
  Deeper RNN"

  Each unit has a single recurrent weight connected to its last hidden state.

  Args:
    num_units: int, The number of units in the RNN cell.
    recurrent_min_abs: float, minimum absolute value of each recurrent weight.
    recurrent_max_abs: (optional) float, maximum absolute value of each
      recurrent weight. For `relu` activation, `pow(2, 1/timesteps)` is
      recommended. If None, recurrent weights will not be clipped.
      Default: None.
    recurrent_initializer: (optional) The initializer to use for the recurrent
      weights. The default is a uniform distribution in the range `[-1, 1]` if
      `recurrent_max_abs` is not set or in
      `[-recurrent_max_abs, recurrent_max_abs]` if it is and
      `recurrent_max_abs < 1`.
    activation: Nonlinearity to use.  Default: `relu`.
    reuse: (optional) Python boolean describing whether to reuse variables
      in an existing scope.  If not `True`, and the existing scope already has
      the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               recurrent_min_abs=0,
               recurrent_max_abs=None,
               recurrent_initializer=None,
               activation=None,
               reuse=None,
               name=None,
               batch_norm=False,
               in_training=False):
    super(IndRNNCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._recurrent_min_abs = recurrent_min_abs
    self._recurrent_max_abs = recurrent_max_abs
    self._recurrent_initializer = recurrent_initializer
    self._activation = activation or nn_ops.relu

    self._batch_norm = batch_norm
    self._in_training = in_training

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._input_kernel = self.add_variable(
        "input_kernel",
        shape=[input_depth, self._num_units])

    self._hierarchy_kernel1 = self.add_variable(
        "hierarchy_kernel1",
        shape=[self._num_units, self._num_units])

    if self._recurrent_initializer is None:
      # Initialize the recurrent weights uniformly in [-max_abs, max_abs] or
      # [-1, 1] if max_abs exceeds 1
      init_bound = 1.0
      if self._recurrent_max_abs and self._recurrent_max_abs < init_bound:
        init_bound = self._recurrent_max_abs

      self._recurrent_initializer = init_ops.random_uniform_initializer(
          minval=-init_bound,
          maxval=init_bound
      )

    self._recurrent_kernel = self.add_variable(
        "recurrent_kernel",
        shape=[self._num_units], initializer=self._recurrent_initializer)

    # Clip the absolute values of the recurrent weights to the specified minimum
    if self._recurrent_min_abs:
      abs_kernel = math_ops.abs(self._recurrent_kernel)
      min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
      self._recurrent_kernel = math_ops.multiply(
          math_ops.sign(self._recurrent_kernel),
          min_abs_kernel
      )

    # Clip the absolute values of the recurrent weights to the specified maximum
    if self._recurrent_max_abs:
      self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel,
                                                      -self._recurrent_max_abs,
                                                      self._recurrent_max_abs)

    self._hierarchy_kernel = self.add_variable(
        "hierarchy_kernel",
        shape=[self._num_units], initializer=self._recurrent_initializer)

    if self._recurrent_min_abs:
      abs_kernel = math_ops.abs(self._hierarchy_kernel)
      min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
      self._hierarchy_kernel = math_ops.multiply(
          math_ops.sign(self._hierarchy_kernel),
          min_abs_kernel
      )

    if self._recurrent_max_abs:
      self._hierarchy_kernel = clip_ops.clip_by_value(self._hierarchy_kernel,
                                                      -self._recurrent_max_abs,
                                                      self._recurrent_max_abs)

    self._bias = self.add_variable(
        "bias",
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    if self._batch_norm:
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)

    self.built = True

  def call(self, inputs, state, last_state=None):
    """Run one step of the IndRNN.

    Calculates the output and new hidden state using the IndRNN equation

      `output = new_state = act(W * input + u (*) state + b)`

    , where `*` is the matrix multiplication and `(*)` is the Hadamard product.

    Args:
      inputs: Tensor, 2-dimensional tensor of shape `[batch, num_units]`.
      state: Tensor, 2-dimensional tensor of shape `[batch, num_units]`
        containing the previous hidden state.

    Returns:
      A tuple containing the output and new hidden state. Both are the same
        2-dimensional tensor of shape `[batch, num_units]`.
    """
    gate_inputs = math_ops.matmul(inputs, self._input_kernel)
    is_training = True
    gate_inputs = batch_normal(gate_inputs, 'gate_inputs', is_training)
    recurrent_update = math_ops.multiply(state, self._recurrent_kernel)
    recurrent_update = batch_normal(recurrent_update, 'recurrent_update', is_training)
    #if last_state:
    #    hierarchy_update = math_ops.multiply(last_state, self._hierarchy_kernel)
    #    gate_inputs = math_ops.add(gate_inputs, hierarchy_update)

        #gate_inputs = math_ops.add(gate_inputs, last_state)

        #hierarchy_update = math_ops.matmul(last_state, self._hierarchy_kernel1)
        #gate_inputs = math_ops.add(gate_inputs, hierarchy_update)
    #recurrent_update = math_ops.add(recurrent_update, tf.tile(math_ops.reduce_mean(recurrent_update, 1, keep_dims=True), [1,128]))
    gate_inputs = math_ops.add(gate_inputs, recurrent_update)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    if self._batch_norm:
        output = self.bn(output, training=self._in_training)
    return output, output


class MultiRNNCell(rnn_cell_impl._LayerRNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    super(MultiRNNCell, self).__init__()
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_sequence(cells):
      raise TypeError(
          "cells must be a list or tuple, but saw: %s." % cells)

    self._cells = cells
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

    self.last_states = [None for _ in range(len(cells)+1)]

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        return super(MultiRNNCell, self).zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run this multi-layer cell on inputs, starting from state."""
    cur_state_pos = 0
    cur_inp = inputs
    new_states = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        cur_inp, new_state = cell(cur_inp, cur_state, self.last_states[i+1])
        new_states.append(new_state)
        self.last_states[i] = new_state

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))

    return cur_inp, new_states
