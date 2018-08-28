# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tools for analyzing a TensorFlow graph.

This module exports the function determine_ready_tensors_and_table_initializers
which analyzes a TensorFlow graph to determine which tensors and table
initializers are "ready".  The concept of readiness arises as tf.Transform
works by building a single TF graph containing placeholders for the outputs
of analyzers.  These placeholders are progressively replaced by constants in
a number of phases, where in each phase we run some analyzers and replace their
outputs with constants.  We analyze the structure of the graph to determine
which analyzers to run in each phase.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools


import six
import tensorflow as tf

_INITIALIZABLE_TABLE_OP_TYPES = [
    'CuckooTable',
    'CuckooTableV2',
    'HashTable',
    'HashTableV2',
    'IndexTable',
    'IndexTableV2',
]

_TABLE_INIT_OP_TYPES = [
    'InitializeTable',
    'InitializeTableV2',
    'InitializeTableFromTextFile',
    'InitializeTableFromTextFileV2'
]


def _decompose_tensor_or_sparse_tensor(tensor):
  if isinstance(tensor, tf.SparseTensor):
    yield tensor.indices
    yield tensor.values
    yield tensor.dense_shape
  else:
    yield tensor


class _UnexpectedPlaceholderError(Exception):

  def __init__(self, op):
    tensor = op.outputs[0]
    msg = 'An unexpected placeholder was encountered ({})'.format(tensor)
    super(_UnexpectedPlaceholderError, self).__init__(msg)
    self.tensor = tensor


class _UnexpectedTableError(Exception):

  def __init__(self, op):
    msg = 'An unexpected initializable table was encountered ({})'.format(op)
    super(_UnexpectedTableError, self).__init__(msg)
    self.op = op


class _GraphAnalyzer(object):
  """Class that analyzes a graph to determine readiness of tensors.

  Args:
    known_ready: A dict from `Tensor` or `SparseTensor` or `Operation` to
        bool, indicating if this tensor is known to be ready or not ready.
  """

  def __init__(self, known_ready):
    self._memoized_ready_to_run = {}
    # Set the value of ready_to_run for each component of the tensors in
    # `known_ready`.
    for tensor, is_ready in six.iteritems(known_ready):
      for component in _decompose_tensor_or_sparse_tensor(tensor):
        self._memoized_ready_to_run[component] = is_ready

  def _ready_to_run_internal(self, tensor_or_op, stack=None):
    """Returns whether a given  `Tensor` or `Operation` is ready to run.

    Recursively computes whether a tensor or operation is ready to run, using
    `known_ready` as terminal nodes.  An error is thrown if a table or
    placeholder is reached: they must be set using known_ready.  This function
    is memoized using self._memoized_ready_to_run to cache computed values.
    Cycles are ignored (so a cycle is considered ready to run) and cycles are
    detected using `stack`.

    Args:
      tensor_or_op: A `Tensor` or `Operation`.
      stack: (optional) The tensors or operations that are on the stack (used to
          avoid cycles).

    Returns:
      Whether this op or tensor is ready to run.

    Raises:
      _UnexpectedTableError: If an initializable table op is encountered.
      _UnexpectedPlaceholderError: If a placeholder is encountered.
    """
    if tensor_or_op in self._memoized_ready_to_run:
      return self._memoized_ready_to_run[tensor_or_op]

    if isinstance(tensor_or_op, tf.Operation):
      if tensor_or_op.type in _INITIALIZABLE_TABLE_OP_TYPES:
        raise _UnexpectedTableError(tensor_or_op)
      if tensor_or_op.type == 'Placeholder':
        raise _UnexpectedPlaceholderError(tensor_or_op)
      parents = itertools.chain(tensor_or_op.inputs,
                                tensor_or_op.control_inputs)
    elif isinstance(tensor_or_op, tf.Tensor):
      parents = [tensor_or_op.op]
    else:
      raise TypeError('Expected Tensor or Operation, got {} of type {}'.format(
          tensor_or_op, type(tensor_or_op)))

    # Check that all parents are ready to run, ignoring parents that result
    # in a loop.  We assume that any loop is a valid while loop and so it will
    # be able to run as long as all the other parents are ready.
    if stack is None:
      stack = []
    stack.append(tensor_or_op)
    result = all(self._ready_to_run_internal(parent, stack)
                 for parent in parents if parent not in stack)
    assert tensor_or_op is stack.pop()

    self._memoized_ready_to_run[tensor_or_op] = result
    return result

  def ready_to_run(self, tensor_or_op):
    """Determine if a given tensor or op is ready to run.

    A tensor is ready to run if every tensor in all its transitive dependencies
    are set to `True` in `known_ready`.

    Note that if a placeholder is encountered, this will result in an error as
    it is assumed that all placeholders are keys in `known_ready`.  This is
    to avoid unexpected behavior when the user creates placeholders (as opposed
    to placeholders created by the tf.Transform framework).

    Similarly encountering a Table op is an error because a table should be
    a key in `known_ready` (in the case of analyzing the main session run) or
    should not be encountered (in the case of analyzing the graph init run).

    Args:
      tensor_or_op: A `Tensor`, `SparseTensor` or `Operation`

    Returns:
      A bool indicating whether then tensor is ready to run.

    Raises:
      ValueError: If a placeholder or table is encountered.
    """
    if not isinstance(tensor_or_op, (tf.Tensor, tf.SparseTensor, tf.Operation)):
      raise TypeError(
          'Expected Tensor, SparseTensor or Operation got {} of type {}'.format(
              tensor_or_op, type(tensor_or_op)))
    return all(
        self._ready_to_run_internal(component)
        for component in _decompose_tensor_or_sparse_tensor(tensor_or_op))


def determine_ready_tensors_and_table_initializers(graph, fetches, feeds,
                                                   replaced_tensors_ready):
  """Determines which tensors will be ready when running the graph.

  Determines which tensors from `fetches` are ready to run, using following
  algorithm.

  1. Determine which table initializers are ready to run.  A table initializer
     is an element of the TABLE_INITIALIZERS collection and it is ready to run
     if all the tensors it depends on are set to ready in
     `replaced_tensors_ready`.

  2. Determine which of `fetches` are ready to run.  A fetch is ready to run if
     it only depends on tensors in `feeds` and tensors that are set to ready in
     `replaced_tensors_ready`.

  Args:
    graph: a `Graph`.
    fetches: a list of `Tensor` or `SparseTensor`s
    feeds: a list of `Tensor` or `SparseTensor`s
    replaced_tensors_ready: a dict from `Tensor` to bool indicating whether a
        `Tensor` is ready in this phase.

  Returns:
    A pair (ready_table_initializers, ready_fetches) where
        ready_table_initializers a list containing the table initializers that
        are ready to run, and ready_fetches is the elements of `fetches` that
        are ready to run.

  Raises:
    ValueError: If unexpected placeholders or tables are encountered, or table
        initializers do not have the expected structure in the graph.
  """
  # Determine which table initializers are ready, based on the replaced tensors.
  # Since no input tensors are fed during table initialization, we do not set
  # the value of any tensors in `feeds`.

  graph_analyzer_for_table_init = _GraphAnalyzer(replaced_tensors_ready)
  ready_table_initializers = []
  ready_in_feed = {}

  for table_init_op in graph.get_collection(tf.GraphKeys.TABLE_INITIALIZERS):
    if table_init_op.type not in _TABLE_INIT_OP_TYPES:
      raise ValueError(
          'Table initializer {} did not have expected op type'.format(
              table_init_op))
    if not table_init_op.inputs:
      raise ValueError(
          'Table initializer {} did not have expected number if inputs '
          '(expected >= 1 inputs, got 0)'.format(table_init_op))
    table_op = table_init_op.inputs[0].op
    try:
      ready = all(map(graph_analyzer_for_table_init.ready_to_run,
                      table_init_op.inputs[1:]))
    except _UnexpectedPlaceholderError as e:
      raise ValueError(
          'The table initializer {} depended on a placeholder ({}).  Note '
          'placeholders will not be fed during table initialization'.format(
              table_init_op, e.tensor))
    except _UnexpectedTableError as e:
      raise ValueError(
          'The table initializer {} depended on an initializable table ({}). '
          'Note tables are initialized in one pass so a table initializer '
          'cannot depend on the output of an initializeable table'.format(
              table_init_op, e.op))

    ready_in_feed[table_op] = ready
    if ready:
      ready_table_initializers.append(table_init_op)

  # Now determine which tensors are ready to run once the table has been
  # initialized.
  ready_in_feed.update(replaced_tensors_ready)
  ready_in_feed.update({tensor: True for tensor in feeds})
  graph_analyzer_for_feed = _GraphAnalyzer(ready_in_feed)
  ready_fetches = []
  for tensor in fetches:
    try:
      if graph_analyzer_for_feed.ready_to_run(tensor):
        ready_fetches.append(tensor)
    except _UnexpectedPlaceholderError as e:
      raise ValueError(
          'The tensor {} depended on a placeholder ({}) that was not in the '
          'feed dict.  This may have be caused by manually adding a '
          'placeholder to the graph'.format(tensor, e.tensor))
    except _UnexpectedTableError as e:
      raise ValueError(
          'The tensor {} depended on an initializable table ({}) that was not '
          'tracked by the graph analysis.  This may be caused by adding an '
          'initializable table without adding its initializer to the '
          'collection tf.GraphKeys.TABLE_INITIALIZERS'.format(tensor, e.op))

  return (ready_table_initializers, ready_fetches)
