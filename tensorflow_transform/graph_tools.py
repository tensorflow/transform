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

import collections
import itertools

# GOOGLE-INITIALIZATION

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
    'InitializeTableFromTextFileV2',
    'LookupTableImport',
    'LookupTableImportV2',
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


class _AnalysisResult(
    collections.namedtuple('_AnalysisResult', ['is_ready_to_run', 'path'])):
  pass


class _SourceInfo(
    collections.namedtuple('_SourceInfo', ['is_ready_to_run', 'name'])):
  pass


class _GraphAnalyzer(object):
  """Class that analyzes a graph to determine readiness of tensors.

  Args:
    source_info_dict: A dict from `Tensor` or `Operation` to `_SourceInfo`.
    translate_path_fn: A function with the signature:
        (identifier, parents) -> Any which will be used to construct a unique
        path for a given `Tensor`.
  """

  def __init__(self, source_info_dict, translate_path_fn):
    self._memoized_analyze_tensor_result = {}
    self._source_info_dict = source_info_dict
    self._translate_path_fn = translate_path_fn

  def _maybe_analyze_tensor(self, tensor_or_op, stack=None):
    """Implements memoization for graph analysis."""
    if tensor_or_op in self._memoized_analyze_tensor_result:
      return self._memoized_analyze_tensor_result[tensor_or_op]

    result = self._analyze_tensor_internal(tensor_or_op, stack)
    self._memoized_analyze_tensor_result[tensor_or_op] = result
    return result

  def _analyze_tensor_internal(self, tensor_or_op, stack):
    """Returns whether a given  `Tensor` or `Operation` is ready to run.

    Recursively computes whether a tensor or operation is ready to run, using
    `source_info_dict` as terminal nodes.  An error is thrown if a table or
    placeholder is reached: they must be set using source_info_dict.
    This function is memoized using the _maybe_analyze_tensor method to cache
    computed values, therefore it should not be called directly.
    Cycles are ignored (so a cycle is considered ready to run) and cycles are
    detected using `stack`.

    Args:
      tensor_or_op: A `Tensor` or `Operation`.
      stack: (optional) The tensors or operations that are on the stack (used to
          avoid cycles).

    Returns:
      An _AnalysisResult which includes whether this op or tensor is ready to
      run, and a path from it to it's sources.

    Raises:
      _UnexpectedTableError: If an initializable table op is encountered.
      _UnexpectedPlaceholderError: If a placeholder is encountered.
    """
    if tensor_or_op in self._source_info_dict:
      source_info = self._source_info_dict[tensor_or_op]
      # source_info.name may be None but that just means that it relies on an
      # output of a previous analyzer, so that's ok.
      return _AnalysisResult(
          is_ready_to_run=source_info.is_ready_to_run,
          path=self._translate_path_fn(source_info.name))

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
    parent_results = [
        self._maybe_analyze_tensor(parent, stack)
        for parent in parents
        if parent not in stack
    ]
    result = _AnalysisResult(
        all(res.is_ready_to_run for res in parent_results),
        self._translate_path_fn(
            tensor_or_op, parents=[res.path for res in parent_results]))
    assert tensor_or_op is stack.pop()

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
        self._maybe_analyze_tensor(component).is_ready_to_run
        for component in _decompose_tensor_or_sparse_tensor(tensor_or_op))

  def get_unique_path(self, tensor):
    """Gets the analyzed path from the tensor or op to its root(s).

    This path is defined recursively as:
      Path(root) := translate_path_fn(root)
      Path(x)    := translate_path_fn(
                            x,
                            [translate_path_fn(p) for p in parents(x)])

    When root is defined as a tensor that has no parents.

    Args:
      tensor: A `Tensor` for which a path should be computed.

    Returns:
      The result of translate_path_fn on the computed path as described above.
    """
    if not isinstance(tensor, tf.Tensor):
      raise TypeError('Expected Tensor got {} of type {}'.format(
          tensor, type(tensor)))
    return self._maybe_analyze_tensor(tensor).path


def _set_unique_value_in_dict(input_dict, key, value):
  assert value not in input_dict.values(), value
  input_dict[key] = value


class InitializableGraphAnalyzer(object):
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
    input_signature: A dict whose keys are strings and values are `Tensor`s or
      `SparseTensor`s.
    replaced_tensors_ready: a dict from `Tensor` to bool indicating whether a
        `Tensor` is ready in this phase.
    translate_path_fn: (Optional) A function with the signature:
        (identifier, optional(parents)) -> Any
        which will be used to construct a unique path for a given `Tensor`.

  Raises:
    ValueError: If unexpected placeholders or tables are encountered, or table
        initializers do not have the expected structure in the graph.
  """

  def __init__(self,
               graph,
               input_signature,
               replaced_tensors_ready,
               translate_path_fn=None):

    if translate_path_fn is None:
      translate_path_fn = lambda x, parents=None: None

    self._ready_table_initializers = []

    initial_source_infos_dict = self._make_source_infos_dict(
        {}, replaced_tensors_ready)

    # Determine which table initializers are ready, based on the replaced
    # tensors. Since no input tensors are fed during table initialization, we do
    # not set the value of any tensors in `input_signature`.
    graph_analyzer_for_table_init = _GraphAnalyzer(initial_source_infos_dict,
                                                   translate_path_fn)
    complete_source_info_dict = self._make_source_infos_dict(
        input_signature, replaced_tensors_ready)

    for table_init_op in graph.get_collection(
        tf.compat.v1.GraphKeys.TABLE_INITIALIZERS):
      source_info = self._get_table_init_op_source_info(
          table_init_op, graph_analyzer_for_table_init, translate_path_fn)

      # We are using the table init op information and the table op information,
      # since that is a unique description of the table op.
      table_op = table_init_op.inputs[0].op
      complete_source_info_dict[table_op] = source_info
      if source_info.is_ready_to_run:
        self._ready_table_initializers.append(table_init_op)

    # Now determine which tensors are ready to run once the table has been
    # initialized.
    self._graph_analyzer = _GraphAnalyzer(complete_source_info_dict,
                                          translate_path_fn)

  def _make_source_infos_dict(self, input_signature, replaced_tensors_ready):
    """Builds a dictionary from source tensors to _SourceInfos.

    This dictionary stores information about the sources of the graph.
    Each tensor in replaced_tensors_ready is a source whose readiness is known
    and has no name.  Each tensor (or component of a tensor) in input_signature
    is ready to run and has a name determined by the signature.

    Args:
      input_signature: A dict whose keys are strings and values are `Tensor`s or
        `SparseTensor`s.
      replaced_tensors_ready: a dict from `Tensor` to bool indicating whether a
        `Tensor` is ready in this phase.

    Returns:
      a dictionary from source tensors to _SourceInfos.
    """
    result = {}
    for tensor_or_op, is_ready in six.iteritems(replaced_tensors_ready):
      for component in _decompose_tensor_or_sparse_tensor(tensor_or_op):
        result[component] = _SourceInfo(is_ready, None)

    for name, tensor in input_signature.items():
      if isinstance(tensor, tf.SparseTensor):
        _set_unique_value_in_dict(result, tensor.indices,
                                  _SourceInfo(True, '{}$indices'.format(name)))
        _set_unique_value_in_dict(result, tensor.values,
                                  _SourceInfo(True, '{}$values'.format(name)))
        _set_unique_value_in_dict(
            result, tensor.dense_shape,
            _SourceInfo(True, '{}$dense_shape'.format(name)))
      else:
        _set_unique_value_in_dict(result, tensor,
                                  _SourceInfo(True, '{}$tensor'.format(name)))
    return result

  def _get_table_init_op_source_info(self, table_init_op, graph_analyzer,
                                     translate_path_fn):
    """Gets a _SourceInfo for a given table init op."""

    if table_init_op.type not in _TABLE_INIT_OP_TYPES:
      raise ValueError(
          'Table initializer {} did not have expected op type'.format(
              table_init_op))
    if not table_init_op.inputs:
      raise ValueError(
          'Table initializer {} did not have expected number if inputs '
          '(expected >= 1 inputs, got 0)'.format(table_init_op))
    table_op = table_init_op.inputs[0].op
    table_init_inputs = table_init_op.inputs[1:]
    try:
      ready = all(map(graph_analyzer.ready_to_run, table_init_inputs))
      path = translate_path_fn(
          table_op,
          parents=list(map(graph_analyzer.get_unique_path, table_init_inputs)))
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
    return _SourceInfo(ready, path)

  @property
  def ready_table_initializers(self):
    return self._ready_table_initializers

  def ready_to_run(self, tensor):
    """Determine if a given tensor or op is ready to run."""
    try:
      return self._graph_analyzer.ready_to_run(tensor)
    except _UnexpectedPlaceholderError as e:
      raise ValueError(
          'The tensor {} depended on a placeholder ({}) that was not in the '
          'input_signature.  This may have be caused by manually adding a '
          'placeholder to the graph'.format(tensor, e.tensor))
    except _UnexpectedTableError as e:
      raise ValueError(
          'The tensor {} depended on an initializable table ({}) that was not '
          'tracked by the graph analysis.  This may be caused by adding an '
          'initializable table without adding its initializer to the '
          'collection tf.GraphKeys.TABLE_INITIALIZERS'.format(tensor, e.op))

  def get_unique_path(self, tensor):
    """Gets the analyzed path from the tensor or op to its root(s).

    This path is defined recursively as:
      Path(source)          := translate_path_fn(name)
      Path(op)              := translate_path_fn(
                                    op,
                                    [translate_path_fn(p) for p in parents(op)])
      Path(internal_tensor) := translate_path_fn(
                                    internal_tensor,
                                    [translate_path_fn(parent_op)])

    When root is defined as a tensor that has no parents.

    Args:
      tensor: A `Tensor` for which a path should be computed.

    Returns:
      The result of translate_path_fn on the computed path as described above.
    """
    return self._graph_analyzer.get_unique_path(tensor)
