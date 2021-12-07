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

import collections
import copy
import hashlib
import itertools
from typing import Iterable, List, Mapping, Optional, Set, Union
import uuid
from absl import logging

import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import common_types
from tensorflow_transform import nodes
from tensorflow_transform import tf_utils
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import func_graph as tf_func_graph
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.util import object_identity
# pylint: enable=g-direct-tensorflow-import

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
    'InitializeTableFromDataset',
    'LookupTableImport',
    'LookupTableImportV2',
    # If a TF 2 SavedModel/Hub module with tables is loaded inside the
    # pre-processing fn, a StatefulPartitionedCall is added to the
    # TABLE_INITIALIZERS collection.
    'StatefulPartitionedCall',
]


def _decompose_tensor_or_op(tensor_or_op):
  """Yields the raw components of a `tf.CompositeTensor`.

  If tensor_or_op is a `tf.Operation`, or `tf.Tensor`, then
  _decompose_tensor_or_op will act as a pass through.

  Args:
    tensor_or_op: `tf.Tensor`, `tf.CompositeTensor`, or `tf.Operation`.

  Yields:
    A tf.Tensor or tf.Operation, depending on what tensor_or_op is.
  """
  if isinstance(tensor_or_op, composite_tensor.CompositeTensor):
    for component in tf.nest.flatten(tensor_or_op, expand_composites=True):
      yield component
  else:
    yield tensor_or_op


def retrieve_sources(sinks, ignore_control_dependencies=False):
  """Captures subgraph between sources and sinks.

  Walk a Graph backwards from `sinks` and return any sources encountered in the
  subgraph. This util is refactored from `_map_subgraph` in
  tensorflow/.../ops/op_selector.py.

  Arguments:
    sinks:  An iterable of Operations where the subgraph terminates.
    ignore_control_dependencies: (Optional) If `True`, ignore any
      `control_inputs` for all ops while walking the graph.

  Returns:
    The set of placeholders upon which `sinks` depend. This could also contain
    placeholders representing `captures` in the graph.
  """
  stop_at_tensors = object_identity.ObjectIdentitySet()
  ops_to_visit = object_identity.ObjectIdentitySet(sinks)
  visited_ops = object_identity.ObjectIdentitySet()
  potential_extra_sources = object_identity.ObjectIdentitySet()
  while ops_to_visit:
    op = ops_to_visit.pop()
    visited_ops.add(op)

    if op.type == 'Placeholder':
      potential_extra_sources.update(op.outputs)

    input_ops = [t.op for t in op.inputs if t not in stop_at_tensors]
    if not ignore_control_dependencies:
      input_ops = itertools.chain(input_ops, op.control_inputs)
    for input_op in input_ops:
      if input_op not in visited_ops:
        ops_to_visit.add(input_op)

  return potential_extra_sources


def get_func_graph_for_name(graph, func_name):
  """Returns the FuncGraph associated to the given func_name if possible."""
  outer_graph = graph
  while graph is not None:
    func = graph._get_function(str(func_name))  # pylint: disable=protected-access
    if func is not None:
      if hasattr(func, 'graph'):
        return func.graph
      # `outer_graph` may not be the same as `ops.get_default_graph()` e.g.
      # in the case of nested if ops or when the gradient is being computed
      # from inside a Defun. We build the `func_graph` with `outer_graph` as its
      # outer graph.
      with outer_graph.as_default():
        # This is a _DefinedFunction.
        func_graph = (
            function_def_to_graph.function_def_to_graph(func.definition))
      if func_graph is not None:
        return func_graph
    if hasattr(graph, 'outer_graph'):
      graph = graph.outer_graph
    else:
      raise ValueError(
          'Function {} does not exist in the graph.'.format(func_name))


class _UnexpectedPlaceholderError(Exception):

  def __init__(self, op, func_graph_name):
    tensor = op.outputs[0]
    msg = 'An unexpected placeholder was encountered ({})'.format(tensor)
    super().__init__(msg)
    self.tensor = tensor
    self.func_graph_name = func_graph_name


class _UnexpectedTableError(Exception):

  def __init__(self, op, func_graph_name):
    msg = 'An unexpected initializable table was encountered ({})'.format(op)
    super().__init__(msg)
    self.op = op
    self.func_graph_name = func_graph_name


def _reraise_unexpected_error(func):
  """A decorator that reraises certain exceptions with modified msg and type."""

  def wrapper(self, tensor_or_op):
    """Wrapper when calling func to re-raise exceptions."""
    try:
      return func(self, tensor_or_op)
    except _UnexpectedPlaceholderError as e:
      if e.func_graph_name:
        raise ValueError(
            'The tensor_or_op {} depended on a placeholder ({}) that is part '
            'of a tf.function graph ({}), this is not supported. This may be a '
            'result of calling a tf.Transform analyzer in a tf.function'
            ''.format(tensor_or_op, e.tensor, e.func_graph_name))
      else:
        raise ValueError(
            'The tensor_or_op {} depended on a placeholder ({}) that was not '
            'in the input_signature.  This may have be caused by manually '
            'adding a placeholder to the graph'.format(tensor_or_op, e.tensor))
    except _UnexpectedTableError as e:
      if e.func_graph_name:
        raise ValueError(
            'The tensor_or_op {} depended on an initializable table ({}) that '
            'is part of a tf.function graph ({}), this is not supported. This'
            ' may be a result of initializing a table in a tf.function'
            ''.format(tensor_or_op, e.op, e.func_graph_name))
      else:
        raise ValueError(
            'The tensor_or_op {} depended on an initializable table ({}) that '
            'was not tracked by the graph analysis.  This may be caused by '
            'adding an initializable table without adding its initializer to '
            'the collection tf.GraphKeys.TABLE_INITIALIZERS'.format(
                tensor_or_op, e.op))

  return wrapper


_AnalysisResult = tfx_namedtuple.namedtuple(
    '_AnalysisResult', ['is_ready_to_run', 'path', 'dependent_sources'])

_SourceInfo = tfx_namedtuple.namedtuple('_SourceInfo',
                                        ['is_ready_to_run', 'name'])


class _GraphAnalyzer:
  """Class that analyzes a graph to determine readiness of tensors."""

  def __init__(self, source_info_dict, translate_path_fn, graph):
    """Init method for _GraphAnalyzer.

    Args:
      source_info_dict: A dict from `Tensor Reference` or `Operation` to
        `_SourceInfo`.
      translate_path_fn: A function with the signature: (identifier, parents) ->
        Any which will be used to construct a unique path for a given `Tensor`.
      graph: A `tf.Graph` which the given tensors belong to.
    """
    self._memoized_analyze_tensor_result = {}
    self._source_info_dict = source_info_dict
    self._translate_path_fn = translate_path_fn
    self._graph = graph

  def _get_parents(self, tensor_or_op):
    """Get the parents of the given `tensor_or_op`."""
    if tf_utils.hashable_tensor_or_op(tensor_or_op) in self._source_info_dict:
      return []

    # func_graph_name is not None only if the graph is a FuncGraph.
    func_graph_name = getattr(self._graph, 'name', None)
    if isinstance(tensor_or_op, tf.Operation):
      if tensor_or_op.type in _INITIALIZABLE_TABLE_OP_TYPES:
        raise _UnexpectedTableError(tensor_or_op, func_graph_name)
      if tensor_or_op.type == 'Placeholder':
        # If we're not in the context of a tf.function, this is an error.
        if func_graph_name is None:
          raise _UnexpectedPlaceholderError(tensor_or_op, func_graph_name)
        # If we're in the context of a tf.function and this op is part of its
        # inputs, that's expected.
        if tensor_or_op not in [x.op for x in self._graph.inputs]:
          raise _UnexpectedPlaceholderError(tensor_or_op, func_graph_name)
      parents = list(
          itertools.chain(tensor_or_op.inputs, tensor_or_op.control_inputs))
    elif isinstance(tensor_or_op, tf.Tensor):
      parents = [tensor_or_op.op]
    else:
      raise TypeError('Expected Tensor or Operation, got {} of type {}'.format(
          tensor_or_op, type(tensor_or_op)))
    return parents

  def _compute_analysis_results_for_func_attributes(self, tensor_or_op,
                                                    parent_analysis_results):
    """Analyzes `FuncGraph`s if tensor_or_op has them as attributes.

    This functionality is added to support `Operation`s such as PartitionedCall
    (tf.function call) and control flow ops which use `func` attributes.

    These func attributes are references to `FuncGraph`s which can also be
    analyzed, and the result of their analysis can be used as additional
    information for the current node (`tensor_or_op`).

    Since `FuncGraph`s are completely different graphs than the one that this
    _GraphAnalyzer is analyzing, their analysis wouldn't be taken into account
    when analysing the current graph even though they will affect the runtime
    results of running it. This is why we have to manually analyze those
    sub-graphs as well as the main graph when computing graph information such
    as dependent_inputs, unique_path, etc.

    Args:
      tensor_or_op: A `Tensor` or `Operation` object.
      parent_analysis_results: A list of `_AnalysisResult`s, results of analysis
        of the parents of tensor_or_op.

    Returns:
      A list of `_AnalysisResult`s, the results of analysis of `tensor_or_op`'s
      func attributes. All `Tensor`s in dependent_sources belong to self._graph.
    """
    if not isinstance(tensor_or_op, tf.Operation):
      return []
    func_attributes = [
        attr.name for attr in tensor_or_op.op_def.attr if attr.type == 'func'
    ]
    func_names = [tensor_or_op.get_attr(str(n)).name for n in func_attributes]
    func_graphs = [get_func_graph_for_name(self._graph, n) for n in func_names]

    result = []
    for func_graph in func_graphs:
      if not hasattr(func_graph, 'inputs'):
        # Since the body of the graph is not visible we insert a random string
        # to the path in order to reflect that we don't know its full contents.
        result.append(
            _AnalysisResult(
                is_ready_to_run=True,
                path=self._translate_path_fn(uuid.uuid4().hex),
                dependent_sources={}))
        continue
      op_inputs = list(
          itertools.chain(tensor_or_op.inputs, tensor_or_op.control_inputs))
      assert len(op_inputs) == len(parent_analysis_results), (
          op_inputs, parent_analysis_results)
      func_graph_inputs_ready = [
          (next_input, r.is_ready_to_run)
          for (next_input, r) in zip(func_graph.inputs, parent_analysis_results)
      ]
      infos = {
          tf_utils.hashable_tensor_or_op(t):
          _SourceInfo(ready, 'FuncGraphInput[{}]'.format(idx))
          for idx, (t, ready) in enumerate(func_graph_inputs_ready)
      }
      func_graph_analyzer = _GraphAnalyzer(infos, self._translate_path_fn,
                                           func_graph)
      analyzed_list = [
          func_graph_analyzer.analyze_tensor(t) for t in func_graph.outputs
      ]

      if len(tensor_or_op.inputs) == len(func_graph.inputs):
        tensor_pairs = zip(tensor_or_op.inputs, func_graph.inputs)
      else:
        # Control flow ops such as while store this information in captures.
        tensor_pairs = func_graph.captures
      tensor_map = {
          tf_utils.hashable_tensor_or_op(b): a for a, b in tensor_pairs
      }

      # Make sure that the dependent sources Tensors are translated from the
      # FuncGraph to the outer graph in order to align with the rest of the
      # traversal.
      for analysis in analyzed_list:
        translated_dependent_sources = {
            tf_utils.hashable_tensor_or_op(tensor_map[s])
            for s in analysis.dependent_sources
            if s in tensor_map
        }
        result.append(
            analysis._replace(dependent_sources=translated_dependent_sources))
    return result

  def _compute_analysis_result(self, tensor_or_op, parent_analysis_results):
    """Compute analysis result for a tensor or op with its parent results."""
    hashable = tf_utils.hashable_tensor_or_op(tensor_or_op)
    if hashable in self._source_info_dict:
      source_info = self._source_info_dict[hashable]
      # source_info.name may be None but that just means that it relies on an
      # output of a previous analyzer, so that's ok.
      return _AnalysisResult(
          is_ready_to_run=source_info.is_ready_to_run,
          path=self._translate_path_fn(source_info.name),
          dependent_sources={hashable})

    func_graphs_analysis_results = (
        self._compute_analysis_results_for_func_attributes(
            tensor_or_op, parent_analysis_results))

    result = _AnalysisResult(
        is_ready_to_run=all(
            analysis_result.is_ready_to_run
            for analysis_result in (parent_analysis_results +
                                    func_graphs_analysis_results)),
        path=self._translate_path_fn(
            tensor_or_op,
            parents=[
                parent_analysis_result.path
                for parent_analysis_result in parent_analysis_results
            ] +
            [func_result.path for func_result in func_graphs_analysis_results]),
        dependent_sources=set())
    for parent_analysis_result in parent_analysis_results:
      result.dependent_sources.update(parent_analysis_result.dependent_sources)
    for func_result in func_graphs_analysis_results:
      result.dependent_sources.update(func_result.dependent_sources)
    return result

  def analyze_tensor(self, tensor_or_op):
    """Analyzes the `tensor_or_op` for its dependencies and readiness.

    Computes the transitive dependencies of a tensor or operation and decides
    whether it is ready to run using iterative DFS. `source_info_dict` are used
    as terminal nodes.  An error is thrown if a table or placeholder is reached:
    they must be set using source_info_dict. This function is memoized using the
    _memoized_analyze_tensor_result cache. Cycles are ignored (so a cycle is
    considered ready to run).

    Args:
      tensor_or_op: A `Tensor` or `Operation`.

    Returns:
      An _AnalysisResult which includes whether this op or tensor is ready to
      run, a path from it to its sources and its dependent sources from
      `source_info_dict`.

    Raises:
      _UnexpectedTableError: If an initializable table op is encountered.
      _UnexpectedPlaceholderError: If a placeholder is encountered.
    """
    stack = collections.deque()
    # Note that because tensors are no longer hashable, we need to convert to
    # their reference in order to use them in sets or dicts.
    stack.append(tf_utils.hashable_tensor_or_op(tensor_or_op))
    # Contains the nodes of the path starting from tensor_or_op to current
    # visiting node, used for loop detection. We assume that any loop is a
    # valid while loop and so it will be able to run as long as all the other
    # parents are ready.
    path = set()
    while stack:
      current = stack[-1]
      if current in self._memoized_analyze_tensor_result:
        stack.pop()
        continue
      path.add(current)
      parents = self._get_parents(tf_utils.deref_tensor_or_op(current))
      parents = [parent for parent in map(tf_utils.hashable_tensor_or_op,
                                          parents) if parent not in path]
      if all(
          parent in self._memoized_analyze_tensor_result for parent in parents):
        parent_results = [
            self._memoized_analyze_tensor_result[parent] for parent in parents
        ]
        current_result = self._compute_analysis_result(
            tf_utils.deref_tensor_or_op(current), parent_results)
        self._memoized_analyze_tensor_result[current] = current_result
        path.discard(stack.pop())
      else:
        stack.extend(parents)
    return self._memoized_analyze_tensor_result[tf_utils.hashable_tensor_or_op(
        tensor_or_op)]

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
      tensor_or_op: A `Tensor`, `SparseTensor`, `RaggedTensor` or `Operation`

    Returns:
      A bool indicating whether then tensor is ready to run.

    Raises:
      ValueError: If a placeholder or table is encountered.
      _UnexpectedTableError: If an initializable table op is encountered.
      _UnexpectedPlaceholderError: If a placeholder is encountered.
    """
    if not isinstance(
        tensor_or_op,
        (tf.Tensor, tf.SparseTensor, tf.RaggedTensor, tf.Operation)):
      raise TypeError(
          'Expected Tensor, SparseTensor, RaggedTensor, or Operation got {} of type {}'
          .format(tensor_or_op, type(tensor_or_op)))
    return all(
        self.analyze_tensor(component).is_ready_to_run
        for component in _decompose_tensor_or_op(tensor_or_op))

  def get_unique_path(self, tensor):
    """Gets the analyzed path from the tensor to its root(s).

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

    Raises:
      TypeError: if the given tensor is not of type `Tensor`
      _UnexpectedTableError: If an initializable table op is encountered.
      _UnexpectedPlaceholderError: If a placeholder is encountered.
    """
    if not isinstance(tensor, tf.Tensor):
      raise TypeError('Expected Tensor got {} of type {}'.format(
          tensor, type(tensor)))
    return self.analyze_tensor(tensor).path


def _set_unique_value_in_dict(input_dict, key, value):
  assert value not in input_dict.values(), value
  input_dict[tf_utils.hashable_tensor_or_op(key)] = value


class InitializableGraphAnalyzer:
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
  """

  def __init__(self,
               graph,
               input_signature,
               replaced_tensors_ready,
               translate_path_fn=None):
    """Init method for InitializableGraphAnalyzer.

    Args:
      graph: a `Graph`.
      input_signature: A dict whose keys are strings and values are `Tensor`s,
        `SparseTensor`s, or `RaggedTensor`s.
      replaced_tensors_ready: a list of `Tensor`, `SparseTensor`s, or
        `RaggedTensor`s, bool pairs indicating whether the `Tensor`,
        `SparseTensor`s, or `RaggedTensor`s is ready in this phase.
      translate_path_fn: (Optional) A function with the signature: (identifier,
        optional(parents)) -> Any which will be used to construct a unique path
        for a given `Tensor`.

    Raises:
      ValueError: If unexpected placeholders or tables are encountered, or table
          initializers do not have the expected structure in the graph.
    """

    if translate_path_fn is None:
      translate_path_fn = lambda x, parents=None: None

    self._ready_table_initializers = []
    self._input_signature = input_signature
    replaced_tensors_ready = {tf_utils.hashable_tensor_or_op(t): ready
                              for t, ready in replaced_tensors_ready}

    initial_source_infos_dict = self._make_source_infos_dict(
        {}, replaced_tensors_ready)

    # Determine which table initializers are ready, based on the replaced
    # tensors. Since no input tensors are fed during table initialization, we do
    # not set the value of any tensors in `input_signature`.
    graph_analyzer_for_table_init = _GraphAnalyzer(initial_source_infos_dict,
                                                   translate_path_fn, graph)
    complete_source_info_dict = self._make_source_infos_dict(
        input_signature, replaced_tensors_ready)

    for table_init_op_or_tensor in graph.get_collection(
        tf.compat.v1.GraphKeys.TABLE_INITIALIZERS):
      # Handle the case when an initializer was lifted out of the graph context.
      if table_init_op_or_tensor is None:
        continue

      if isinstance(graph, tf_func_graph.FuncGraph):
        self._log_warning('Tables initialized inside a tf.function  will be'
                          ' re-initialized on every invocation of the function.'
                          ' This  re-initialization can have significant impact'
                          ' on performance. Consider lifting  them out of the'
                          ' graph context using  `tf.init_scope`.: {}'.format(
                              table_init_op_or_tensor.name))

      table_init_op, table_input_ops = (
          self._get_table_init_op_and_inputs(table_init_op_or_tensor))
      source_info = self._get_table_init_op_source_info(
          table_init_op, graph_analyzer_for_table_init, translate_path_fn)

      for key in table_input_ops:
        complete_source_info_dict[tf_utils.hashable_tensor_or_op(
            key)] = source_info
      if source_info.is_ready_to_run:
        self._ready_table_initializers.append(table_init_op_or_tensor)

    # Now determine which tensors are ready to run once the table has been
    # initialized.
    self._graph_analyzer = _GraphAnalyzer(complete_source_info_dict,
                                          translate_path_fn, graph)

  def _log_warning(self, message: str):
    logging.warning(message)

  def _get_table_init_op_and_inputs(self, table_init_op_or_tensor):
    """Get a tuple of table init op and keys for its input ops."""
    # If a TF2 exported SavedModel with a table is loaded inside the
    # preprocessing_fn, the TABLE_INITIALIZERS collection of the outer graph
    # contains a Tensor whose parent op is of type StatefulPartitionedCall.
    # The nested func graph for this StatefulPartitionedCall contains the
    # table initializer.
    if (isinstance(table_init_op_or_tensor, tf.Tensor) and
        table_init_op_or_tensor.op.type == 'StatefulPartitionedCall'):
      result = (table_init_op_or_tensor.op,
                [input_t.op for input_t in table_init_op_or_tensor.op.inputs])
    else:
      assert isinstance(table_init_op_or_tensor, tf.Operation)
      # We are using the table init op information and the table op information,
      # since that is a unique description of the table op.
      table_ops = []
      for input_t in table_init_op_or_tensor.inputs:
        # One of the inputs to the initializer op should be the table op. If
        # no table op is found, (as in the case of a StatefulPartitionedCall)
        # all inputs are added to the source dict.
        if input_t.dtype == tf.resource:
          table_ops.append(input_t.op)
      assert len(table_ops) == 1
      result = (table_init_op_or_tensor, [table_ops[0]])
    return result

  def _make_source_infos_dict(self, input_signature, replaced_tensors_ready):
    """Builds a dictionary from source tensors to _SourceInfos.

    This dictionary stores information about the sources of the graph.
    Each tensor in replaced_tensors_ready is a source whose readiness is known
    and has no name.  Each tensor (or component of a tensor) in input_signature
    is ready to run and has a name determined by the signature.

    Args:
      input_signature: A dict whose keys are strings and values are `Tensor`s,
        `SparseTensor`s, or `RaggedTensor`s.
      replaced_tensors_ready: a dict from `Tensor`, `SparseTensor`s, or
        `RaggedTensor`s to bool indicating whether the tensor is ready in this
        phase.

    Returns:
      a dictionary from source tensors to _SourceInfos.
    """
    result = {}
    for tensor_or_op, is_ready in replaced_tensors_ready.items():
      for component in _decompose_tensor_or_op(
          tf_utils.deref_tensor_or_op(tensor_or_op)):
        result[tf_utils.hashable_tensor_or_op(component)] = _SourceInfo(
            is_ready, None)

    for name, tensor in input_signature.items():
      if isinstance(tensor, tf.Tensor):
        _set_unique_value_in_dict(result, tensor,
                                  _SourceInfo(True, '{}$tensor'.format(name)))
      elif isinstance(tensor, composite_tensor.CompositeTensor):
        for idx, tensor_component in enumerate(_decompose_tensor_or_op(tensor)):
          _set_unique_value_in_dict(
              result, tensor_component,
              _SourceInfo(True, '{}$composite_tensor_{}'.format(name, idx)))
      else:
        raise TypeError(
            'Expected Tensor, or CompositeTensor, got {} of type {}'.format(
                tensor, type(tensor)))
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
      if e.func_graph_name:
        raise e
      raise ValueError(
          'The table initializer {} depended on a placeholder ({}).  Note '
          'placeholders will not be fed during table initialization'.format(
              table_init_op, e.tensor))
    except _UnexpectedTableError as e:
      if e.func_graph_name:
        raise e
      raise ValueError(
          'The table initializer {} depended on an initializable table ({}). '
          'Note tables are initialized in one pass so a table initializer '
          'cannot depend on the output of an initializeable table'.format(
              table_init_op, e.op))
    return _SourceInfo(ready, path)

  @property
  def ready_table_initializers(self):
    return self._ready_table_initializers

  @_reraise_unexpected_error
  def ready_to_run(self, tensor_or_op):
    """Determine if a given tensor or op is ready to run."""
    return self._graph_analyzer.ready_to_run(tensor_or_op)

  @_reraise_unexpected_error
  def get_unique_path(self, tensor):
    """Gets the analyzed path from the tensor to its root(s).

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
    return self._graph_analyzer.get_unique_path(tensor)

  @_reraise_unexpected_error
  def get_dependent_inputs(self, tensor_or_op):
    """Gets the inputs that the given `tensor_or_op` transitively depends on.

    Args:
      tensor_or_op: A `Tensor`, `SparseTensor`, `RaggedTensor` or `Operation`.

    Returns:
      A dict of name to `Tensor`, `SparseTensor`, or `RaggedTensor` (sub-dict of
      `input_signature`) that the given `tensor_or_op` depends on.

    Raises:
      TypeError: If `tensor_or_op` is of an unsupported type.
    """
    if not isinstance(
        tensor_or_op,
        (tf.Tensor, tf.SparseTensor, tf.RaggedTensor, tf.Operation)):
      raise TypeError(
          'Expected Tensor, SparseTensor, RaggedTensor or Operation got {} of '
          'type {}'.format(tensor_or_op, type(tensor_or_op)))

    dependents = set()
    for component in _decompose_tensor_or_op(tensor_or_op):
      dependents.update(
          self._graph_analyzer.analyze_tensor(component).dependent_sources)

    result = {}
    for name, tensor in self._input_signature.items():
      if any(
          tf_utils.hashable_tensor_or_op(component) in dependents
          for component in _decompose_tensor_or_op(tensor)):
        result[name] = tensor
    return result


class _QuietInitializableGraphAnalyzer(InitializableGraphAnalyzer):
  """A `InitializableGraphAnalyzer` which doesn't log any warnings."""

  def _log_warning(self, message: str):
    pass


def get_dependent_inputs(graph, input_tensors, output_tensors):
  """Returns tensors in input_tensors that (transitively) produce output_tensors.

  Args:
    graph: A `tf.Graph`. It could be the (intermediate) output tf graph in any
      transform phase (including phase 0 where no tensor replacement has yet
      happened).
    input_tensors: A dict of logical name to `tf.Tensor`, `tf.SparseTensor`, or
      `tf.RaggedTensor`. Logical name doesn't have any implications in this
      method and can be anything. In some cases it is the feature name
      corresponding to the input tensor.
    output_tensors: A dict of logical name to `tf.Tensor`, `tf.SparseTensor`, or
      `tf.RaggedTensor`, or a list of `tf.Tensor`, `tf.SparseTensor`, or
      `tf.RaggedTensor`.

  Returns:
    A dict of logical name to `tf.Tensor`, `tf.SparseTensor`, or
    `tf.RaggedTensor` that are filtered from input_tensors (transitively)
    producing output_tensors
  """
  if isinstance(output_tensors, list):
    output_container = output_tensors
  else:
    output_container = output_tensors.values()

  # Since this method may be called before all tensor replacements are ready, to
  # fulfill the precondition of InitializableGraphAnalyzer, we fake the
  # readiness of tensor replacements. Note that the readiness of replacement
  # tensors doesn't affect the correctness of dependencies tracing.
  tensor_sinks = graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  sink_tensors_ready = [(sink.tensor, False) for sink in tensor_sinks]
  graph_analyzer = _QuietInitializableGraphAnalyzer(graph, input_tensors,
                                                    sink_tensors_ready)
  dependent_inputs = {}
  for output_tensor in output_container:
    dependent_inputs.update(graph_analyzer.get_dependent_inputs(output_tensor))
  return {
      name: tensor
      for name, tensor in input_tensors.items()
      if name in dependent_inputs
  }


def _serialize_op_attr(op_attr):
  """Deterministicly serializes tf.Operation attrs since it is a map."""
  sorted_attributes = sorted(op_attr.items(), key=lambda kv: kv[0])
  if 'f' in op_attr:
    # This is a tf.Function node, and it includes attributes that are
    # inconsistent across runs such as _gradient_op_type, config_proto, so we
    # only keep input and output types since other information will arrive from
    # the FuncGraph attributes.
    sorted_attributes = [
        kv for kv in sorted_attributes if kv[0] in ('Tin', 'Tout')
    ]
  result = []
  for key, attr_value in sorted_attributes:
    result.append(key)
    attr_value = copy.deepcopy(attr_value)
    if attr_value.list.func:
      raise ValueError(
          'Unable to serialize op attributes that contain a `list.func` field')
    if attr_value.HasField('func'):
      # There should be a separate call for the FuncGraph attributes.
      attr_value.ClearField('func')
    result.append(attr_value.SerializeToString())
  return result


def describe_path_as_analyzer_cache_hash(
    x: Optional[Union[tf.Operation, tf.Tensor, str]],
    parents: Optional[List[bytes]] = None) -> Optional[bytes]:
  """Constructs a hash to describe a unique TF graph path.

  Note: We do not rely on names for hashing since it can be fragile.

  Args:
    x: The current TF graph node.
    parents: Results of previous calls to this function, where x was an ancestor
      to the current node x.

  Returns:
    A bytes hash of the path from x to its sources. None if x is None.
  """
  # This may happen in cases where tensors are outputs of previous analyzers,
  # we don't need to describe a path for those.
  if x is None:
    assert parents is None
    return None
  parents = parents or []
  if any(p is None for p in parents):
    return None

  if isinstance(x, tf.Operation):
    values = _serialize_op_attr(x.node_def.attr)
  elif isinstance(x, tf.Tensor):
    # No need to add x.op to the hash since that should be included in parents.
    values = [tf.compat.as_str_any(x.value_index)]
  else:
    assert isinstance(x, (str, bytes))
    values = [x]

  h = hashlib.sha1()
  for value in values:
    encoded = tf.compat.as_bytes(value)
    h.update(encoded)

  for p in parents:
    h.update(p)

  return h.digest()


class SourcedTensorsVisitor(nodes.Visitor):
  """Visitor used to extract tensors that are inputs to `TensorSource` nodes."""

  def __init__(self):
    self.sourced_tensors = []

  def visit(self, operation_def, input_values):
    if isinstance(operation_def, analyzer_nodes.TensorSource):
      for tensor in operation_def.tensors:
        self.sourced_tensors.append(tensor)
    return nodes.OperationNode(operation_def, input_values).outputs

  def validate_value(self, value):
    assert isinstance(value, nodes.ValueNode)


def _retrieve_source_keys(
    sourced_tensors: Iterable[tf.Tensor],
    structured_inputs: Mapping[str, common_types.TensorType]) -> Set[str]:
  """Retrieve input keys that sourced_tensors depend on."""
  result = set()
  sinks = [t.op for t in sourced_tensors]
  sources = retrieve_sources(sinks, ignore_control_dependencies=True)
  hashable_sources = [tf_utils.hashable_tensor_or_op(s) for s in sources]
  for key, value in structured_inputs.items():
    components = ([
        tf_utils.hashable_tensor_or_op(v)
        for v in _decompose_tensor_or_op(value)
    ])
    if any([s in components for s in hashable_sources]):
      result.add(key)
  return result


AnalyzersFingerprint = tfx_namedtuple.TypedNamedTuple(
    'AnalyzersFingerprint', [('source_keys', Set[str]),
                             ('unique_path_hash', Set[bytes])])


def get_analyzers_fingerprint(
    graph: tf.Graph, structured_inputs: Mapping[str, common_types.TensorType]
) -> Mapping[str, AnalyzersFingerprint]:
  """Computes fingerprints for all analyzers in `graph`.

  Args:
    graph: a TF Graph.
    structured_inputs: a dict from keys to batches of placeholder graph tensors.

  Returns:
    A mapping from analyzer name to a set of paths that define its fingerprint.
  """
  result = {}
  tensor_sinks = graph.get_collection(analyzer_nodes.ALL_REPLACEMENTS)
  # The value for the keys in this dictionary are unused and can be arbitrary.
  sink_tensors_ready = {
      tf_utils.hashable_tensor_or_op(tensor_sink.tensor): False
      for tensor_sink in tensor_sinks
  }
  graph_analyzer = InitializableGraphAnalyzer(
      graph, structured_inputs, list(sink_tensors_ready.items()),
      describe_path_as_analyzer_cache_hash)
  for tensor_sink in tensor_sinks:
    # Retrieve tensors that are inputs to the analyzer's value node.
    visitor = SourcedTensorsVisitor()
    nodes.Traverser(visitor).visit_value_node(tensor_sink.future)
    source_keys = _retrieve_source_keys(visitor.sourced_tensors,
                                        structured_inputs)
    paths = set()
    for tensor in visitor.sourced_tensors:
      # Obtain fingerprint for each tensor that is an input to the value node.
      path = graph_analyzer.get_unique_path(tensor)
      if path is not None:
        paths.add(path)
    result[str(tensor_sink.tensor.name)] = AnalyzersFingerprint(
        source_keys, paths)
  return result
