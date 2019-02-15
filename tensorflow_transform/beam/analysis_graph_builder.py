# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Functions to create the implementation graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

# GOOGLE-INITIALIZATION

import tensorflow as tf
from tensorflow_transform import analyzer_cache
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import graph_tools
from tensorflow_transform import nodes
from tensorflow_transform.beam import beam_nodes


def _tensor_name(tensor):
  """Get a name of a tensor without trailing ":0" when relevant."""
  # tensor.name is unicode in Python 3 and bytes in Python 2 so convert to
  # bytes here.
  name = str(tensor.name)
  return name[:-2] if name.endswith(':0') else name


class _ReadyVisitor(nodes.Visitor):
  """Visitor to determine if a node is ready to run."""

  def __init__(self, graph_analyzer):
    self._graph_analyzer = graph_analyzer

  def visit(self, operation_def, input_values):
    if isinstance(operation_def, analyzer_nodes.TensorSource):
      is_ready = all(self._graph_analyzer.ready_to_run(tensor)
                     for tensor in operation_def.tensors)
    else:
      is_ready = all(input_values)
    return (is_ready,) * operation_def.num_outputs

  def validate_value(self, value):
    assert isinstance(value, bool)


class _TranslateVisitor(nodes.Visitor):
  """Visitor that translates the operation graph.

  The original graph is defined by the user in the preprocessing_fn.  The
  translated graph represents a Beam pipeline.
  """

  def __init__(self):
    self.phase = None
    self.extracted_values_dict = None
    self.intermediate_output_signature = None

  def visit(self, operation_def, input_values):
    if isinstance(operation_def, analyzer_nodes.TensorSource):
      tensors = operation_def.tensors
      label = operation_def.label
      # Add tensor to signature so it gets produced by the SavedModel.
      for tensor in tensors:
        self.intermediate_output_signature[_tensor_name(tensor)] = tensor
      keys = tuple(map(_tensor_name, tensors))
      output = nodes.apply_operation(
          beam_nodes.ExtractFromDict, self.extracted_values_dict,
          keys=keys, label=label)
      return (output,)
    else:
      return nodes.OperationNode(operation_def, input_values).outputs

  def validate_value(self, value):
    assert isinstance(value, nodes.ValueNode)


class _OptimizationView(
    collections.namedtuple(
        '_OptimizationView',
        ['prefer_fine_grained_view', 'flattened_view', 'fine_grained_view'])):
  """A container for operation outputs during _OptimizeVisitor traversal.

  This is used in order to maintain both a flattened view, and a fine grained
  view that can be used for caching.

  `prefer_fine_grained_view` is a hint that means that if True, the
  `fine_grained_view` should be used.  It should be set to true if the upstream
  view has cacheing operations that haven't been flattened yet.
  """

  def __init__(self, prefer_fine_grained_view, flattened_view,
               fine_grained_view):
    if prefer_fine_grained_view and not fine_grained_view:
      raise ValueError(
          'Cannot prefer fine_grained_view when one is not provided')
    self._validate_flattened_view(flattened_view)
    self._validate_fine_grained_view(fine_grained_view)
    super(_OptimizationView, self).__init__()

  def __str__(self):
    return '{}[{}]'.format(self.parent_operation.operation_def.label,
                           self.value_index)

  def _validate_flattened_view(self, view):
    assert view is self.flattened_view
    assert view is not None
    assert isinstance(view, nodes.ValueNode), view

  def _validate_fine_grained_view(self, view):
    assert view is self.fine_grained_view
    if view is None:
      return
    assert isinstance(view, collections.OrderedDict), view
    for value in view.values():
      assert isinstance(value, nodes.ValueNode), value


class _OptimizeVisitor(nodes.Visitor):
  """Visitor optimizes the operation graph (see nodes.py).

  This operates on the translated graph which is emitted by the
  `_TranslateVisitor`, and performs optimizations.

  Namely, when enabled, this enables reading and writing from/to analyzer
  accumulator cache to avoid recomputing them over already seen datasets.
  This type of optimization requires also creating a partitioned view of the
  input data, according to the `is_partitionable` annotation.
  """

  def __init__(self, dataset_keys, cache_location):
    self._dataset_keys = sorted(dataset_keys)
    self._cache_location = cache_location
    self._flattened_apply_saved_model = None

  def _validate_operation_def(self, operation_def):
    if operation_def.cache_coder is not None:
      if not operation_def.is_partitionable:
        raise ValueError('Non partitionable OperationDefs cannot be cacheable')
    if operation_def.is_partitionable or operation_def.cache_coder is not None:
      if operation_def.num_outputs != 1:
        raise ValueError('Cacheable OperationDefs must have exactly 1 output')

  def visit(self, operation_def, input_values):
    self._validate_operation_def(operation_def)

    # TODO(b/37788560): Possibly make this generic instead of special casing the
    # ApplySavedModel operation.
    if (isinstance(operation_def, beam_nodes.ApplySavedModel) and
        operation_def.phase == 0):
      return self._visit_apply_savedmodel_operation(operation_def, input_values)

    if self._cache_location and operation_def.is_partitionable:
      return self._visit_partitionable_operation(operation_def, input_values)

    if input_values and any(v.fine_grained_view and v.prefer_fine_grained_view
                            for v in input_values):
      # We can 'flatten' the cached outputs of the parent operation since this
      # operation doesn't support partitioning.
      disaggregated_input_values = []
      for view in input_values:
        disaggregated_input_values.extend(view.fine_grained_view.values())

      # Checking that all cache has the same size.
      assert len({len(value) for value in disaggregated_input_values}) == 1

      next_inputs = nodes.apply_multi_output_operation(
          beam_nodes.Flatten,
          *disaggregated_input_values,
          label='FlattenCache[{}]'.format(operation_def.label))
    else:
      # Parent operation output is not cacheable, therefore we can just use
      # a flattened view.
      next_inputs = tuple(v.flattened_view for v in input_values)

    flattened_view = nodes.OperationNode(operation_def, next_inputs).outputs

    return tuple(
        _OptimizationView(
            prefer_fine_grained_view=False,
            flattened_view=flat,
            fine_grained_view=None) for flat in flattened_view)

  def _visit_partitionable_operation(self, operation_def, upstream_views):
    # TODO(b/37788560) Possibly support partitionable operations with multiple
    # inputs.
    (upstream_view,) = upstream_views
    prefer_fine_grained_view = (
        upstream_view.prefer_fine_grained_view or
        upstream_view.fine_grained_view and
        operation_def.cache_coder is not None)

    if upstream_view.fine_grained_view:
      value_nodes = collections.OrderedDict()
      for key in self._dataset_keys:

        if operation_def.cache_coder is not None:
          # TODO(b/37788560): Add instrumentation.
          # TODO(b/37788560): Use a better cache key than label. A good
          # alternative is to reuse graph_tools logic to compose names that
          # include properties and fingerprint it.
          cache_file_path = analyzer_cache.make_cache_file_path(
              key, operation_def.label)
          # TODO(b/37788560): Come up with a more abstract way to do this that
          # also ensures concistency.
          pattern = '{}-00000*.gz'.format(
              os.path.join(self._cache_location.input_cache_dir,
                           cache_file_path))
          try:
            if tf.gfile.Glob(pattern):
              op_outputs = nodes.apply_multi_output_operation(
                  analyzer_nodes.ReadCache,
                  path=cache_file_path,
                  coder=operation_def.cache_coder,
                  label='ReadCache[{}][{}]'.format(operation_def.label, key))
              value_nodes[key] = op_outputs
              continue
          except tf.errors.NotFoundError:
            pass
        else:
          cache_file_path = None

        values = upstream_view.fine_grained_view[key]
        op_outputs = nodes.OperationNode(
            operation_def._replace(
                label='{}[{}]'.format(operation_def.label, key)),
            (values,)).outputs
        if cache_file_path is not None:
          op_outputs = nodes.apply_multi_output_operation(
              analyzer_nodes.WriteCache,
              *op_outputs,
              path=cache_file_path,
              coder=operation_def.cache_coder,
              label='WriteCache[{}][{}]'.format(operation_def.label, key))
        value_nodes[key] = op_outputs

      fine_grained_views = (
          [collections.OrderedDict()] * operation_def.num_outputs)
      for key in self._dataset_keys:
        for idx in range(operation_def.num_outputs):
          fine_grained_views[idx][key] = value_nodes[key][idx]
    else:
      fine_grained_views = (None,) * operation_def.num_outputs

    flattened_views = nodes.OperationNode(
        operation_def, (upstream_view.flattened_view,)).outputs

    return tuple(
        _OptimizationView(
            prefer_fine_grained_view=prefer_fine_grained_view,
            flattened_view=flat,
            fine_grained_view=fine)
        for flat, fine in zip(flattened_views, fine_grained_views))

  def _visit_apply_savedmodel_operation(self, operation_def, upstream_views):
    (upstream_view,) = upstream_views
    if upstream_view.fine_grained_view:
      raise ValueError(
          'Was not expecting a fine_grained_view input for ApplySavedModel')

    fine_grained_view = collections.OrderedDict()
    for key in self._dataset_keys:
      (fine_grained_view[key],) = (
          nodes.OperationNode(
              operation_def._replace(
                  dataset_key=key,
                  label='{}[{}]'.format(operation_def.label, key)),
              (upstream_view.flattened_view,)).outputs)

    (flattened_view,) = nodes.OperationNode(
        operation_def, (upstream_view.flattened_view,)).outputs

    return (_OptimizationView(
        prefer_fine_grained_view=False,
        flattened_view=flattened_view,
        fine_grained_view=fine_grained_view),)

  def validate_value(self, value):
    assert isinstance(value, _OptimizationView), value
    if value.fine_grained_view:
      assert set(value.fine_grained_view.keys()) == set(
          self._dataset_keys), ('{} != {}'.format(
              value.fine_grained_view.keys(), self._dataset_keys))


def _perform_cache_optimization(saved_model_future, cache_location,
                                dataset_keys):
  optimize_visitor = _OptimizeVisitor(dataset_keys or {}, cache_location)
  optimize_traverser = nodes.Traverser(optimize_visitor)
  return optimize_traverser.visit_value_node(saved_model_future).flattened_view


def build(graph,
          input_signature,
          output_signature,
          dataset_keys=None,
          cache_location=None):
  """Returns a list of `Phase`s describing how to execute the pipeline.

  The default graph is assumed to contain some `Analyzer`s which must be
  executed by doing a full pass over the dataset, and passing the inputs for
  that analyzer into some implementation, then taking the results and replacing
  the `Analyzer`s outputs with constants in the graph containing these results.

  The execution plan is described by a list of `Phase`s.  Each phase contains
  a list of `Analyzer`s, which are the `Analyzer`s which are ready to run in
  that phase, together with a list of ops, which are the table initializers that
  are ready to run in that phase.

  An `Analyzer` or op is ready to run when all its dependencies in the graph
  have been computed.  Thus if the graph is constructed by

  def preprocessing_fn(input)
    x = inputs['x']
    scaled_0 = x - tft.min(x)
    scaled_0_1 = scaled_0 / tft.max(scaled_0)

  Then the first phase will contain the analyzer corresponding to the call to
  `min`, because `x` is an input and so is ready to compute in the first phase,
  while the second phase will contain the analyzer corresponding to the call to
  `max` since `scaled_1` depends on the result of the call to `tft.min` which
  is computed in the first phase.

  More generally, we define a level for each op and each `Analyzer` by walking
  the graph, assigning to each operation the max level of its inputs, to each
  `Tensor` the level of its operation, unless it's the output of an `Analyzer`
  in which case we assign the level of its `Analyzer` plus one.

  Args:
    graph: A `tf.Graph`.
    input_signature: A dict whose keys are strings and values are `Tensor`s or
      `SparseTensor`s.
    output_signature: A dict whose keys are strings and values are `Tensor`s or
      `SparseTensor`s.
    dataset_keys: (Optional) A set of strings which are dataset keys, they
      uniquely identify these datasets across analysis runs.
    cache_location: (Optional): A `CacheLocation` object.

  Returns:
    A list of `Phase`s.

  Raises:
    ValueError: if the graph cannot be analyzed.
  """
  tensor_sinks = graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  graph.clear_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  phase = 0
  tensor_bindings = []
  sink_tensors_ready = {
      tensor_sink.tensor: False for tensor_sink in tensor_sinks
  }
  translate_visitor = _TranslateVisitor()
  translate_traverser = nodes.Traverser(translate_visitor)

  while not all(sink_tensors_ready.values()):
    # Determine which table init ops are ready to run in this phase
    # Determine which keys of pending_tensor_replacements are ready to run
    # in this phase, based in whether their dependencies are ready.
    graph_analyzer = graph_tools.InitializableGraphAnalyzer(
        graph, input_signature.values(), sink_tensors_ready)
    ready_traverser = nodes.Traverser(_ReadyVisitor(graph_analyzer))

    # Now create and apply a SavedModel with all tensors in tensor_bindings
    # bound, which outputs all the tensors in the required tensor tuples.
    intermediate_output_signature = collections.OrderedDict()
    saved_model_future = nodes.apply_operation(
        beam_nodes.CreateSavedModel,
        *tensor_bindings,
        table_initializers=tuple(graph_analyzer.ready_table_initializers),
        output_signature=intermediate_output_signature,
        label='CreateSavedModelForAnalyzerInputs[{}]'.format(phase))

    extracted_values_dict = nodes.apply_operation(
        beam_nodes.ApplySavedModel,
        saved_model_future,
        dataset_key=None,
        phase=phase,
        label='ApplySavedModel[{}]'.format(phase))

    translate_visitor.phase = phase
    translate_visitor.intermediate_output_signature = (
        intermediate_output_signature)
    translate_visitor.extracted_values_dict = extracted_values_dict
    for tensor, value_node, is_asset_filepath in tensor_sinks:
      # Don't compute a binding/sink/replacement that's already been computed
      if sink_tensors_ready[tensor]:
        continue

      if not ready_traverser.visit_value_node(value_node):
        continue

      translated_value_node = translate_traverser.visit_value_node(value_node)

      name = _tensor_name(tensor)
      tensor_bindings.append(
          nodes.apply_operation(
              beam_nodes.CreateTensorBinding,
              translated_value_node,
              tensor=str(tensor.name),
              is_asset_filepath=is_asset_filepath,
              label='CreateTensorBinding[{}]'.format(name)))
      sink_tensors_ready[tensor] = True

    phase += 1

  # We need to make sure that the representation of this output_signature is
  # deterministic.
  output_signature = collections.OrderedDict(
      sorted(output_signature.items(), key=lambda t: t[0]))

  # TODO(KesterTong): check all table initializers are ready, check all output
  # tensors are ready.
  saved_model_future = nodes.apply_operation(
      beam_nodes.CreateSavedModel,
      *tensor_bindings,
      table_initializers=tuple(
          graph.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)),
      output_signature=output_signature,
      label='CreateSavedModel')

  return _perform_cache_optimization(saved_model_future, cache_location,
                                     dataset_keys)
