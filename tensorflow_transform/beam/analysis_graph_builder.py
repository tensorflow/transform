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

import collections
import hashlib

import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import graph_tools
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.beam import beam_nodes
from tensorflow_transform.beam import combiner_packing_util
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple


# Used for debugging only. This will point to the most recent graph built.
_ANALYSIS_GRAPH = None


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
    self._visited_operation_def_labels = set()

  def _validate_operation_label_uniqueness(self, operation_def):
    assert operation_def.label not in self._visited_operation_def_labels, (
        f'An operation with label {operation_def.label} '
        'already exists in the operations graph.')
    self._visited_operation_def_labels.add(operation_def.label)

  def visit(self, operation_def, input_values):
    self._validate_operation_label_uniqueness(operation_def)

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
    tfx_namedtuple.namedtuple('_OptimizationView', [
        'prefer_fine_grained_view', 'flattened_view', 'fine_grained_view',
        'hashed_path'
    ])):
  """A container for operation outputs during _OptimizeVisitor traversal.

  This is used in order to maintain both a flattened view, and a fine grained
  view that can be used for caching.

  `prefer_fine_grained_view` is a hint that means that if True, the
  `fine_grained_view` should be used.  It should be set to true if the upstream
  view has cacheing operations that haven't been flattened yet.
  """

  def __init__(self, prefer_fine_grained_view, flattened_view,
               fine_grained_view, hashed_path):
    if prefer_fine_grained_view and not fine_grained_view:
      raise ValueError(
          'Cannot prefer fine_grained_view when one is not provided')
    del hashed_path
    self._validate_flattened_view(flattened_view)
    self._validate_fine_grained_view(fine_grained_view)
    super().__init__()

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

  def __init__(self, dataset_keys, cache_dict, tensor_keys_to_paths,
               cache_output_nodes):
    """Init method for _OptimizeVisitor.

    Args:
      dataset_keys: An iterable of strings which are keys for a partitioned
        dataset.
      cache_dict: A dictionary of input cache that can be used in place of a
        cacheable accumulate operation. A dictionary from dataset_keys to
        dictionaries of cache keys to PCollections. This can be None if there is
        no cache.
      tensor_keys_to_paths: A dictionary from a tensor key to a unique TF graph
        path hash.
      cache_output_nodes: A dictionary from (dataset_key, cache_key) to encoded
        cache ValueNode. This is the output cache for this graph.
    """
    self._sorted_dataset_keys = sorted(dataset_keys)
    self._cache_dict = cache_dict
    self._tensor_keys_to_paths = tensor_keys_to_paths
    self.cache_output_nodes = cache_output_nodes

  def _validate_operation_def(self, operation_def):
    if operation_def.cache_coder is not None:
      if not operation_def.is_partitionable:
        raise ValueError(
            'Non partitionable OperationDefs cannot be cacheable: {}'.format(
                operation_def.label))
    if operation_def.is_partitionable or operation_def.cache_coder is not None:
      if operation_def.num_outputs != 1:
        raise ValueError(
            'Cacheable OperationDefs must have exactly 1 output: {}'.format(
                operation_def.label))

  def _make_next_hashed_path(self, parent_hashed_paths, operation_def):
    # Making a copy of parent_hashed_paths.
    paths_to_hash = list(parent_hashed_paths)
    paths_to_hash.append(tf.compat.as_bytes(operation_def.__class__.__name__))

    if isinstance(operation_def, beam_nodes.ExtractFromDict):
      for key in operation_def.keys:
        path = self._tensor_keys_to_paths[key]
        paths_to_hash.append(path)
    else:
      for attr in sorted(
          [x for x in dir(operation_def) if x not in operation_def._fields]):
        if attr.startswith('_') or callable(getattr(operation_def, attr)):
          continue
        paths_to_hash.append(
            tf.compat.as_bytes(str((attr, getattr(operation_def, attr)))))
      for field in operation_def._fields:
        paths_to_hash.append(
            tf.compat.as_bytes(
                str((field, operation_def.get_field_str(field)))))

    hash_container = hashlib.sha1()
    for path in paths_to_hash:
      if path is None:
        return None
      hash_container.update(path)
    return hash_container.digest()

  def visit(self, operation_def, input_values):
    self._validate_operation_def(operation_def)

    if (isinstance(operation_def, beam_nodes.ApplySavedModel) and
        operation_def.phase == 0):
      return self._visit_apply_savedmodel_operation(operation_def, input_values)

    # When self._cache_dict is None this means that we shouldn't do any cacheing
    # for this pipeline, and so there's no need to create any fine grained
    # views.
    if self._cache_dict is not None and operation_def.is_partitionable:
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
        _OptimizationView(  # pylint: disable=g-complex-comprehension
            prefer_fine_grained_view=False,
            flattened_view=flat,
            fine_grained_view=None,
            hashed_path=None) for flat in flattened_view)

  def _visit_partitionable_operation(self, operation_def, upstream_views):

    # This is a hint for whether or not the `fine_grained_view` should be used
    # downstream.  It should be set to true if either the upstream view has
    # cacheing operations that haven't been flattened yet, or the current
    # operation is cacheable.
    all_fine_grained_views_available = all(
        v.fine_grained_view for v in upstream_views)
    prefer_fine_grained_view = (
        any(v.prefer_fine_grained_view for v in upstream_views) or
        all_fine_grained_views_available and
        operation_def.cache_coder is not None)

    next_hashed_path = self._make_next_hashed_path(
        [v.hashed_path for v in upstream_views], operation_def)
    if all_fine_grained_views_available:
      fine_grained_views = (self._apply_operation_on_fine_grained_view(
          operation_def, tuple(v.fine_grained_view for v in upstream_views),
          next_hashed_path),)
    else:
      fine_grained_views = (None,) * operation_def.num_outputs

    flattened_views = nodes.OperationNode(
        operation_def, tuple(v.flattened_view for v in upstream_views)).outputs

    assert len(fine_grained_views) == len(flattened_views)
    return tuple(
        _OptimizationView(  # pylint: disable=g-complex-comprehension
            prefer_fine_grained_view=prefer_fine_grained_view,
            flattened_view=flat,
            fine_grained_view=fine,
            hashed_path=next_hashed_path)
        for flat, fine in zip(flattened_views, fine_grained_views))

  def _apply_operation_on_fine_grained_view(self, operation_def,
                                            fine_grained_views,
                                            next_hashed_path):
    """Applies a shardable operation on a fine grained view.

    This also updates `cache_output_nodes` when necessary.

    Args:
      operation_def: A shardable `OperationDef`.
      fine_grained_views: A tuple of `_OptimizationView.fine_grained_view`s.
      next_hashed_path: The hashed path for the currently processed
        operation_def.

    Returns:
      The resulting list of `_OptimizationView.fine_grained_view`s.
    """
    result_fine_grained_view = collections.OrderedDict()

    cache_entry_key = analyzer_cache.make_cache_entry_key(
        tf.compat.as_bytes(operation_def.label) + b'-' + next_hashed_path)

    for (dataset_idx, dataset_key) in enumerate(self._sorted_dataset_keys):
      # We use an index for the label in order to make beam labels more stable.
      infix = 'AnalysisIndex{}'.format(dataset_idx)
      if (operation_def.cache_coder and self._cache_dict.get(
          dataset_key, {}).get(cache_entry_key) is not None):
        decode_cache = analyzer_nodes.DecodeCache(
            dataset_key,
            cache_entry_key,
            coder=operation_def.cache_coder,
            label='DecodeCache[{}][{}]'.format(operation_def.label, infix))
        (op_output,) = nodes.OperationNode(decode_cache, tuple()).outputs
      else:
        value_nodes = tuple(v[dataset_key] for v in fine_grained_views)
        (op_output,) = nodes.OperationNode(
            operation_def._replace(
                label='{}[{}]'.format(operation_def.label, infix)),
            value_nodes).outputs
        if operation_def.cache_coder:
          encode_cache = nodes.apply_operation(
              analyzer_nodes.EncodeCache,
              op_output,
              coder=operation_def.cache_coder,
              label='EncodeCache[{}][{}]'.format(operation_def.label, infix))
          self.cache_output_nodes[(dataset_key, cache_entry_key)] = encode_cache
      result_fine_grained_view[dataset_key] = op_output

    return result_fine_grained_view

  def _visit_apply_savedmodel_operation(self, operation_def, upstream_views):
    if any(v.fine_grained_view for v in upstream_views):
      raise ValueError(
          'Was not expecting a fine_grained_view input for ApplySavedModel')
    (saved_model_path_upstream_view, input_upstream_view) = upstream_views

    fine_grained_view = collections.OrderedDict()
    for (dataset_idx, dataset_key) in enumerate(self._sorted_dataset_keys):
      infix = 'AnalysisIndex{}'.format(dataset_idx)
      input_node = nodes.apply_operation(
          beam_nodes.ExtractInputForSavedModel,
          dataset_key=dataset_key,
          label='ExtractInputForSavedModel[{}]'.format(infix))
      # We use an index for the label in order to make beam labels more stable.
      (fine_grained_view[dataset_key],) = (
          nodes.OperationNode(
              operation_def._replace(
                  label='{}[{}]'.format(operation_def.label, infix)),
              (saved_model_path_upstream_view.flattened_view,
               input_node)).outputs)

    (flattened_view,) = nodes.OperationNode(
        operation_def, (saved_model_path_upstream_view.flattened_view,
                        input_upstream_view.flattened_view)).outputs

    return (_OptimizationView(
        prefer_fine_grained_view=False,
        flattened_view=flattened_view,
        fine_grained_view=fine_grained_view,
        hashed_path=b'APPLY_SAVEDMODEL'),)

  def validate_value(self, value):
    assert isinstance(value, _OptimizationView), value
    if value.fine_grained_view:
      assert set(value.fine_grained_view.keys()) == set(
          self._sorted_dataset_keys), ('{} != {}'.format(
              value.fine_grained_view.keys(), self._sorted_dataset_keys))


def _perform_cache_optimization(saved_model_future, dataset_keys,
                                tensor_keys_to_paths, cache_dict):
  """Performs cache optimization on the given graph."""
  cache_output_nodes = {}
  optimize_visitor = _OptimizeVisitor(dataset_keys or {}, cache_dict,
                                      tensor_keys_to_paths, cache_output_nodes)
  optimize_traverser = nodes.Traverser(optimize_visitor)
  optimized = optimize_traverser.visit_value_node(
      saved_model_future).flattened_view

  if cache_dict is None:
    assert not cache_output_nodes
    cache_output_nodes = None

  return optimized, cache_output_nodes


class _InspectVisitor(nodes.Visitor):
  """A visitor that inspects the graph and looks for dataset keys in use."""

  def __init__(self, required_dataset_keys_output):
    self._required_dataset_keys = required_dataset_keys_output

  def visit(self, operation_def, input_values):
    if isinstance(operation_def, beam_nodes.ExtractInputForSavedModel):
      self._required_dataset_keys.add(operation_def.dataset_key)
    return nodes.OperationNode(operation_def, input_values).outputs

  def validate_value(self, value):
    assert isinstance(value, nodes.ValueNode)


def _build_analysis_graph_for_inspection(preprocessing_fn, specs, dataset_keys,
                                         input_cache, force_tf_compat_v1):
  """Builds the analysis graph for inspection."""
  if not force_tf_compat_v1:
    assert all([isinstance(s, tf.TypeSpec) for s in specs.values()]), specs
  graph, structured_inputs, structured_outputs = (
      impl_helper.trace_preprocessing_function(
          preprocessing_fn,
          specs,
          use_tf_compat_v1=tf2_utils.use_tf_compat_v1(force_tf_compat_v1)))

  transform_fn_future, cache_dict = build(
      graph,
      structured_inputs,
      structured_outputs,
      dataset_keys=dataset_keys,
      cache_dict=input_cache)
  return transform_fn_future, cache_dict


def get_analysis_dataset_keys(preprocessing_fn,
                              specs,
                              dataset_keys,
                              input_cache,
                              force_tf_compat_v1):
  """Computes the dataset keys that are required in order to perform analysis.

  Args:
    preprocessing_fn: A tf.transform preprocessing_fn.
    specs: A dict of feature name to tf.TypeSpecs. If `force_tf_compat_v1` is
      True, this can also be feature specifications.
    dataset_keys: A set of strings which are dataset keys, they uniquely
      identify these datasets across analysis runs.
    input_cache: A cache dictionary.
    force_tf_compat_v1: If `True`, use Tensorflow in compat.v1 mode.

  Returns:
    A set of dataset keys that are required for analysis.
  """
  transform_fn_future, _ = _build_analysis_graph_for_inspection(
      preprocessing_fn, specs, dataset_keys, input_cache, force_tf_compat_v1)

  result = set()
  inspect_visitor = _InspectVisitor(result)
  inspect_traverser = nodes.Traverser(inspect_visitor)
  _ = inspect_traverser.visit_value_node(transform_fn_future)

  # If None is present this means that a flattened version of the entire dataset
  # is required, therefore this will be returning all of the given dataset_keys.
  if any(k.is_flattened_dataset_key() for k in result):
    result = dataset_keys
  return result


def get_analysis_cache_entry_keys(preprocessing_fn,
                                  specs,
                                  dataset_keys,
                                  force_tf_compat_v1):
  """Computes the cache entry keys that would be useful for analysis.

  Args:
    preprocessing_fn: A tf.transform preprocessing_fn.
    specs: A dict of feature name to tf.TypeSpecs. If `force_tf_compat_v1` is
      True, this can also be feature specifications.
    dataset_keys: A set of strings which are dataset keys, they uniquely
      identify these datasets across analysis runs.
    force_tf_compat_v1: If `True`, use Tensorflow in compat.v1 mode.

  Returns:
    A set of cache entry keys which would be useful for analysis.
  """
  _, cache_dict = _build_analysis_graph_for_inspection(preprocessing_fn, specs,
                                                       dataset_keys, {},
                                                       force_tf_compat_v1)
  return set([cache_key for _, cache_key in cache_dict.keys()])


def build(graph,
          input_signature,
          output_signature,
          dataset_keys=None,
          cache_dict=None):
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
    cache_dict: (Optional): A cache dictionary.

  Returns:
    A pair of:
      * list of `Phase`s
      * A dictionary of output cache `ValueNode`s.

  Raises:
    ValueError: if the graph cannot be analyzed.
  """
  tensor_sinks = graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  graph.clear_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  phase = 0
  tensor_bindings = []
  sink_tensors_ready = {
      tf_utils.hashable_tensor_or_op(tensor_sink.tensor):
          False for tensor_sink in tensor_sinks
  }
  translate_visitor = _TranslateVisitor()
  translate_traverser = nodes.Traverser(translate_visitor)

  analyzers_input_signature = {}
  graph_analyzer = None

  extracted_input_node = nodes.apply_operation(
      beam_nodes.ExtractInputForSavedModel,
      dataset_key=analyzer_cache._make_flattened_dataset_key(),  # pylint: disable=protected-access
      label='ExtractInputForSavedModel[FlattenedDataset]')

  while not all(sink_tensors_ready.values()):
    infix = 'Phase{}'.format(phase)
    # Determine which table init ops are ready to run in this phase
    # Determine which keys of pending_tensor_replacements are ready to run
    # in this phase, based in whether their dependencies are ready.
    graph_analyzer = graph_tools.InitializableGraphAnalyzer(
        graph, input_signature, list(sink_tensors_ready.items()),
        graph_tools.describe_path_as_analyzer_cache_hash)
    ready_traverser = nodes.Traverser(_ReadyVisitor(graph_analyzer))

    # Now create and apply a SavedModel with all tensors in tensor_bindings
    # bound, which outputs all the tensors in the required tensor tuples.
    intermediate_output_signature = collections.OrderedDict()
    saved_model_future = nodes.apply_operation(
        beam_nodes.CreateSavedModel,
        *tensor_bindings,
        table_initializers=tuple(graph_analyzer.ready_table_initializers),
        output_signature=intermediate_output_signature,
        label='CreateSavedModelForAnalyzerInputs[{}]'.format(infix))

    extracted_values_dict = nodes.apply_operation(
        beam_nodes.ApplySavedModel,
        saved_model_future,
        extracted_input_node,
        phase=phase,
        label='ApplySavedModel[{}]'.format(infix))

    translate_visitor.phase = phase
    translate_visitor.intermediate_output_signature = (
        intermediate_output_signature)
    translate_visitor.extracted_values_dict = extracted_values_dict
    for tensor, value_node, is_asset_filepath in tensor_sinks:
      hashable_tensor = tf_utils.hashable_tensor_or_op(tensor)
      # Don't compute a binding/sink/replacement that's already been computed
      if sink_tensors_ready[hashable_tensor]:
        continue

      if not ready_traverser.visit_value_node(value_node):
        continue

      translated_value_node = translate_traverser.visit_value_node(value_node)

      name = _tensor_name(tensor)
      tensor_bindings.append(
          nodes.apply_operation(
              beam_nodes.CreateTensorBinding,
              translated_value_node,
              tensor_name=str(tensor.name),
              dtype_enum=tensor.dtype.as_datatype_enum,
              is_asset_filepath=is_asset_filepath,
              label=analyzer_nodes.sanitize_label(
                  'CreateTensorBinding[{}]'.format(name))))
      sink_tensors_ready[hashable_tensor] = True

    analyzers_input_signature.update(intermediate_output_signature)
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
          graph.get_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)),
      output_signature=output_signature,
      label='CreateSavedModel')

  tensor_keys_to_paths = {
      tensor_key:
      graph_analyzer.get_unique_path(analyzers_input_signature[tensor_key])
      for tensor_key in analyzers_input_signature
  }
  (optimized_saved_model_future,
   output_cache_value_nodes) = _perform_cache_optimization(
       saved_model_future, dataset_keys, tensor_keys_to_paths, cache_dict)

  (optimized_saved_model_future, output_cache_value_nodes) = (
      combiner_packing_util.perform_combiner_packing_optimization(
          optimized_saved_model_future, output_cache_value_nodes, phase))

  global _ANALYSIS_GRAPH
  _ANALYSIS_GRAPH = optimized_saved_model_future
  return optimized_saved_model_future, output_cache_value_nodes
