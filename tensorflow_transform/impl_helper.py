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
"""Helper/utility functions that a tf-transform implementation would find handy."""

import functools
import os
import re
from typing import Callable, Dict, List, Mapping, Optional, FrozenSet

from absl import logging
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import annotators
from tensorflow_transform import common_types
from tensorflow_transform import graph_context
from tensorflow_transform import graph_tools
from tensorflow_transform import schema_inference
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.output_wrapper import TFTransformOutput
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import tensor_to_arrow
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
# pylint: enable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


_VALID_SCOPE_REGEX = re.compile('^[A-Za-z0-9]*$')
_INVALID_SCOPE_CHAR = re.compile('[^A-Za-z0-9_.\\-/>]')

METADATA_DIR_NAME = '.tft_metadata'

_FEATURE_VALUE_KIND_TO_NP_DTYPE = {
    'float_list': np.float32,
    'int64_list': np.int64,
    'bytes_list': object,
}


def batched_placeholders_from_specs(specs):
  """Returns placeholders for the given tf.TypeSpecs or feature specs.

  Args:
    specs: a Dict[Text, Union[tf.TypeSpec, FeatureSpec]]. Note that the values
      in this dict must be of the same type. Mixing is not allowed.

  Returns:
    A dictionary from strings to `Tensor`, `SparseTensor`s, or `RaggedTensor`s.

  Raises:
    ValueError: when the TypeSpec or feature spec has an unsupported dtype.
  """
  if not (all([_is_feature_spec(s) for s in specs.values()]) or
          all([isinstance(s, tf.TypeSpec) for s in specs.values()])):
    raise TypeError('Specs must be all tf.TypeSpecs or feature specs. '
                    'Mixing is not allowed. Got: {}'.format(specs))

  result = {}
  for name, spec in specs.items():
    if isinstance(spec, tf.RaggedTensorSpec):
      # TODO(b/159717195): clean up protected-access
      spec_dtype = spec._dtype  # pylint: disable=protected-access
    else:
      spec_dtype = spec.dtype
    if spec_dtype not in (tf.int64, tf.float32, tf.string):
      raise ValueError('Feature {} ({}, {}) had invalid dtype'.format(
          name, spec, type(spec)))
    if isinstance(spec, tf.TypeSpec):
      result[name] = _batched_placeholder_from_typespec(name, spec)
    else:
      result[name] = _batched_placeholder_from_feature_spec(name, spec)

  return result


def _is_feature_spec(spec):
  return isinstance(spec, (tf.io.VarLenFeature, tf.io.SparseFeature,
                           tf.io.FixedLenFeature, tf.io.RaggedFeature))


def _sanitize_scope_name(name):
  scope_name = _INVALID_SCOPE_CHAR.sub('_', name)
  if not _VALID_SCOPE_REGEX.match(scope_name):
    scope_name = 'F_{}'.format(scope_name)
  return scope_name


def _batched_placeholder_from_typespec(name, typespec):
  """Creates a batched placeholder from a tf.TypeSpec."""
  if isinstance(typespec,
                (tf.TensorSpec, tf.SparseTensorSpec, tf.RaggedTensorSpec)):
    sanitized_name = _sanitize_scope_name(name)
    with tf.name_scope(sanitized_name):
      return tf.nest.map_structure(
          lambda tspec: tf.raw_ops.Placeholder(  # pylint: disable=g-long-lambda
              dtype=tspec.dtype,
              shape=tspec.shape,
              name=sanitized_name),
          typespec,
          expand_composites=True)

  raise ValueError('Unsupported typespec: {}({}) for feature {}'.format(
      typespec, type(typespec), name))


def _batched_placeholder_from_feature_spec(name, feature_spec):
  """Creates a batched placeholder from a feature spec."""
  scope_name = _sanitize_scope_name(name)
  if isinstance(feature_spec, tf.io.FixedLenFeature):
    return tf.compat.v1.placeholder(
        feature_spec.dtype, [None] + feature_spec.shape, name=scope_name)
  elif isinstance(feature_spec, tf.io.VarLenFeature):
    return tf.compat.v1.sparse_placeholder(
        feature_spec.dtype, [None, None], name=scope_name)
  elif isinstance(feature_spec, tf.io.SparseFeature):
    shape = [None] + feature_spec.size if isinstance(
        feature_spec.size, list) else [None, feature_spec.size]
    return tf.compat.v1.sparse_placeholder(
        feature_spec.dtype, shape, name=scope_name)

  raise ValueError('Unsupported feature spec: {}({}) for feature {}'.format(
      feature_spec, type(feature_spec), name))


def _example_to_dict(example: bytes) -> Dict[str, Optional[np.ndarray]]:
  """Converts serialized tf.Example to Python dictionary."""
  example = tf.train.Example.FromString(example)
  result = {}
  # Sort the produced dict by keys to make the order deterministic.
  for name, feature in sorted(example.features.feature.items()):
    kind = feature.WhichOneof('kind')
    # Use None if the value kind is not set (can occur for passthrough values).
    result[name] = (None if kind is None else np.array(
        getattr(feature, kind).value,
        dtype=_FEATURE_VALUE_KIND_TO_NP_DTYPE[kind]))
  return result


def record_batch_to_instance_dicts(
    record_batch: pa.RecordBatch, schema: schema_pb2.Schema
) -> List[common_types.InstanceDictType]:
  """Converts pa.RecordBatch to list of Python dictionaries.

  Args:
    record_batch: the batch to be converted.
    schema: A `Schema` proto.

  Returns:
    A list of dicts where each dict is an in-memory representation of an
        instance.
  """
  # Alternatively, we could've used `record_batch.to_pylist()`, but
  # RaggedTensors would be represented as nested lists (as opposed to array of
  # values + row lengths), so we make a trip through flat examples first.
  coder = example_coder.RecordBatchToExamplesEncoder(schema)
  examples = coder.encode(record_batch)
  # Dense tensor instances must be reshaped according to their spec shape.
  # Scalars are represented as Python scalars (as opposed to singleton arrays).
  feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
  dense_reshape_fns = {}
  def _extract_singleton_item(
      singleton: np.ndarray,
  ) -> common_types.PrimitiveType:
    return singleton.item()
  for name, spec in feature_spec.items():
    if isinstance(spec, tf.io.FixedLenFeature):
      if spec.shape:
        dense_reshape_fns[name] = functools.partial(
            np.reshape, newshape=spec.shape
        )
      else:
        dense_reshape_fns[name] = _extract_singleton_item
  result = []
  for example in examples:
    instance_dict = _example_to_dict(example)
    for name, reshape_fn in dense_reshape_fns.items():
      instance_dict[name] = reshape_fn(instance_dict[name])
    result.append(instance_dict)
  return result


def validate_varlen_sparse_value(
    name: str, batched_value: common_types.SparseTensorValueType
):
  """Checks that the given SparseTensor is 2-D ragged and left-aligned."""
  indices = np.asarray(batched_value.indices)
  if indices.shape[1] != 2:
    raise ValueError(f'Encountered non 2-D varlen sparse feature {name}')
  if indices.shape[0] == 0:
    return
  indices_diff = np.diff(indices, axis=0)
  instance_index_diff, value_index_diff = indices_diff[:, 0], indices_diff[:, 1]
  if np.any(instance_index_diff < 0):
    raise ValueError(
        f'Encountered decreasing instance indices for feature {name}: {indices}'
    )
  if np.any(np.logical_and(instance_index_diff == 0, value_index_diff != 1)):
    raise ValueError(
        f'Encountered non-consecutive value indices for feature {name}:'
        f' {indices}'
    )
  (instance_boundaries,) = np.where(instance_index_diff != 0)
  if np.any(indices[np.append(instance_boundaries + 1, 0), 1] != 0):
    raise ValueError(
        f'Encountered non-zero starting value indices for feature {name}:'
        f' {indices}'
    )


def get_type_specs_from_feature_specs(
    feature_specs: Dict[str, common_types.FeatureSpecType],
    ragged_sequence_features: FrozenSet[str] = frozenset(),
) -> Dict[str, tf.TypeSpec]:
  """Returns `tf.TypeSpec`s for the given feature specs.

  Returns a dictionary of type_spec with the same type and shape as defined by
  `feature_specs`.

  Args:
    feature_specs: A TensorFlow feature spec.
    ragged_sequence_features: Set of names of features representing ragged
      sequence tensors.

  Returns:
    A dictionary from strings to `tf.TensorSpec`, `tf.SparseTensorSpec` or
    `tf.RaggedTensorSpec`s.

  Raises:
    ValueError: If the feature spec contains feature types not supported.
  """
  result = {}
  for name, feature_spec in feature_specs.items():
    if isinstance(feature_spec, tf.io.FixedLenFeature):
      result[name] = tf.TensorSpec([None] + list(feature_spec.shape),
                                   feature_spec.dtype)
    elif isinstance(feature_spec, tf.io.VarLenFeature):
      result[name] = tf.SparseTensorSpec([None, None], feature_spec.dtype)
    elif isinstance(feature_spec, tf.io.SparseFeature):
      shape = [None] + [None if dim == -1 else dim for dim in feature_spec.size]
      result[name] = tf.SparseTensorSpec(shape, feature_spec.dtype)
    elif isinstance(feature_spec, tf.io.RaggedFeature):
      # Number of dimensions is number of partitions + 1 + 1 batch dimension.
      shape = [None, None]
      ragged_rank = 1
      # Ragged sequence tensors will have additional sequence dimension.
      if name in ragged_sequence_features:
        shape.append(None)
        ragged_rank += 1
      for partition in feature_spec.partitions:
        if isinstance(partition, tf.io.RaggedFeature.UniformRowLength):  # pytype: disable=attribute-error
          shape.append(partition.length)
        else:
          shape.append(None)
          ragged_rank += 1
      result[name] = tf.RaggedTensorSpec(
          shape=shape,
          dtype=feature_spec.dtype,
          ragged_rank=ragged_rank,
          row_splits_dtype=feature_spec.row_splits_dtype)
    else:
      raise ValueError('Invalid feature spec {}.'.format(feature_spec))
  return result


def make_tensor_to_arrow_converter(
    schema: schema_pb2.Schema) -> tensor_to_arrow.TensorsToRecordBatchConverter:
  """Constructs a `tf.Tensor` to `pa.RecordBatch` converter."""
  # Ragged sequence features will have an additional (sequence) dimension that
  # doesn't come from feature partition. Hence, we need to generate type spec
  # accordingly.
  ragged_sequence_features = set()
  feature_specs = schema_utils.schema_as_feature_spec(schema).feature_spec
  for feature in schema.feature:
    if feature.type == schema_pb2.FeatureType.STRUCT:
      for child_feature in feature.struct_domain.feature:
        ragged_sequence_features.add(child_feature.name)
  type_specs = get_type_specs_from_feature_specs(
      feature_specs, frozenset(ragged_sequence_features)
  )

  # Make sure that SparseFeatures are handled as generic SparseTensors as
  # opposed to VarLenSparse. Note that at this point only sparse outputs with
  # rank >2 are inferred as SparseFeatures, but this is likely to change.
  sparse_tensor_names = set()
  for name, spec in feature_specs.items():
    if isinstance(spec, tf.io.SparseFeature):
      sparse_tensor_names.add(name)
  options = tensor_to_arrow.TensorsToRecordBatchConverter.Options(
      sparse_tensor_value_column_name_template=schema_inference
      .SPARSE_VALUES_NAME_TEMPLATE,
      sparse_tensor_index_column_name_template=schema_inference
      .SPARSE_INDICES_NAME_TEMPLATE,
      generic_sparse_tensor_names=frozenset(sparse_tensor_names))
  return tensor_to_arrow.TensorsToRecordBatchConverter(type_specs, options)


# TODO(b/149997088): Split into two APIs one that will just trace the
# `preprocessing_fn` using tf.function as is and another that will return
# specific outputs requested for.
def get_traced_transform_fn(
    preprocessing_fn: Callable[
        [Mapping[str, common_types.TensorType]],
        Mapping[str, common_types.TensorType],
    ],
    input_signature: Mapping[str, tf.TypeSpec],
    tf_graph_context: graph_context.TFGraphContext,
    output_keys_to_name_map: Optional[Dict[str, str]] = None,
) -> tf.types.experimental.GenericFunction:
  """Get preprocessing_fn traced using tf.function.

  Args:
    preprocessing_fn: A user defined python function to be traced.
    input_signature: `tf.TypeSpec`s describing the inputs to the
      `preprocessing_fn`.
    tf_graph_context: A `TFGraphContext` context manager to invoke the
      `preprocessing_fn` in.
    output_keys_to_name_map: (Optional) A map from output dictionary keys to the
      names of the tensors that they represent.

  Returns:
    A tf.function object representing a function with the same input signature
    as `preprocessing_fn`.
    If `output_keys_to_name_map` is None or there are no more TFT analyzers to
    evaluate in the `preprocessing_fn`, the output signature of this
    tf.function
    is the same as the `preprocessing_fn`.
    Otherwise, its output signature contains the keys in
    `output_keys_to_name_map` and the tensor represented by the corresponding
    dictionary values.
  """

  assert all([isinstance(s, tf.TypeSpec) for s in input_signature.values()])

  # TODO(b/177672051): Investigate performance impact of enabling autograph.
  @tf.function(input_signature=[input_signature], autograph=False)
  def transform_fn(inputs):
    graph = ops.get_default_graph()
    # If any analyzers have already been evaluated, pass them using the
    # `graph_context.TFGraphContext`. This will be used in place of the analyzer
    # nodes.
    # The user defined `preprocessing_fn` may directly modify its inputs which
    # is not allowed in a tf.function. Hence, we make a copy here.
    inputs_copy = tf_utils.copy_tensors(inputs)
    with tf_graph_context:
      transformed_features = preprocessing_fn(inputs_copy)
    # An empty `TENSOR_REPLACEMENTS` collection symbolizes that there is no
    # analyzer left for Transform to evaluate. Either if this collection is
    # empty or if no specific outputs have been requested, return
    # the same output as `preprocessing_fn` (i.e, transformed_features).
    if (output_keys_to_name_map is None or
        not graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)):
      return transformed_features
    else:
      return {
          key: graph.get_tensor_by_name(value)
          for key, value in output_keys_to_name_map.items()
      }

  return transform_fn


def _trace_preprocessing_fn_v1(preprocessing_fn, specs):
  """Trace TF1 graph for `preprocessing_fn`."""
  with tf.compat.v1.Graph().as_default() as graph:
    with tf.compat.v1.name_scope('inputs'):
      structured_inputs = batched_placeholders_from_specs(specs)
      # In order to avoid a bug where import_graph_def fails when the
      # input_map and return_elements of an imported graph are the same
      # (b/34288791), we avoid using the placeholder of an input column as an
      # output of a graph. We do this by applying tf.identity to all inputs of
      # the preprocessing_fn.  Note this applies at the level of raw tensors.
      # TODO(b/34288791): Remove this workaround and use a shallow copy of
      # inputs instead.  A shallow copy is needed in case
      # self._preprocessing_fn mutates its input.
      copied_inputs = tf_utils.copy_tensors(structured_inputs)

    structured_outputs = preprocessing_fn(copied_inputs)
  return graph, structured_inputs, structured_outputs


def _trace_preprocessing_fn_v2(preprocessing_fn, specs, base_temp_dir):
  """Trace TF2 graph for `preprocessing_fn`."""
  tf_graph_context = graph_context.TFGraphContext(
      module_to_export=tf.Module(),
      temp_dir=base_temp_dir,
      evaluated_replacements=None)
  with annotators.object_tracker_scope(annotators.ObjectTracker()):
    concrete_fn = get_traced_transform_fn(
        preprocessing_fn, specs, tf_graph_context).get_concrete_function()
  return (concrete_fn.graph,
          tf2_utils.get_structured_inputs_from_func_graph(concrete_fn.graph),
          concrete_fn.structured_outputs)


def trace_preprocessing_function(preprocessing_fn,
                                 input_specs,
                                 use_tf_compat_v1,
                                 base_temp_dir=None):
  """Trace graph for `preprocessing_fn`.

  Args:
    preprocessing_fn: A user defined python function to be traced.
    input_specs: A dictionary from input feature name to its FeatureSpec or
      TypeSpec. If use_tf_compat_v1 is `False`, input_specs must be a dictionary
      of TypeSpecs.
    use_tf_compat_v1: (Optional) If `True`, the `preprocessing_fn` is traced as
      a TF 1.x graph. Else, it is traced using tf.function.
    base_temp_dir: (Optional) Base path to write any dummy assets to during
      tracing. Required when `use_tf_compat_v1` is `False`.

  Returns:
    A tuple of:

      0. the graph representing the traced `preprocessing_fn`
      1. the graph's structured inputs
      2. the graph's structured outputs
  """
  if use_tf_compat_v1:
    return _trace_preprocessing_fn_v1(preprocessing_fn, input_specs)
  else:
    return _trace_preprocessing_fn_v2(
        preprocessing_fn, input_specs, base_temp_dir
    )


def _trace_and_write_transform_fn(
    saved_model_dir: str,
    preprocessing_fn: Callable[
        [Mapping[str, common_types.TensorType]],
        Mapping[str, common_types.TensorType],
    ],
    input_signature: Mapping[str, tf.TypeSpec],
    base_temp_dir: Optional[str],
    tensor_replacement_map: Optional[Dict[str, tf.Tensor]],
    output_keys_to_name_map: Optional[Dict[str, str]],
    save_options: Optional[tf.saved_model.SaveOptions],
) -> function.ConcreteFunction:
  """Trace `preprocessing_fn` and serialize to a SavedModel."""
  tf_graph_context = graph_context.TFGraphContext(
      module_to_export=tf.Module(),
      temp_dir=base_temp_dir,
      evaluated_replacements=tensor_replacement_map,
  )
  transform_fn = get_traced_transform_fn(
      preprocessing_fn,
      input_signature,
      tf_graph_context,
      output_keys_to_name_map=output_keys_to_name_map,
  )
  return saved_transform_io_v2.write_v2_saved_model(
      tf_graph_context.module_to_export,
      transform_fn,
      'transform_fn',
      saved_model_dir,
      save_options,
  )


def _trace_and_get_metadata(
    concrete_transform_fn: function.ConcreteFunction,
    structured_inputs: Mapping[str, common_types.TensorType],
    preprocessing_fn: Callable[
        [Mapping[str, common_types.TensorType]],
        Mapping[str, common_types.TensorType],
    ],
    base_temp_dir: Optional[str],
    tensor_replacement_map: Optional[Dict[str, tf.Tensor]],
) -> dataset_metadata.DatasetMetadata:
  """Compute and return metadata for the outputs of `concrete_transform_fn`."""
  tf_graph_context = graph_context.TFGraphContext(
      module_to_export=tf.Module(),
      temp_dir=base_temp_dir,
      evaluated_replacements=tensor_replacement_map,
  )
  concrete_metadata_fn = schema_inference.get_traced_metadata_fn(
      preprocessing_fn,
      structured_inputs,
      tf_graph_context,
      evaluate_schema_overrides=True,
  )
  return dataset_metadata.DatasetMetadata(
      schema=schema_inference.infer_feature_schema_v2(
          concrete_transform_fn.structured_outputs,
          concrete_metadata_fn,
          evaluate_schema_overrides=True,
      )
  )


def _validate_analyzers_fingerprint(
    baseline_analyzers_fingerprint: Mapping[
        str, graph_tools.AnalyzersFingerprint
    ],
    graph: tf.Graph,
    structured_inputs: Mapping[str, common_types.TensorType],
):
  """Validates analyzers fingerprint in `graph` is same as baseline."""
  analyzers_fingerprint = graph_tools.get_analyzers_fingerprint(
      graph, structured_inputs
  )
  error_msg = (
      'The order of analyzers in your `preprocessing_fn` appears to be '
      'non-deterministic. This can be fixed either by changing your '
      '`preprocessing_fn` such that tf.Transform analyzers are encountered '
      'in a deterministic order or by passing a unique name to each '
      'analyzer API call.'
  )
  for analyzer in analyzers_fingerprint:
    if analyzer not in baseline_analyzers_fingerprint:
      prefix_msg = (
          f'Analyzer node ({analyzer}) not found in '
          f'{baseline_analyzers_fingerprint.keys()}. '
      )
      raise RuntimeError(prefix_msg + error_msg)
    if (
        baseline_analyzers_fingerprint[analyzer].source_keys
        != analyzers_fingerprint[analyzer].source_keys
    ):
      raise RuntimeError(error_msg)

    if (
        baseline_analyzers_fingerprint[analyzer].unique_path_hash
        != analyzers_fingerprint[analyzer].unique_path_hash
    ):
      logging.warning(
          "Analyzer (%s) node's cache key varies on repeated tracing."
          ' This warning is safe to ignore if you either specify `name` for all'
          ' analyzers or if the order in which they are invoked is'
          ' deterministic. If not, please file a bug with details.',
          analyzer,
      )


def trace_and_write_v2_saved_model(
    saved_model_dir: str,
    preprocessing_fn: Callable[
        [Mapping[str, common_types.TensorType]],
        Mapping[str, common_types.TensorType],
    ],
    input_signature: Mapping[str, tf.TypeSpec],
    base_temp_dir: Optional[str],
    baseline_analyzers_fingerprint: Mapping[
        str, graph_tools.AnalyzersFingerprint
    ],
    tensor_replacement_map: Optional[Dict[str, tf.Tensor]],
    output_keys_to_name_map: Optional[Dict[str, str]],
    save_options: Optional[tf.saved_model.SaveOptions],
):
  """Writes out a SavedModelV2 with preprocessing_fn traced using tf.function.

  The SavedModel written contains a method called `transform_fn` that
  represents the traced `preprocessing_fn`. Additionally, if this is the final
  SavedModel being written out, it will contain a method called `metadata_fn`
  that provides deferred schema annotations.

  Args:
    saved_model_dir: Path to write SavedModel to.
    preprocessing_fn: A user defined python function to be traced.
    input_signature: TypeSpecs describing the inputs to the `preprocessing_fn`.
    base_temp_dir: Base path to write temporary artifacts to.
    baseline_analyzers_fingerprint: A mapping from analyzer name to a set of
      paths that define its fingerprint.
    tensor_replacement_map: A map from placeholder tensor names to their
      evaluated replacement tensors.
    output_keys_to_name_map: A map from output dictionary keys to the names of
      the tensors that they represent.
    save_options: The options to use when saving the saved_model.

  Returns:
    A tuple containing a pair of `tf.ConcreteFunction`s:
      1. The traced preprocessing_fn.
      2. A metadata_fn that returns a dictionary containing the deferred
      annotations added to the graph when invoked with any valid input.

  Raises:
    RuntimeError: if analyzers in `preprocessing_fn` are encountered in a
    non-deterministic order.
  """
  concrete_transform_fn = _trace_and_write_transform_fn(
      saved_model_dir, preprocessing_fn, input_signature, base_temp_dir,
      tensor_replacement_map, output_keys_to_name_map, save_options)
  structured_inputs = tf2_utils.get_structured_inputs_from_func_graph(
      concrete_transform_fn.graph)
  _validate_analyzers_fingerprint(baseline_analyzers_fingerprint,
                                  concrete_transform_fn.graph,
                                  structured_inputs)

  # If the `TENSOR_REPLACEMENTS` graph collection is empty, all TFT analyzers
  # in the `preprocessing_fn` have already been evaluated.
  if not concrete_transform_fn.graph.get_collection(
      analyzer_nodes.TENSOR_REPLACEMENTS):
    metadata = _trace_and_get_metadata(concrete_transform_fn, structured_inputs,
                                       preprocessing_fn, base_temp_dir,
                                       tensor_replacement_map)
    metadata_io.write_metadata(metadata,
                               os.path.join(saved_model_dir, METADATA_DIR_NAME))


def _assert_no_analyzers_in_graph(graph):
  if graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS):
    raise RuntimeError('TFT analyzers found when tracing the given '
                       '`preprocessing_fn`. Please use '
                       '`tft.beam.AnalyzeDataset` to analyze this function.')


def analyze_in_place(preprocessing_fn, force_tf_compat_v1, feature_specs,
                     type_specs, transform_output_path):
  """Analyzes the `preprocessing_fn` in-place without looking at the data.

  This should only be used if the `preprocessing_fn` contains no TFT
  analyzers or TFT mappers that use analyzers.

  Writes out a transform function and transformed metadata to subdirs under
  `transform_output_path`.

  Args:
    preprocessing_fn: The tf.Transform preprocessing_fn.
    force_tf_compat_v1: If True, call Transform's API to use Tensorflow in
      tf.compat.v1 mode.
    feature_specs: a Dict from input feature key to its feature spec.
    type_specs: a Dict from input feature key to its type spec.
    transform_output_path: An absolute path to write the output to.

  Raises:
    RuntimeError if `preprocessing_fn` contains TFT analyzers.
  """
  use_tf_compat_v1 = tf2_utils.use_tf_compat_v1(force_tf_compat_v1)
  transform_fn_path = os.path.join(transform_output_path,
                                   TFTransformOutput.TRANSFORM_FN_DIR)
  if use_tf_compat_v1:
    graph, structured_inputs, structured_outputs = (
        trace_preprocessing_function(
            preprocessing_fn, feature_specs, use_tf_compat_v1=use_tf_compat_v1))
    _assert_no_analyzers_in_graph(graph)
    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.tables_initializer())
      saved_transform_io.write_saved_transform_from_session(
          sess, structured_inputs, structured_outputs, transform_fn_path)

      transformed_metadata = dataset_metadata.DatasetMetadata(
          schema=schema_inference.infer_feature_schema(structured_outputs,
                                                       graph, sess))
  else:
    concrete_transform_fn = _trace_and_write_transform_fn(
        saved_model_dir=transform_fn_path,
        preprocessing_fn=preprocessing_fn,
        input_signature=type_specs,
        base_temp_dir=None,
        tensor_replacement_map=None,
        output_keys_to_name_map=None,
        save_options=None)
    _assert_no_analyzers_in_graph(concrete_transform_fn.graph)
    structured_inputs = tf2_utils.get_structured_inputs_from_func_graph(
        concrete_transform_fn.graph)
    transformed_metadata = _trace_and_get_metadata(
        concrete_transform_fn=concrete_transform_fn,
        structured_inputs=structured_inputs,
        preprocessing_fn=preprocessing_fn,
        base_temp_dir=None,
        tensor_replacement_map=None)
  transformed_metadata_dir = os.path.join(
      transform_output_path, TFTransformOutput.TRANSFORMED_METADATA_DIR)
  metadata_io.write_metadata(transformed_metadata, transformed_metadata_dir)
