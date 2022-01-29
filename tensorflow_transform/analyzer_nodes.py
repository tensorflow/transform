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
"""Nodes that define analyzers.

`OperationNode`s are objects that describe how to perform a full pass analysis
over some input tensors.  They are described by an `OperationDef`.  This module
contains the `OperationDef` subclasses that define specific operations such as
computing a mean or vocabulary.  It also contains a special `OperationDef`,
`ExtractTensors` which represents the operation of extracting the values of a
tuple of `Tensor`s into a `PCollection`.
"""

import abc
import json
import os
import struct
from typing import Any, Optional, Sequence, Type
import uuid

import numpy as np
import tensorflow as tf
from tensorflow_transform import common_types
from tensorflow_transform import nodes
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.graph_context import TFGraphContext
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
# pylint: disable=g-enable-tensorflow-import

# Key for graph collection containing `TensorSink` objects representing TFT
# analyzers.
TENSOR_REPLACEMENTS = 'tft_tensor_replacements'
# Key for graph collection containing `TensorSink` objects representing TFT
# analyzers irrespective of whether they have been evaluated or not.
ALL_REPLACEMENTS = 'tft_all_replacements'


def sanitize_label(label: str) -> str:
  return label.replace('/', '#')


def _make_label(cls: Type[nodes.OperationDef],
                label: Optional[str] = None) -> str:
  if label is None:
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    label = '{}[{}]'.format(cls.__name__, scope)
  return sanitize_label(label)


TemporaryAssetInfo = tfx_namedtuple.namedtuple('TemporaryAssetInfo',
                                               ['value', 'file_format'])


class TensorInfo(
    tfx_namedtuple.namedtuple('TensorInfo',
                              ['dtype', 'shape', 'temporary_asset_info'])):
  """A container for attributes of output tensors from analyzers.

  Fields:
    dtype: The TensorFlow dtype.
    shape: The shape of the tensor.
    temporary_asset_info: A named tuple containing information about the
      temporary asset file to write out while tracing the TF graph.
  """

  def __new__(
      cls: Type['TensorInfo'], dtype: tf.dtypes.DType,
      shape: Sequence[Optional[int]],
      temporary_asset_info: Optional[TemporaryAssetInfo]) -> 'TensorInfo':
    if not isinstance(dtype, tf.DType):
      raise TypeError('dtype must be a TensorFlow dtype, got {}'.format(dtype))
    if temporary_asset_info is not None and not isinstance(
        temporary_asset_info, TemporaryAssetInfo):
      raise TypeError(
          'temporary_asset_info should be an instance of TemporaryAssetInfo or '
          f'None, got {temporary_asset_info}')
    return super(TensorInfo, cls).__new__(
        cls,
        dtype=dtype,
        shape=shape,
        temporary_asset_info=temporary_asset_info)


class TensorSource(
    tfx_namedtuple.namedtuple('TensorSource', ['tensors', 'label']),
    nodes.OperationDef):
  """An `OperationDef` that defines extracting a tuple of tensor values.

  This `OperationDef` defines an operation that extracts the values of the given
  tensors into a PCollection of tuples of values.  It is used as a source for
  analyzers, which further transform

  This OperationDef accepts zero inputs and return a single output representing
  the PCollection of tuples of values.  It will be converted in
  tensorflow_transform.beam.analysis_graph_builder.build to an operation that
  extracts the tensors for a dictionary of tensors, after running a beam.ParDo
  to produce tensor values by running the graph on its inputs.

  Fields:
    tensors: The tensors whose values should be extracted.
    label: A unique label for this operation.
  """

  def __new__(cls, tensors):
    for tensor in tensors:
      if not isinstance(tensor, tf.Tensor):
        raise TypeError('tensor must be a Tensor, got {} of type {}'.format(
            tensor, type(tensor)))
    return super(TensorSource, cls).__new__(
        cls, tensors=tensors, label=_make_label(cls))


def get_input_tensors_value_nodes(tensor_inputs):
  return nodes.apply_operation(TensorSource, tensors=tensor_inputs)


TensorSink = tfx_namedtuple.namedtuple(
    'TensorSink', ['tensor', 'future', 'is_asset_filepath'])


def _bind_future_as_tensor_v1(future: nodes.ValueNode,
                              tensor_info: TensorInfo,
                              name: Optional[str] = None) -> tf.Tensor:
  """Bind a future value as a tensor to a TF1 graph."""
  result = tf.compat.v1.placeholder(tensor_info.dtype, tensor_info.shape, name)
  is_asset_filepath = tensor_info.temporary_asset_info is not None
  tf.compat.v1.add_to_collection(TENSOR_REPLACEMENTS,
                                 TensorSink(result, future, is_asset_filepath))
  return result


_TemporaryAnalyzerOutputWrapper = tfx_namedtuple.namedtuple(
    '_TemporaryAnalyzerOutputWrapper', ['eager_asset_path', 'graph_tensor'])


def _write_to_temporary_asset_file(
    temp_dir: str, temporary_asset_info: TemporaryAssetInfo) -> str:
  """Returns path to temporary asset file created during tracing."""
  # TODO(b/170111921): This temporary file should have a unique name to
  # avoid namespace collisions between temporary files that contain data
  # of different dtypes.
  base_filename = uuid.uuid4().hex
  if temporary_asset_info.file_format == 'text':
    result = os.path.join(temp_dir, base_filename)
    with tf.io.gfile.GFile(result, 'w') as f:
      f.write(temporary_asset_info.value)
  elif temporary_asset_info.file_format == 'tfrecord_gzip':
    result = os.path.join(temp_dir, '{}.tfrecord.gz'.format(base_filename))
    with tf.io.TFRecordWriter(result, 'GZIP') as f:
      f.write(temporary_asset_info.value)
  else:
    raise ValueError(
        'File format should be one of \'text\' or \'tfrecord_gzip\'. Received '
        f'{temporary_asset_info.file_format}')
  return result


def _get_temporary_analyzer_output(
    temp_dir: str,
    tensor_info: TensorInfo,
    name: Optional[str] = None) -> _TemporaryAnalyzerOutputWrapper:
  """Create a temporary graph tensor using attributes in `tensor_info`.

  Args:
    temp_dir: Path to a directory to write out any temporary asset files to.
    tensor_info: A `TensorInfo` object containing attributes to create the graph
      tensor.
    name: A string (or None). The created graph tensor uses this name.

  Returns:
    A named tuple `_TemporaryAnalyzerOutputWrapper` with:
      eager_asset_path: If the analyzer output is an asset file, an eager tensor
        pointing to the file path. Else, None.
      graph_tensor: The graph tensor representing the analyzer output.
  """
  asset = None
  with tf.name_scope('temporary_analyzer_output'):
    temporary_asset_info = tensor_info.temporary_asset_info
    is_asset_filepath = temporary_asset_info is not None
    if is_asset_filepath:
      # Placeholders cannot be used for assets, if this graph will be serialized
      # to a SavedModel, as they will be initialized with the init op. If a
      # `temp_dir` is provided, it is assumed that this graph will be
      # serialized and a temporary asset file is written out. Else, a
      # placeholder is returned.
      # TODO(b/149997088): Reduce number of temporary files written out.
      if temp_dir:
        with tf.init_scope():
          temporary_asset_filepath = _write_to_temporary_asset_file(
              temp_dir, temporary_asset_info)
          asset = tf.constant(temporary_asset_filepath)
        graph_tensor = tf.constant(
            temporary_asset_filepath,
            dtype=tensor_info.dtype,
            shape=tensor_info.shape,
            name=name)
      else:
        graph_tensor = tf.raw_ops.Placeholder(
            dtype=tensor_info.dtype, shape=tensor_info.shape, name=name)
    else:
      # Using a placeholder with no default value causes tracing to fail if
      # there is any control flow dependent on a child tensor of this
      # placeholder. Hence, provide a temporary default value for it.
      # If dtype is string, we want a tensor that contains '0's instead of b'[]
      # to allow string to numeric conversion ops to trace successfully.
      temporary_dtype = (
          tf.int64 if tensor_info.dtype == tf.string else tensor_info.dtype)
      temporary_tensor = tf2_utils.supply_missing_tensor(
          1, tf.TensorShape(tensor_info.shape), temporary_dtype)
      if tensor_info.dtype == tf.string:
        temporary_tensor = tf.strings.as_string(temporary_tensor)
      graph_tensor = tf.raw_ops.PlaceholderWithDefault(
          input=temporary_tensor, shape=tensor_info.shape, name=name)
    return _TemporaryAnalyzerOutputWrapper(asset, graph_tensor)


def _bind_future_as_tensor_v2(
    future: nodes.ValueNode,
    tensor_info: TensorInfo,
    name: Optional[str] = None) -> common_types.TemporaryAnalyzerOutputType:
  """Bind a future value as a tensor to a TF2 FuncGraph.

    If the future is expected to write out an asset file and this method is
    invoked within a `TFGraphContext` that was provided a temporary directory,
    a temporary file is written out by this method.

    This could write out a significant number of temporary files depending on
    number of times the `preprocessing_fn` is traced and number of asset files
    in each tracing.

  Args:
    future: Future whose result should replace the graph tensor to which its
      bound.
    tensor_info: A `TensorInfo` object containing attributes to create the graph
      tensor.
    name: (Optional) If provided, the graph tensor created uses this name.

  Returns:
    A graph tensor or `tf.saved_model.Asset` that this future is bound to. If
    this future has already been evaluated in a previous TFT phase, it is
    directly returned.
  """
  graph = ops.get_default_graph()
  temp_dir = TFGraphContext.get_or_create_temp_dir()
  temporary_analyzer_info = _get_temporary_analyzer_output(
      temp_dir, tensor_info, name)
  is_asset_filepath = tensor_info.temporary_asset_info is not None

  # TODO(b/149997088): Switch to using a counter instead of tensor names.
  # Check if an evaluated value exists for this analyzer node.
  evaluated_replacements = TFGraphContext.get_evaluated_replacements()
  # evaluated_replacements is a dictionary from placeholder name to evaluated
  # tensor.
  # If `preprocessing_fn` was traced previously and this future was then
  # evaluated in a TFT phase, the result will be present in this dictionary.
  analyzer_name = temporary_analyzer_info.graph_tensor.name
  tensor_sink = TensorSink(temporary_analyzer_info.graph_tensor, future,
                           is_asset_filepath)
  graph.add_to_collection(ALL_REPLACEMENTS, tensor_sink)
  if (evaluated_replacements is not None and
      analyzer_name in evaluated_replacements):
    replaced_result = evaluated_replacements[analyzer_name]
    if is_asset_filepath:
      graph.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS,
                              replaced_result)
      return replaced_result
    else:
      # Without the identity wrapper some V2 tests fail with AttributeError:
      # Tensor.name is meaningless when eager execution is enabled.
      # TODO(b/149997088): Remove the identity wrapper once we no longer rely on
      # tensor names.
      return tf.identity(replaced_result)
  else:
    graph.add_to_collection(TENSOR_REPLACEMENTS, tensor_sink)
    eager_asset_path = temporary_analyzer_info.eager_asset_path
    if is_asset_filepath and eager_asset_path is not None:
      tf_utils.track_asset_analyzer_output(eager_asset_path,
                                           temporary_analyzer_info.graph_tensor)
      graph.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS,
                              eager_asset_path)
    return temporary_analyzer_info.graph_tensor


def bind_future_as_tensor(
    future: nodes.ValueNode,
    tensor_info: TensorInfo,
    name: Optional[str] = None) -> common_types.TemporaryAnalyzerOutputType:
  """Bind a future value as a tensor."""
  # TODO(b/165884902): Use tf.inside_function after dropping TF 2.3 support.
  if isinstance(ops.get_default_graph(), func_graph.FuncGraph):
    # If the default graph is a `FuncGraph`, tf.function was used to trace the
    # preprocessing fn.
    return _bind_future_as_tensor_v2(future, tensor_info, name)
  else:
    return _bind_future_as_tensor_v1(future, tensor_info, name)


def wrap_as_tensor(
    output_value_node: nodes.ValueNode
) -> common_types.TemporaryAnalyzerOutputType:
  analyzer_def = output_value_node.parent_operation.operation_def
  assert isinstance(analyzer_def, AnalyzerDef)
  return bind_future_as_tensor(
      output_value_node,
      analyzer_def.output_tensor_infos[output_value_node.value_index])


class Combiner:
  """Analyze using combiner function.

  This object mirrors a beam.CombineFn, that will receive a beam PCollection
  representing the batched input tensors.
  """

  def __repr__(self):
    return '<{}>'.format(self.__class__.__name__)

  def create_accumulator(self):
    """Return a fresh, empty accumulator.

    Returns: An empty accumulator.  This can be any Python value.
    """
    raise NotImplementedError

  def add_input(self, accumulator, batch_values):
    """Return result of folding a batch of inputs into accumulator.

    Args:
      accumulator: the current accumulator
      batch_values: A list of ndarrays representing the values of the inputs for
          a batch, which should be added to the accumulator.

    Returns: An accumulator that includes the batch of inputs.
    """
    raise NotImplementedError

  def merge_accumulators(self, accumulators):
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: the accumulators to merge

    Returns: The sole merged accumulator.
    """
    raise NotImplementedError

  def compact(self, accumulator):
    """Returns an equivalent but more compact represenation of the accumulator.

    Args:
      accumulator: the current accumulator.

    Returns: A more compact accumulator.
    """
    return accumulator

  def extract_output(self, accumulator):
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: the final accumulator value.

    Returns: A list of ndarrays representing the result of this combiner.
    """
    raise NotImplementedError

  def output_tensor_infos(self):
    """Return the number / types of outputs that are produced by extract_output.

    Returns: An iterable of `TensorInfo` describing how the outputs that
      extract_output will produce should be wrapped as `Tensor`s.

    Types are required to be TensorFlow dtypes.
    """
    raise NotImplementedError

  @property
  def accumulator_coder(self):
    return JsonNumpyCacheCoder()


class CacheCoder(metaclass=abc.ABCMeta):
  """A coder iterface for encoding and decoding cache items."""

  def __repr__(self):
    return '<{}>'.format(self.__class__.__name__)

  @abc.abstractmethod
  def encode_cache(self, cache):
    pass

  @abc.abstractmethod
  def decode_cache(self, encoded_cache):
    pass


class JsonNumpyCacheCoder(CacheCoder):
  """An accumulator cache coder that can handle lists."""

  def _convert_numpy_dtype(self, x):
    if hasattr(x, 'tolist'):
      return x.tolist()
    return x

  def encode_cache(self, accumulator):
    if isinstance(accumulator, (list, tuple)):
      primitive_accumulator = [
          self._convert_numpy_dtype(a) for a in accumulator
      ]
    else:
      primitive_accumulator = self._convert_numpy_dtype(accumulator)
    # Need to wrap in np.array and call tolist to make it JSON serializable.
    return tf.compat.as_bytes(json.dumps(primitive_accumulator))

  def decode_cache(self, encoded_accumulator):
    return np.array(json.loads(tf.compat.as_text(encoded_accumulator)))


class AnalyzerDef(nodes.OperationDef, metaclass=abc.ABCMeta):
  """A subclass of OperationDef whose outputs can be constant tensors.

  An AnalyzerDef is an OperationDef that also provides enough information to
  wrap each of its outputs as constant `Tensor`s in the graph.  By inserting
  the output of the AnalyzerDef back into the graph, the user can define
  multiple levels of anaylsis and transformation.

  All `OperationDef`s are placeholders for operations that will be implemented
  as `beam.PTransform`s.  This is done by a registration system.  The subclasses
  defined below that inherit from `AnalyzerDef` have there implementations
  registered in the module `tensorflow_transform.beam.analyzer_impls`.
  """

  @property
  @abc.abstractmethod
  def output_tensor_infos(self):
    """A description on how to wrap the outputs of this AnalyzerDef.

    An `OperationDef` defines the number of outputs it creates.  An
    `AnalyzerDef` must implemented this property that defines not only the
    number of outputs but how to wrap each output as a tensor.
    """
    pass

  @property
  def num_outputs(self):
    """The number of outputs returned by this operation."""
    return len(self.output_tensor_infos)


# We do the packing of combiners after the caching optimization. Hence, we don't
# name the packed operations as cacheable. The rationale behind doing the
# combiner packing after the cache optimization is that this optimization is
# more of a Beam execution level optimization and we want to keep it towards the
# end. So that, once Beam can automatically pack combines, we can remove this.
class PackedCombineAccumulate(
    tfx_namedtuple.namedtuple('PackedCombineAccumulate',
                              ['combiners', 'label']), nodes.OperationDef):
  """An analyzer that packs a list of combiners into a single beam CombineFn.

  Fields:
    combiners:  A list of `analysis_graph_builder._CombinerOpWrapper` objects.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, combiners, label):
    return super(PackedCombineAccumulate, cls).__new__(
        cls, combiners=combiners, label=_make_label(cls, label))

  @property
  def num_outputs(self):
    return 1

  # Note that this will not have any effect as packing of combiners is done
  # after the caching optimization.
  @property
  def is_partitionable(self):
    return True


class PackedCombineMerge(
    tfx_namedtuple.namedtuple('PackedCombineMerge', ['combiners', 'label']),
    nodes.OperationDef):
  """An analyzer that packs a list of combiners into a single beam CombineFn.

  Fields:
    combiners:  A list of `analysis_graph_builder._CombinerOpWrapper` objects.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, combiners, label):
    return super(PackedCombineMerge, cls).__new__(
        cls, combiners=combiners, label=_make_label(cls, label))

  @property
  def num_outputs(self):
    return 1


class CacheableCombineAccumulate(
    tfx_namedtuple.namedtuple('CacheableCombineAccumulate',
                              ['combiner', 'label']), nodes.OperationDef):
  """An analyzer that runs a beam CombineFn to accumulate without merging.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.Combine`.

  Fields:
    combiner: The Combiner to be applies to the inputs.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, combiner):
    return super(CacheableCombineAccumulate, cls).__new__(
        cls, combiner=combiner, label=_make_label(cls))

  @property
  def num_outputs(self):
    return 1

  @property
  def is_partitionable(self):
    return True

  @property
  def cache_coder(self):
    return self.combiner.accumulator_coder


class CacheableCombineMerge(
    tfx_namedtuple.namedtuple('CacheableCombineMerge', ['combiner', 'label']),
    nodes.OperationDef):
  """An analyzer that runs a beam CombineFn to only merge computed accumulators.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.Combine`.

  Fields:
    combiner: The Combiner to be applied to the inputs.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, combiner):
    return super(CacheableCombineMerge, cls).__new__(
        cls, combiner=combiner, label=_make_label(cls))

  @property
  def num_outputs(self):
    return 1


class _CombinerPerKeyAccumulatorCoder(CacheCoder):
  """Coder for per-key combiner accumulators."""

  def __init__(self, value_coder):
    self._combiner_coder = value_coder
    self._vocabulary_coder = _BaseKVCoder()
    super().__init__()

  def __repr__(self):
    return '<{}[{}[{}]]>'.format(self.__class__.__name__,
                                 repr(self._vocabulary_coder),
                                 repr(self._combiner_coder))

  def encode_cache(self, accumulator):
    key, value = accumulator
    encoded_value = self._combiner_coder.encode_cache(value)
    return self._vocabulary_coder.encode_cache((key, encoded_value))

  def decode_cache(self, encoded_accumulator):
    accumulator = self._vocabulary_coder.decode_cache(encoded_accumulator)
    key, encoded_value = accumulator
    value = self._combiner_coder.decode_cache(encoded_value)
    return (key, value)


class CacheableCombinePerKeyAccumulate(
    tfx_namedtuple.namedtuple('CacheableCombinePerKeyAccumulate',
                              ['combiner', 'label']), AnalyzerDef):
  """An analyzer that runs `beam.CombinePerKey` to accumulate without merging.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.CombinePerKey`.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._IntermediateAccumulateCombineImpl`.

  Fields:
    combiner: The Combiner to be applied to the inputs.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, combiner):
    return super(CacheableCombinePerKeyAccumulate, cls).__new__(
        cls, combiner=combiner, label=_make_label(cls))

  @property
  def num_outputs(self):
    return 1

  @property
  def is_partitionable(self):
    return True

  @property
  def cache_coder(self):
    return _CombinerPerKeyAccumulatorCoder(self.combiner.accumulator_coder)


class CacheableCombinePerKeyMerge(
    tfx_namedtuple.namedtuple('CacheableCombinePerKeyMerge',
                              ['combiner', 'label']), nodes.OperationDef):
  """An analyzer that runs `beam.CombinePerKey` to only merge accumulators.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.CombinePerKey`.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._MergeAccumulatorsCombinePerKeyImpl`

  Fields:
    combiner: The Combiner to use for merging and extracting outputs.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, combiner):
    return super(CacheableCombinePerKeyMerge, cls).__new__(
        cls, combiner=combiner, label=_make_label(cls))


class CacheableCombinePerKeyFormatKeys(
    tfx_namedtuple.namedtuple('CacheableCombinePerKeyFormatKeys',
                              ['combiner', 'label']), AnalyzerDef):
  """An analyzer that formats output for the non-stored per-key case.

  This analyzer converts the (key, output) pairs into a tuple of keys (of type
  string) and outputs.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._CombinePerKeyFormatKeysImpl`

  Fields:
    combiner: The Combiner to use for extracting outputs.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, combiner):
    return super(CacheableCombinePerKeyFormatKeys, cls).__new__(
        cls, combiner=combiner, label=_make_label(cls))

  @property
  def output_tensor_infos(self):
    # Returns a key vocab and one output per combiner output.
    return [TensorInfo(tf.string, (None,), None)] + [
        TensorInfo(info.dtype, (None,) + info.shape, info.temporary_asset_info)
        for info in self.combiner.output_tensor_infos()
    ]


class CacheableCombinePerKeyFormatLarge(
    tfx_namedtuple.namedtuple('CacheableCombinePerKeyFormatLarge', ['label']),
    nodes.OperationDef):
  """An analyzer that formats output prior to writing to file for per-key case.

  This operation operates on the output of CacheableCombinePerKeyAccumulate and
  is implemented by `tensorflow_transform.beam.analyzer_impls.
  _CombinePerKeyFormatLargeImpl`.
  """
  __slots__ = ()

  def __new__(cls):
    return super(CacheableCombinePerKeyFormatLarge, cls).__new__(
        cls, label=_make_label(cls))

  @property
  def num_outputs(self):
    return 1


class ScaleAndFlattenPerKeyBucketBouandaries(
    tfx_namedtuple.namedtuple('PostProcessPerKeyBucketBoundaries',
                              ['output_tensor_dtype', 'label']), AnalyzerDef):
  """An analyzer which takes quantile boundaries per key and combines them.

  It receives a 2-d array of boundaries, computes scales and shifts to each
  row separately, a new boundaries 1-d array which is a combination of
  boundaries for all the keys, and the number of buckets defined for each key.

  This outputs boundaries, scale_factor_per_key, shift_per_key, num_buckets.

  For example, for an input boundaries matrix, [[0, 1, 2], [0, 1, 2]] it will
  return:
  boundaries: [0, 0.5, 1, 1.5, 2]
  scale_factor_per_key: [0.5, 0.5]
  shift_per_key: [0, 1]
  num_buckets: 4

  So the transformation of each input x before computing its bucket should be:
  F(x, key) = x * scale_factor_per_key[key] + shift_per_key[key]
  """
  __slots__ = ()

  def __new__(cls, output_tensor_dtype):
    return super(ScaleAndFlattenPerKeyBucketBouandaries, cls).__new__(
        cls, output_tensor_dtype=output_tensor_dtype, label=_make_label(cls))

  @property
  def output_tensor_infos(self):
    # Boundaries, scale_factor_per_key, shift_per_key, num_buckets.
    return [TensorInfo(self.output_tensor_dtype,
                       (None,), None)] * 3 + [TensorInfo(tf.int64, (), None)]


class VocabularyAccumulate(
    tfx_namedtuple.namedtuple('VocabularyAccumulate',
                              ['vocab_ordering_type', 'input_dtype', 'label']),
    nodes.OperationDef):
  """An operation that accumulates unique words with their frequency or weight.

  This operation is implemented by
  `tensorflow_transform.beam.analyzer_impls._VocabularyAccumulateImpl`.
  """
  __slots__ = ()

  def __new__(cls, vocab_ordering_type, input_dtype=tf.string.name):
    return super(VocabularyAccumulate, cls).__new__(
        cls,
        vocab_ordering_type=vocab_ordering_type,
        input_dtype=input_dtype,
        label=_make_label(cls))

  @property
  def num_outputs(self):
    return 1

  @property
  def is_partitionable(self):
    return True

  @property
  def cache_coder(self):
    return _VocabularyAccumulatorCoder(input_dtype=self.input_dtype)


class _BaseKVCoder(CacheCoder):
  """Coder for key-value based accumulators."""

  def __init__(self):
    self._lengths_prefix_format = 'qq'
    self._lengths_prefix_length = struct.calcsize(self._lengths_prefix_format)
    super().__init__()

  def encode_cache(self, accumulator):
    token, value = accumulator
    len_token, len_value = len(token), len(value)
    return struct.pack(
        '{}{}s{}s'.format(self._lengths_prefix_format, len_token, len_value),
        len_token, len_value, token, value)

  def decode_cache(self, encoded_accumulator):
    (len_token, len_value) = struct.unpack_from(
        self._lengths_prefix_format,
        encoded_accumulator[:self._lengths_prefix_length])
    accumulator = struct.unpack_from(
        '{}s{}s'.format(len_token, len_value),
        encoded_accumulator[self._lengths_prefix_length:])
    return accumulator


class _VocabularyAccumulatorCoder(_BaseKVCoder):
  """Coder for vocabulary accumulators."""

  def __init__(self, input_dtype=tf.string.name):
    self._input_dtype = tf.dtypes.as_dtype(input_dtype)
    super().__init__()

  def encode_cache(self, accumulator):
    token, value = accumulator
    if self._input_dtype is not tf.string:
      token = tf.compat.as_bytes(json.dumps(token))
    # If the value is a _WeightedMeanAndVarAccumulator, cast each field to a
    # list for serialization.
    if isinstance(value, tuple):
      value = [
          a.tolist()
          for a in (value.count, value.mean, value.variance, value.weight)
      ]
    value = tf.compat.as_bytes(json.dumps(value))
    return super().encode_cache((token, value))

  def decode_cache(self, encoded_accumulator):
    accumulator = super().decode_cache(encoded_accumulator)
    token, value = accumulator
    if self._input_dtype is not tf.string:
      token = json.loads(tf.compat.as_text(token))

    value = json.loads(tf.compat.as_text(value))
    if isinstance(value, list):
      # If the value is a _WeightedMeanAndVarAccumulator (serialized to tuple),
      # cast each field back to a np.array.
      (count, mean, variance, weight) = value
      value = (np.array(count), np.array(mean), np.array(variance),
               np.array(weight))
    return token, value


class VocabularyCount(
    tfx_namedtuple.namedtuple('VocabularyCount', ['label']),
    nodes.OperationDef):
  """An operation counts the total number of tokens in a vocabulary.

  This operation takes in the output of VocabularyAccumulate and is implemented
  by `tensorflow_transform.beam.analyzer_impls._VocabularyCountImpl`.

  The output of this operation is a singleton Integer.

  Fields:
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, label):
    return super().__new__(cls, label=_make_label(cls, label))

  @property
  def num_outputs(self):
    return 1


class VocabularyMerge(
    tfx_namedtuple.namedtuple('VocabularyMerge', [
        'vocab_ordering_type', 'use_adjusted_mutual_info', 'min_diff_from_avg',
        'label'
    ]), nodes.OperationDef):
  """An operation that merges the accumulators produced by VocabularyAccumulate.

  This operation operates on the output of VocabularyAccumulate and is
  implemented by `tensorflow_transform.beam.analyzer_impls._VocabularyMergeImpl`
  .

  See `tft.vocabulary` for a description of the parameters.
  """
  __slots__ = ()

  def __new__(cls, vocab_ordering_type, use_adjusted_mutual_info,
              min_diff_from_avg):
    return super(VocabularyMerge, cls).__new__(
        cls,
        vocab_ordering_type=vocab_ordering_type,
        use_adjusted_mutual_info=use_adjusted_mutual_info,
        min_diff_from_avg=min_diff_from_avg,
        label=_make_label(cls))

  @property
  def num_outputs(self):
    return 1


class VocabularyPrune(
    tfx_namedtuple.namedtuple('VocabularyPrune', [
        'top_k', 'frequency_threshold', 'informativeness_threshold',
        'coverage_top_k', 'coverage_frequency_threshold',
        'coverage_informativeness_threshold', 'key_fn', 'input_dtype', 'label'
    ]), nodes.OperationDef):
  """An operation that filters and orders a computed vocabulary.

  This operation operates on the output of VocabularyMerge and is implemented by
  `tensorflow_transform.beam.analyzer_impls._VocabularyPruneImpl`.

  See `tft.vocabulary` for a description of the parameters.
  """
  __slots__ = ()

  def __new__(cls,
              top_k,
              frequency_threshold,
              input_dtype,
              informativeness_threshold=float('-inf'),
              coverage_top_k=None,
              coverage_frequency_threshold=0,
              coverage_informativeness_threshold=float('-inf'),
              key_fn=None):
    return super(VocabularyPrune, cls).__new__(
        cls,
        top_k=top_k,
        frequency_threshold=frequency_threshold,
        informativeness_threshold=informativeness_threshold,
        coverage_top_k=coverage_top_k,
        coverage_frequency_threshold=coverage_frequency_threshold,
        coverage_informativeness_threshold=coverage_informativeness_threshold,
        key_fn=key_fn,
        input_dtype=input_dtype,
        label=_make_label(cls))

  @property
  def num_outputs(self):
    return 1


class VocabularyOrderAndWrite(
    tfx_namedtuple.namedtuple('VocabularyOrderAndWrite', [
        'vocab_filename', 'store_frequency', 'input_dtype', 'label',
        'fingerprint_shuffle', 'file_format', 'input_is_sorted'
    ]), AnalyzerDef):
  """An analyzer that writes vocabulary files from an accumulator.

  This operation operates on the output of VocabularyPrune and is implemented by
  `tensorflow_transform.beam.analyzer_impls._VocabularyOrderAndWriteImpl`.

  See `tft.vocabulary` for a description of the parameters.
  """
  __slots__ = ()

  def __new__(cls,
              vocab_filename,
              store_frequency,
              fingerprint_shuffle,
              file_format,
              input_dtype=tf.string.name,
              input_is_sorted=False):
    return super(VocabularyOrderAndWrite, cls).__new__(
        cls,
        vocab_filename=vocab_filename,
        store_frequency=store_frequency,
        fingerprint_shuffle=fingerprint_shuffle,
        file_format=file_format,
        input_dtype=input_dtype,
        input_is_sorted=input_is_sorted,
        label=_make_label(cls))

  @property
  def output_tensor_infos(self):
    # Define temporary data for this node to write to a file before the actual
    # vocab file is evaluated and written out.
    temporary_asset_value = (b'TEMPORARY_ASSET_VALUE' if tf.dtypes.as_dtype(
        self.input_dtype) == tf.string else b'-777777')
    if self.store_frequency:
      temporary_asset_value = b'1 %s' % temporary_asset_value

    return [
        TensorInfo(tf.string, [],
                   TemporaryAssetInfo(temporary_asset_value, self.file_format))
    ]


class PTransform(
    tfx_namedtuple.namedtuple('PTransform', [
        'ptransform', 'output_tensor_info_list', 'is_partitionable',
        'cache_coder', 'label'
    ]), AnalyzerDef):
  """(Experimental) OperationDef for PTransform anaylzer.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._PTransformImpl`.

  Fields:
    ptransform: The `beam.PTransform` to be applied to the inputs.
    output_tensor_info_list: A list of `TensorInfo`s that defines the outputs of
        this `PTransform`.
    is_partitionable: Whether or not this PTransform is partitionable.
    cache_coder: (optional) A `CacheCoder` instance.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls,
              ptransform: Any,
              output_tensor_info_list: Sequence[TensorInfo],
              is_partitionable: bool,
              cache_coder: Optional[CacheCoder] = None):
    return super(PTransform, cls).__new__(
        cls,
        ptransform=ptransform,
        output_tensor_info_list=output_tensor_info_list,
        is_partitionable=is_partitionable,
        cache_coder=cache_coder,
        label=_make_label(cls))

  @property
  def output_tensor_infos(self):
    return self.output_tensor_info_list


class EncodeCache(
    tfx_namedtuple.namedtuple('EncodeCache', ['coder', 'label']),
    nodes.OperationDef):
  """OperationDef for encoding a cache instance.

  Fields:
    coder: An instance of CacheCoder used to encode cache.
    label: A unique label for this operation.
  """
  __slots__ = ()

  @property
  def is_partitionable(self):
    return True


class DecodeCache(
    tfx_namedtuple.namedtuple('DecodeCache',
                              ['dataset_key', 'cache_key', 'coder', 'label']),
    nodes.OperationDef):
  """OperationDef for decoding a cache instance.

  Fields:
    coder: An instance of CacheCoder used to decode cache.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def get_field_str(self, field_name):
    if field_name == 'cache_key':
      return '<bytes>'
    return super().get_field_str(field_name)

  @property
  def is_partitionable(self):
    return True


class AddKey(
    tfx_namedtuple.namedtuple('AddKey', ['key', 'label']), nodes.OperationDef):
  """An operation that represents adding a key to a value.

  This operation represents a `beam.Map` that is applied to a PCollection.
  For each element of the PCollection, this corresponding element of the output
  PCollection is a tuple of (key, value).

  Attributes:
    key: The key which should be added to each element of the input PCollection.
    label: A unique label for this operation.
  """
  __slots__ = ()

  @property
  def is_partitionable(self):
    return True


class FlattenLists(
    tfx_namedtuple.namedtuple('FlattenLists', ['label']), nodes.OperationDef):
  """An operation that represents flattening a PCollection of lists.

  Attributes:
    label: A unique label for this operation.
  """

  def __new__(cls):
    return super(FlattenLists, cls).__new__(cls, label=_make_label(cls))

  @property
  def is_partitionable(self):
    return True


class ExtractCombineMergeOutputs(
    tfx_namedtuple.namedtuple('ExtractOutputs',
                              ['output_tensor_info_list', 'label']),
    AnalyzerDef):
  """An operation that represents extracting outputs of a combine merge.

  This operation represents a `beam.Map` that is applied to a PCollection.
  For each element of the PCollection, this corresponding element of the output
  PCollection is a tuple of outputs.

  Attributes:
    output_tensor_info_list: A list of `TensorInfo`s that defines the outputs of
      this operation.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def __new__(cls, output_tensor_info_list):
    return super(ExtractCombineMergeOutputs, cls).__new__(
        cls,
        output_tensor_info_list=output_tensor_info_list,
        label=_make_label(cls))

  @property
  def output_tensor_infos(self):
    return self.output_tensor_info_list


class ExtractPackedCombineMergeOutputs(
    tfx_namedtuple.namedtuple('ExtractOutputs',
                              ['output_tensor_info_list', 'label']),
    AnalyzerDef):
  """An operation that represents extracting outputs of a packed combine merge.

  This operation represents a `beam.Map` that is applied to a PCollection.
  For each element of the PCollection, this corresponding element of the output
  PCollection is a tuple of outputs.

  Attributes:
    output_tensor_info_list: A list of `TensorInfo`s that defines the outputs of
      this operation.
    label: A unique label for this operation.
  """
  __slots__ = ()

  @property
  def output_tensor_infos(self):
    return self.output_tensor_info_list
