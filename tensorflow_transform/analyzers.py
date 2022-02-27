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
"""Functions that involve a full pass over the dataset.

This module contains functions that are used in the preprocessing function, to
define a full pass operation such as computing the sum, min, max or unique
values of a tensor over the entire dataset.  This is implemented by a reduction
operation in the Beam implementation.

From the user's point of view, an analyzer appears as a regular TensorFlow
function, i.e. it accepts and returns tensors.  However it is represented in
the graph as a `Analyzer` which is not a TensorFlow op, but a placeholder for
the computation that takes place outside of TensorFlow.
"""

import functools
import os
import pickle
import re
from typing import Any, Callable, Collection, List, Optional, Tuple, Union
from absl import logging


import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import annotators
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import gaussianization
from tensorflow_transform import nodes
from tensorflow_transform import schema_inference
from tensorflow_transform import tf_utils
from tfx_bsl import sketches
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple
from typing_extensions import Literal

from google.protobuf import descriptor_pb2

__all__ = [
    'count_per_key',
    'covariance',
    'histogram',
    'max',
    'mean',
    'min',
    'pca',
    'quantiles',
    'size',
    'sum',
    'tukey_location',
    'tukey_scale',
    'tukey_h_params',
    'var',
    'vocabulary',
]

# This module defines max and min functions that override the builtins.
builtin_max = max
builtin_min = min


DEFAULT_VOCABULARY_FILE_FORMAT: Literal['text'] = 'text'
ALLOWED_VOCABULARY_FILE_FORMATS = ('text', 'tfrecord_gzip')

VOCAB_FILENAME_PREFIX = 'vocab_'
VOCAB_FREQUENCY_FILENAME_PREFIX = 'vocab_frequency_'

# Experimentally estimated value of top_k after which the exact `tft.vocabulary`
# implementation becomes more efficient than
# `tft.experimental.approximate_vocabulary`.
LARGE_VOCAB_TOP_K = 200_000

# Matches empty strings and strings with \n or \r (including strings with \n or
# \r that contain invalid UTF-8 characters). This has to follow the re2 syntax:
# https://github.com/google/re2/wiki/Syntax.
_EMPTY_STRING_OR_NEWLINE_CHARS_REGEX = r'^$|\C*[\n\r]\C*'

# For some input types, widen the output type of sum analyzer to avoid overflow.
_SUM_OUTPUT_DTYPE_MAP = {
    tf.float16: tf.float32,
    tf.float32: tf.float32,
    tf.float64: tf.float64,
    tf.int8: tf.int64,
    tf.int16: tf.int64,
    tf.int32: tf.int64,
    tf.int64: tf.int64,
    tf.uint8: tf.uint64,
    tf.uint16: tf.uint64,
    tf.uint32: tf.uint64,
    tf.uint64: tf.uint64,
}

_FLOAT_OUTPUT_DTYPE_MAP = {
    tf.float16: tf.float16,
    tf.float32: tf.float32,
    tf.float64: tf.float64,
    tf.int8: tf.float32,
    tf.int16: tf.float32,
    tf.int32: tf.float32,
    tf.int64: tf.float32,
    tf.uint8: tf.float32,
    tf.uint16: tf.float32,
    tf.uint32: tf.float32,
    tf.uint64: tf.float32,
}


def apply_cacheable_combine_operation(
    combiner: analyzer_nodes.Combiner,
    *tensor_inputs: common_types.TensorType) -> Tuple[nodes.ValueNode, ...]:
  """Applies combine operation nodes over the whole dataset.

  Applied nodes are subject to analyzer cache optimization.

  Args:
    combiner: Combiner to be applied.
    *tensor_inputs: Tensors representing inputs to the combiner.

  Returns:
    A tuple of ValueNodes representing outputs of the combiner.
  """
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)

  accumulate_outputs_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombineAccumulate,
      input_values_node,
      combiner=combiner)

  merge_outputs_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombineMerge,
      *accumulate_outputs_value_nodes,
      combiner=combiner)

  return nodes.apply_multi_output_operation(
      analyzer_nodes.ExtractCombineMergeOutputs,
      *merge_outputs_value_nodes,
      output_tensor_info_list=combiner.output_tensor_infos())


def _apply_cacheable_combiner(
    combiner: analyzer_nodes.Combiner,
    *tensor_inputs: common_types.TensorType) -> Tuple[tf.Tensor, ...]:
  """Applies the combiner over the whole dataset possibly utilizing cache.

  Similar to above but returns a tuple of output tensors.

  Args:
    combiner: Combiner to be applied.
    *tensor_inputs: Tensors representing inputs to the combiner.

  Returns:
    A tuple of tensors representing outputs of the combiner.
  """
  outputs_value_nodes = apply_cacheable_combine_operation(
      combiner, *tensor_inputs)
  return tuple(map(analyzer_nodes.wrap_as_tensor, outputs_value_nodes))  # pytype: disable=bad-return-type


def _apply_cacheable_combiner_per_key(
    combiner: analyzer_nodes.Combiner,
    *tensor_inputs: common_types.TensorType) -> Tuple[tf.Tensor, ...]:
  """Similar to _apply_cacheable_combiner but this is computed per key."""
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)

  accumulate_outputs_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombinePerKeyAccumulate,
      input_values_node,
      combiner=combiner)

  merge_output_value_node = nodes.apply_operation(
      analyzer_nodes.CacheableCombinePerKeyMerge,
      *accumulate_outputs_value_nodes,
      combiner=combiner)

  output_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombinePerKeyFormatKeys,
      merge_output_value_node,
      combiner=combiner)

  return tuple(map(analyzer_nodes.wrap_as_tensor, output_value_nodes))


def _apply_cacheable_combiner_per_key_large(
    combiner: analyzer_nodes.Combiner, key_vocabulary_filename: str,
    *tensor_inputs: common_types.TensorType
) -> Union[tf.Tensor, common_types.Asset]:
  """Similar to above but saves the combined result to a file."""
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)

  accumulate_outputs_value_node = nodes.apply_operation(
      analyzer_nodes.CacheableCombinePerKeyAccumulate,
      input_values_node,
      combiner=combiner)

  merge_output_value_node = nodes.apply_operation(
      analyzer_nodes.CacheableCombinePerKeyMerge,
      accumulate_outputs_value_node,
      combiner=combiner)

  keys_and_values_node = nodes.apply_operation(
      analyzer_nodes.CacheableCombinePerKeyFormatLarge,
      merge_output_value_node)

  # `store_frequency` is True by default because we want to write some values
  # alongside the key "vocabulary". Without doing so it would be equivalent to
  # vanilla vocabulary analzyer. `fingerprint_shuffle` is not as important but
  # signifies that the values are not required to be ordered here.
  key_vocabulary_filename_node = nodes.apply_operation(
      analyzer_nodes.VocabularyOrderAndWrite,
      keys_and_values_node,
      vocab_filename=key_vocabulary_filename,
      store_frequency=True,
      fingerprint_shuffle=True,
      # TODO(b/62379925): Use tfrecord.
      file_format='text')

  return analyzer_nodes.wrap_as_tensor(key_vocabulary_filename_node)


class NumPyCombiner(analyzer_nodes.Combiner):
  """Combines the PCollection only on the 0th dimension using nparray.

  Attributes:
    fn: The numpy function representing the reduction to be done.
    default_accumulator_value: The default value each accumulator entry is
      initialized to.
    output_dtypes: The numpy dtype to cast each output to.
    output_shapes: List of tuples representing the shapes of the outputs or
      Nones if the shapes are not fully defined.
  """

  def __init__(self, fn, default_accumulator_value, output_dtypes,
               output_shapes):
    self._fn = fn
    self._default_accumulator_value = default_accumulator_value
    self._default_sub_accumulator = np.array(default_accumulator_value)
    self._output_dtypes = output_dtypes
    if not all(
        isinstance(shape, (tuple, type(None))) for shape in output_shapes):
      raise TypeError('Expected all tuples or Nones, but got %r' %
                      output_shapes)
    self._output_shapes = output_shapes
    if np.isnan(default_accumulator_value):
      # This case is needed because np.nan != np.nan.
      self._is_default_sub_accumulator = self._equals_to_scalar_nan
    else:
      self._is_default_sub_accumulator = self._equals_to_default_sub_accumulator

  def _equals_to_scalar_nan(self, array):
    return not array.shape and np.isnan(array)

  def _equals_to_default_sub_accumulator(self, array):
    # Note that `np.array_equal` below does at most per-element comparison of
    # 0-dim arrays since `_default_sub_accumulator` is a 0-dim array, and
    # `np.array_equal` exits early on a shape mismatch.
    return np.array_equal(array, self._default_sub_accumulator)

  def _is_default_sub_accumulator(self, array):
    raise NotImplementedError('Implementation should be set in __init__.')

  def create_accumulator(self):
    return [
        self._create_sub_accumulator(shape)
        for shape in self._output_shapes
    ]

  def _create_sub_accumulator(self, shape):
    # Returns a default subaccumulator of the given shape if it's fully defined
    # and a 0-dim default array otherwise.
    if shape is None:
      return self._default_sub_accumulator
    else:
      return np.full(shape, self._default_accumulator_value)

  def add_input(self, accumulator, batch_values):
    # TODO(b/112414577): Go back to accepting only a single input.
    # See comment in _numeric_combine.
    # If the first subaccumulator is default, then the accumulator is default
    # and can be discarded.
    if self._is_default_sub_accumulator(accumulator[0]):
      return batch_values
    else:
      return [
          self._fn((sub_accumulator, batch_value), axis=0)
          for sub_accumulator, batch_value in zip(accumulator, batch_values)
      ]

  def merge_accumulators(self, accumulators):
    # TODO(b/422923883): Operate in place on accumulators[0] or batch values
    # internally for vectorization benefits after AccumulateFn is in use.

    # If the first subaccumulator is default, then the accumulator is default
    # and can be discarded.
    non_default_accumulators = [
        accumulator for accumulator in accumulators
        if not self._is_default_sub_accumulator(accumulator[0])
    ]
    if non_default_accumulators:
      return [
          # numpy's sum, min, max, etc functions operate on array-like objects,
          # but not arbitrary iterables. Convert the provided sub_accumulators
          # into a list.
          self._fn(list(sub_accumulators), axis=0)
          for sub_accumulators in zip(*non_default_accumulators)
      ]
    else:
      return self.create_accumulator()

  def extract_output(self, accumulator):
    # For each output, cast that output to the specified type. Note there
    # will be one output for each input tensor to the analyzer.
    return [
        sub_accumulator.astype(output_dtype) for sub_accumulator, output_dtype
        in zip(accumulator, self._output_dtypes)
    ]

  def output_tensor_infos(self):
    return [
        analyzer_nodes.TensorInfo(tf.as_dtype(dtype), shape, None)
        for dtype, shape in zip(self._output_dtypes, self._output_shapes)
    ]


def _get_output_shape_from_input(x):
  if isinstance(x, tf.SparseTensor):
    return x.get_shape().as_list()[1:]

  # When reducing over batch dimensions, with known shape, the result will be
  # the same shape as the input, but without the batch.
  if x.shape.rank is not None:
    return x.shape.as_list()[1:]
  return (None,)


# TODO(b/112414577): Go back to accepting only a single input.
# Currently we accept multiple inputs so that we can implement min and max
# with a single combiner. Once this is done, add a return pytype as well.
def _numeric_combine(inputs: List[tf.Tensor],
                     fn: Callable[[np.ndarray], np.ndarray],
                     default_accumulator_value: Union[float, int],
                     reduce_instance_dims: bool = True,
                     output_dtypes: Optional[List[tf.DType]] = None,
                     key: Optional[tf.Tensor] = None,
                     key_vocabulary_filename: Optional[str] = None):
  """Apply a reduction, defined by a numpy function to multiple inputs.

  Args:
    inputs: A list of tensors, which will be independently reduced.
    fn: A function to reduce tensors across instances/batches, to get a single
        output.
    default_accumulator_value: The default scalar value that each accumulator
        entry is initialized to. Must be properly processed by the reduction
        function.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    output_dtypes: (Optional) A list of dtypes of the output tensors. If None,
        the output tensor has the same type as the input one.
    key: (Optional) Apply the same operation, but on a per-key basis.
    key_vocabulary_filename: (Optional) The file name for the key-output mapping
      file. If None and key are provided, this combiner assumes the keys fit in
      memory and will not store the result in a file. If empty string, a file
      name will be chosen based on the current scope. If not an empty string,
      should be unique within a given preprocessing function.

  Returns:
      Either:
      (A) A list of Tensors with the same length as `inputs`, representing the
          input Tensors that have been reduced by `fn` across instances and
          batches (if key_vocabulary_filename is None).
      (B) A Tensor with the filename where the key-value mapping is stored (if
          key_vocabulary_filename is not None).
  """
  for x in inputs:
    if not isinstance(x, tf.Tensor):
      raise TypeError('Expected a Tensor, but got %r' % x)
  if not np.isscalar(default_accumulator_value):
    raise TypeError('Expected a scalar, but got %r' % default_accumulator_value)

  if output_dtypes is None:
    output_dtypes = [x.dtype for x in inputs]
  if reduce_instance_dims:
    # If reducing over all dimensions, result is scalar.
    output_shapes = [() for _ in inputs]
  else:
    # Reducing over batch dimensions.
    output_shapes = [
        (tuple(x.get_shape()) if x.get_shape().is_fully_defined() else None)
        for x in inputs
    ]
  combiner = NumPyCombiner(fn, default_accumulator_value,
                           [dtype.as_numpy_dtype for dtype in output_dtypes],
                           output_shapes)
  if key is None:
    return _apply_cacheable_combiner(combiner, *inputs)

  if key_vocabulary_filename is None:
    return _apply_cacheable_combiner_per_key(combiner, key, *inputs)

  return _apply_cacheable_combiner_per_key_large(
      combiner, _maybe_get_per_key_vocab_filename(key_vocabulary_filename),
      key, *inputs)


@common.log_api_use(common.ANALYZER_COLLECTION)
def min(  # pylint: disable=redefined-builtin
    x: common_types.TensorType,
    reduce_instance_dims: bool = True,
    name: Optional[str] = None) -> tf.Tensor:
  """Computes the minimum of the values of a `Tensor` over the whole dataset.

  In the case of a `CompositeTensor` missing values will be used in return
  value: for float, NaN is used and for other dtypes the max is used.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
      to arrive at a single scalar output. If False, only collapses the batch
      dimension and outputs a `Tensor` of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` with the same type as `x`.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'min'):
    return _min_and_max(x, reduce_instance_dims, name)[0]


@common.log_api_use(common.ANALYZER_COLLECTION)
def max(  # pylint: disable=redefined-builtin
    x: common_types.TensorType,
    reduce_instance_dims: bool = True,
    name: Optional[str] = None) -> tf.Tensor:
  """Computes the maximum of the values of a `Tensor` over the whole dataset.

  In the case of a `CompositeTensor` missing values will be used in return
  value: for float, NaN is used and for other dtypes the min is used.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
      to arrive at a single scalar output. If False, only collapses the batch
      dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor`. Has the same type as `x`.
  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'max'):
    return _min_and_max(x, reduce_instance_dims, name)[1]


def _min_and_max(x: common_types.TensorType,
                 reduce_instance_dims: bool = True,
                 name: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes the min and max of the values of a `Tensor` or `CompositeTensor`.

  In the case of a `CompositeTensor` missing values will be used in return
  value:
  for float, NaN is used and for other dtypes the min is used.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
      to arrive at a single scalar output. If False, only collapses the batch
      dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    Two `Tensor`s. Both have the same type as `x`.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'min_and_max'):
    output_dtype = x.dtype
    if (not reduce_instance_dims and isinstance(x, tf.SparseTensor) and
        x.dtype.is_floating):
      combine_fn = np.nanmax
      default_accumulator_value = (np.nan if x.dtype.is_floating else
                                   -output_dtype.max)
    elif not reduce_instance_dims and isinstance(x, tf.RaggedTensor):
      raise NotImplementedError(
          'Elemenwise min_and_max does not support RaggedTensors.')
    else:
      combine_fn = np.max
      default_accumulator_value = (-np.inf if x.dtype.is_floating else
                                   -output_dtype.max)

    x_batch_minus_min, x_batch_max = tf_utils.reduce_batch_minus_min_and_max(
        x, reduce_instance_dims)

    minus_x_min, x_max = _numeric_combine(  # pylint: disable=unbalanced-tuple-unpacking
        inputs=[x_batch_minus_min, x_batch_max],
        fn=combine_fn,
        default_accumulator_value=default_accumulator_value,
        reduce_instance_dims=reduce_instance_dims)
    return tf.cast(0 - minus_x_min, output_dtype), tf.cast(x_max, output_dtype)


def _min_and_max_per_key(
    x: common_types.TensorType,
    key: common_types.TensorType,
    reduce_instance_dims: bool = True,
    key_vocabulary_filename: Optional[str] = None,
    name: Optional[str] = None
) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
  """Computes the min and max of the values of a `Tensor` or `CompositeTensor`.

  In the case of a `CompositeTensor` missing values will be used in return
  value: for float, NaN is used and for other dtypes the min is used.

  This function operates under the assumption that the size of the key set
  is small enough to fit in memory. Anything above a certain size larger is not
  guaranteed to be handled properly, but support for larger key sets may be
  available in a future version.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    key: A Tensor or `CompositeTensor` of dtype tf.string.  If `x` is a
      `CompositeTensor`, `key` must exactly match `x` in everything except
      values.
    reduce_instance_dims: By default collapses the batch and instance dimensions
      to arrive at a single scalar output. If False, only collapses the batch
      dimension and outputs a vector of the same shape as the input. The False
      case is not currently supported for _min_and_max_per_key.
    key_vocabulary_filename: (Optional) The file name for the key-output mapping
      file. If None and key are provided, this combiner assumes the keys fit in
      memory and will not store the result in a file. If empty string, a file
      name will be chosen based on the current scope. If not an empty string,
      should be unique within a given preprocessing function.
    name: (Optional) A name for this operation.

  Returns:
    Either:
    (A) Three `Tensor`s. The first is the key vocab of type tf.string, and the
        second two have same type as `x` (if key_vocabulary_filename is None).
    (B) The filename where the key-value mapping is stored (if
        key_vocabulary_filename is not None).

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  if key is None:
    raise ValueError('A key is required for _min_and_max_per_key')

  if not reduce_instance_dims:
    raise NotImplementedError('Per-key elementwise reduction not supported')

  with tf.compat.v1.name_scope(name, 'min_and_max_per_key'):
    output_dtype = x.dtype
    if (not reduce_instance_dims and
        isinstance(x,
                   (tf.SparseTensor, tf.RaggedTensor)) and x.dtype.is_floating):
      combine_fn = np.nanmax
      default_accumulator_value = (np.nan if x.dtype.is_floating else
                                   -output_dtype.max)
    else:
      combine_fn = np.max
      default_accumulator_value = (-np.inf if x.dtype.is_floating else
                                   -output_dtype.max)

    key_vocab, x_batch_minus_min, x_batch_max = (
        tf_utils.reduce_batch_minus_min_and_max_per_key(x, key))

    key_values = _numeric_combine(  # pylint: disable=unbalanced-tuple-unpacking
        inputs=[x_batch_minus_min, x_batch_max],
        fn=combine_fn,
        default_accumulator_value=default_accumulator_value,
        reduce_instance_dims=reduce_instance_dims,
        key=key_vocab,
        key_vocabulary_filename=key_vocabulary_filename)

    if key_vocabulary_filename is not None:
      return key_values

    key, minus_x_min, x_max = key_values
    return (
        key,
        tf.cast(0 - minus_x_min, output_dtype),
        tf.cast(x_max, output_dtype))


def _sum_combine_fn_and_dtype(
    input_dtype: tf.DType
) -> Tuple[tf.DType, Callable[[np.ndarray], np.ndarray]]:
  output_dtype = _SUM_OUTPUT_DTYPE_MAP.get(input_dtype)
  if output_dtype is None:
    raise TypeError('Tensor type %r is not supported' % input_dtype)

  return output_dtype, functools.partial(
      np.sum, dtype=output_dtype.as_numpy_dtype)


@common.log_api_use(common.ANALYZER_COLLECTION)
def sum(  # pylint: disable=redefined-builtin
    x: common_types.TensorType,
    reduce_instance_dims: bool = True,
    name: Optional[str] = None) -> tf.Tensor:
  """Computes the sum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor` or `CompositeTensor`. Its type must be floating point
        (float{16|32|64}),integral (int{8|16|32|64}), or
        unsigned integral (uint{8|16})
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the sum. If `x` is float32 or float64, the sum will
    have the same type as `x`. If `x` is float16, the output is cast to float32.
    If `x` is integral, the output is cast to [u]int64. If `x` is sparse and
    reduce_inst_dims is False will return 0 in place where column has no values
    across batches.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'sum'):
    if reduce_instance_dims:
      x = tf.reduce_sum(input_tensor=tf_utils.get_values(x))
    elif isinstance(x, tf.SparseTensor):
      if x.dtype == tf.uint8 or x.dtype == tf.uint16:
        x = tf.cast(x, tf.int64)
      elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
        TypeError('Data type %r is not supported' % x.dtype)
      x = tf.sparse.reduce_sum(x, axis=0)
    elif isinstance(x, tf.RaggedTensor):
      raise NotImplementedError(
          'Elementwise sum does not support RaggedTensors.')
    else:
      x = tf.reduce_sum(input_tensor=x, axis=0)
    output_dtype, sum_fn = _sum_combine_fn_and_dtype(x.dtype)
    return _numeric_combine(
        inputs=[x],
        fn=sum_fn,
        default_accumulator_value=0,
        reduce_instance_dims=reduce_instance_dims,
        output_dtypes=[output_dtype])[0]


def remove_leftmost_boundary(boundaries: tf.Tensor) -> tf.Tensor:
  """Removes the leftmost boundary from [1, None]-shaped `Tensor` of buckets."""
  return boundaries[:, 1:]


@common.log_api_use(common.ANALYZER_COLLECTION)
def histogram(x: common_types.TensorType,
              boundaries: Optional[Union[tf.Tensor, int]] = None,
              categorical: Optional[bool] = False,
              name: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes a histogram over x, given the bin boundaries or bin count.

  Ex (1):
  counts, boundaries = histogram([0, 1, 0, 1, 0, 3, 0, 1], range(5))
  counts: [4, 3, 0, 1, 0]
  boundaries: [0, 1, 2, 3, 4]

  Ex (2):
  Can be used to compute class weights.
  counts, classes = histogram([0, 1, 0, 1, 0, 3, 0, 1], categorical=True)
  probabilities = counts / tf.reduce_sum(counts)
  class_weights = dict(map(lambda (a, b): (a.numpy(), 1.0 / b.numpy()),
                           zip(classes, probabilities)))

  Args:
    x: A `Tensor` or `CompositeTensor`.
    boundaries: (Optional) A `Tensor` or `int` used to build the histogram;
      ignored if `categorical` is True. If possible, provide boundaries as
      multiple sorted values.  Default to 10 intervals over the 0-1 range, or
      find the min/max if an int is provided (not recommended because
      multi-phase analysis is inefficient).
    categorical: (Optional) A `bool` that treats `x` as discrete values if true.
    name: (Optional) A name for this operation.

  Returns:
    counts: The histogram, as counts per bin.
    boundaries: A `Tensor` used to build the histogram representing boundaries.
  """

  with tf.compat.v1.name_scope(name, 'histogram'):
    x = tf.reshape(tf_utils.get_values(x), [-1])
    if categorical:
      x_dtype = x.dtype
      x = x if x_dtype == tf.string else tf.strings.as_string(x)
      elements, counts = count_per_key(x)
      if x_dtype != elements.dtype:
        elements = tf.strings.to_number(elements, tf.int64)
      return counts, elements

    if boundaries is None:
      boundaries = tf.range(11, dtype=tf.float32) / 10.0
    elif isinstance(boundaries, int) or (isinstance(boundaries, tf.Tensor) and
                                         boundaries.get_shape().ndims == 0):
      min_value, max_value = _min_and_max(x, True)
      boundaries = tf.linspace(
          tf.cast(min_value, tf.float32), tf.cast(max_value, tf.float32),
          tf.cast(boundaries, tf.int64))

    # Shift the boundaries slightly to account for floating point errors,
    # and due to the fact that the rightmost boundary is essentially ignored.
    boundaries = tf.expand_dims(tf.cast(boundaries, tf.float32), 0) - 0.0001

    bucket_indices = tf_utils.assign_buckets(
        tf.cast(x, tf.float32), remove_leftmost_boundary(boundaries))
    bucket_vocab, counts = count_per_key(tf.strings.as_string(bucket_indices))
    counts = tf_utils.reorder_histogram(bucket_vocab, counts,
                                        tf.size(boundaries) - 1)
    return counts, boundaries


@common.log_api_use(common.ANALYZER_COLLECTION)
def size(x: common_types.TensorType,
         reduce_instance_dims: bool = True,
         name: Optional[str] = None) -> tf.Tensor:
  """Computes the total size of instances in a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
      to arrive at a single scalar output. If False, only collapses the batch
      dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` of type int64.
  """
  with tf.compat.v1.name_scope(name, 'size'):
    # Note: Calling `sum` defined in this module, not the builtin.
    if isinstance(x, tf.SparseTensor):
      ones_like_x = tf.SparseTensor(
          indices=x.indices,
          values=tf.ones_like(x.values, tf.int64),
          dense_shape=x.dense_shape)
    else:
      ones_like_x = tf.ones_like(x, dtype=tf.int64)
    return sum(ones_like_x, reduce_instance_dims)


@common.log_api_use(common.ANALYZER_COLLECTION)
def count_per_key(key: common_types.TensorType,
                  key_vocabulary_filename: Optional[str] = None,
                  name: Optional[str] = None):
  """Computes the count of each element of a `Tensor`.

  Args:
    key: A Tensor or `CompositeTensor` of dtype tf.string or tf.int.
    key_vocabulary_filename: (Optional) The file name for the key-output mapping
      file. If None and key are provided, this combiner assumes the keys fit in
      memory and will not store the result in a file. If empty string, a file
      name will be chosen based on the current scope. If not an empty string,
      should be unique within a given preprocessing function.
    name: (Optional) A name for this operation.

  Returns:
    Either:
    (A) Two `Tensor`s: one the key vocab with dtype of input;
        the other the count for each key, dtype tf.int64. (if
        key_vocabulary_filename is None).
    (B) The filename where the key-value mapping is stored (if
        key_vocabulary_filename is not None).

  Raises:
    TypeError: If the type of `x` is not supported.
  """

  with tf.compat.v1.name_scope(name, 'count_per_key'):
    key_dtype = key.dtype
    batch_keys, batch_counts = tf_utils.reduce_batch_count_per_key(key)

    output_dtype, sum_fn = _sum_combine_fn_and_dtype(tf.int64)
    numeric_combine_result = _numeric_combine(
        inputs=[batch_counts],
        fn=sum_fn,
        default_accumulator_value=0,
        reduce_instance_dims=True,
        output_dtypes=[output_dtype],
        key=batch_keys,
        key_vocabulary_filename=key_vocabulary_filename)

    if key_vocabulary_filename is not None:
      return numeric_combine_result
    keys, counts = numeric_combine_result
    if key_dtype is not tf.string:
      keys = tf.strings.to_number(keys, key_dtype)
    return keys, counts


@common.log_api_use(common.ANALYZER_COLLECTION)
def mean(x: common_types.TensorType,
         reduce_instance_dims: bool = True,
         name: Optional[str] = None,
         output_dtype: Optional[tf.DType] = None) -> tf.Tensor:
  """Computes the mean of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor` or `CompositeTensor`. Its type must be floating point
        (float{16|32|64}), or integral ([u]int{8|16|32|64}).
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` containing the mean. If `x` is floating point, the mean will have
    the same type as `x`. If `x` is integral, the output is cast to float32.
    NaNs and infinite input values are ignored.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'mean'):
    return _mean_and_var(x, reduce_instance_dims, output_dtype)[0]


@common.log_api_use(common.ANALYZER_COLLECTION)
def var(x: common_types.TensorType,
        reduce_instance_dims: bool = True,
        name: Optional[str] = None,
        output_dtype: Optional[tf.DType] = None) -> tf.Tensor:
  """Computes the variance of the values of a `Tensor` over the whole dataset.

  Uses the biased variance (0 delta degrees of freedom), as given by
  (x - mean(x))**2 / length(x).

  Args:
    x: `Tensor` or `CompositeTensor`. Its type must be floating point
        (float{16|32|64}), or integral ([u]int{8|16|32|64}).
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` containing the variance. If `x` is floating point, the variance
    will have the same type as `x`. If `x` is integral, the output is cast to
    float32. NaNs and infinite input values are ignored.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'var'):
    return _mean_and_var(x, reduce_instance_dims, output_dtype)[1]


def _mean_and_var(x: common_types.TensorType,
                  reduce_instance_dims: bool = True,
                  output_dtype: Optional[tf.DType] = None):
  """More efficient combined `mean` and `var`.  See `var`."""
  if output_dtype is None:
    output_dtype = _FLOAT_OUTPUT_DTYPE_MAP.get(x.dtype)
    if output_dtype is None:
      raise TypeError('Tensor type %r is not supported' % x.dtype)
  if not reduce_instance_dims and isinstance(x, tf.RaggedTensor):
    raise NotImplementedError(
        'Elementwise mean_and_var does not support RaggedTensors.')

  with tf.compat.v1.name_scope('mean_and_var'):

    x = tf.cast(x, output_dtype)

    x_count, x_mean, x_variance = (
        tf_utils.reduce_batch_count_mean_and_var(x, reduce_instance_dims))

    combine_inputs = _WeightedMeanAndVarAccumulator(
        count=x_count,
        mean=x_mean,
        variance=x_variance,
        weight=tf.zeros([], tf.float32))

    output_shape = ()
    if not reduce_instance_dims:
      # We need to use tf.expand_dims to artificially add a batch dimension.
      output_shape = _get_output_shape_from_input(
          tf.expand_dims(x_count, axis=0))

    x_mean, x_var = _apply_cacheable_combiner(
        WeightedMeanAndVarCombiner(output_dtype.as_numpy_dtype, output_shape),
        *combine_inputs)

  return x_mean, x_var


@common.log_api_use(common.ANALYZER_COLLECTION)
def tukey_location(x: common_types.TensorType,
                   reduce_instance_dims: Optional[bool] = True,
                   output_dtype: Optional[tf.DType] = None,
                   name: Optional[str] = None) -> tf.Tensor:
  """Computes the location of the values of a `Tensor` over the whole dataset.

  This computes the location of x, assuming a Tukey HH distribution, i.e.
  (x - tukey_location) / tukey_scale is a Tukey HH distribution with parameters
  tukey_h_params. See the following publication for the definition of the Tukey
  HH distribution:

  Todd C. Headrick, and Mohan D. Pant. "Characterizing Tukey h and
  hh-Distributions through L-Moments and the L-Correlation," ISRN Applied
  Mathematics, vol. 2012, 2012. doi:10.5402/2012/980153

  Args:
    x: A `Tensor` or `CompositeTensor`. Its type must be floating point
        (float{16|32|64}), or integral ([u]int{8|16|32|64}).
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    output_dtype: (Optional) If not None, casts the output tensor to this type.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the location. If `x` is floating point, the location
    will have the same type as `x`. If `x` is integral, the output is cast to
    float32.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'tukey_location'):
    return _tukey_parameters(x, reduce_instance_dims, output_dtype)[0]


@common.log_api_use(common.ANALYZER_COLLECTION)
def tukey_scale(x: common_types.TensorType,
                reduce_instance_dims: Optional[bool] = True,
                output_dtype: Optional[tf.DType] = None,
                name: Optional[str] = None) -> tf.Tensor:
  """Computes the scale of the values of a `Tensor` over the whole dataset.

  This computes the scale of x, assuming a Tukey HH distribution, i.e.
  (x - tukey_location) / tukey_scale is a Tukey HH distribution with parameters
  tukey_h_params. See the following publication for the definition of the Tukey
  HH distribution:

  Todd C. Headrick, and Mohan D. Pant. "Characterizing Tukey h and
  hh-Distributions through L-Moments and the L-Correlation," ISRN Applied
  Mathematics, vol. 2012, 2012. doi:10.5402/2012/980153


  Args:
    x: A `Tensor` or `CompositeTensor`. Its type must be floating point
        (float{16|32|64}), or integral ([u]int{8|16|32|64}).
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    output_dtype: (Optional) If not None, casts the output tensor to this type.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the scale. If `x` is floating point, the location
    will have the same type as `x`. If `x` is integral, the output is cast to
    float32.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'tukey_scale'):
    return _tukey_parameters(x, reduce_instance_dims, output_dtype)[1]


@common.log_api_use(common.ANALYZER_COLLECTION)
def tukey_h_params(x: common_types.TensorType,
                   reduce_instance_dims: bool = True,
                   output_dtype: Optional[tf.DType] = None,
                   name: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes the h parameters of the values of a `Tensor` over the dataset.

  This computes the parameters (hl, hr) of the samples, assuming a Tukey HH
  distribution, i.e. (x - tukey_location) / tukey_scale is a Tukey HH
  distribution with parameters hl (left parameter) and hr (right parameter).
  See the following publication for the definition of the Tukey HH distribution:

  Todd C. Headrick, and Mohan D. Pant. "Characterizing Tukey h and
  hh-Distributions through L-Moments and the L-Correlation," ISRN Applied
  Mathematics, vol. 2012, 2012. doi:10.5402/2012/980153

  Args:
    x: A `Tensor` or `CompositeTensor`. Its type must be floating point
        (float{16|32|64}), or integral ([u]int{8|16|32|64}).
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    output_dtype: (Optional) If not None, casts the output tensor to this type.
    name: (Optional) A name for this operation.

  Returns:
    The tuple (hl, hr) containing two `Tensor` instances with the hl and hr
    parameters. If `x` is floating point, each parameter will have the same type
    as `x`. If `x` is integral, the output is cast to float32.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'tukey_h_params'):
    return _tukey_parameters(x, reduce_instance_dims, output_dtype)[2:]


def _tukey_parameters(
    x: common_types.TensorType,
    reduce_instance_dims: bool = True,
    output_dtype: Optional[tf.DType] = None
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Efficient computation of L-moments."""
  if output_dtype is None:
    output_dtype = _FLOAT_OUTPUT_DTYPE_MAP.get(x.dtype)
    if output_dtype is None:
      raise TypeError('Tensor type %r is not supported' % x.dtype)

  with tf.compat.v1.name_scope('tukey_parameters'):

    x = tf.cast(x, output_dtype)

    (count_l1, l1, count_l2, l2, count_l3, l3, count_l4, l4) = (
        tf_utils.reduce_batch_count_l_moments(x, reduce_instance_dims))

    combine_inputs = _LMomentsAccumulator(
        count_l1=count_l1,
        count_l2=count_l2,
        count_l3=count_l3,
        count_l4=count_l4,
        l1=l1,
        l2=l2,
        l3=l3,
        l4=l4)

    output_shape = ()
    if not reduce_instance_dims:
      output_shape = _get_output_shape_from_input(x)

    x_loc, x_scale, hl_param, hr_param = _apply_cacheable_combiner(
        _LMomentsCombiner(output_dtype.as_numpy_dtype, output_shape),
        *combine_inputs)

  return x_loc, x_scale, hl_param, hr_param


def _mean_and_var_per_key(
    x: common_types.TensorType,
    key: common_types.TensorType,
    reduce_instance_dims: bool = True,
    output_dtype: Optional[tf.DType] = None,
    key_vocabulary_filename: Optional[str] = None
) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor,
           common_types.Asset]:
  """`mean_and_var` by group, specified by key.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    key: A Tensor or `CompositeTensor` of dtype tf.string.  If `x` is
      a `CompositeTensor`, `key` must exactly match `x` in everything except
      values.
    reduce_instance_dims: (Optional) By default collapses the batch and instance
        dimensions to arrive at a single scalar output. The False case is not
        currently supported for _mean_and_var_per_key.
    output_dtype: (Optional) Desired output dtype, otherwise inferred.
    key_vocabulary_filename: (Optional) The file name for the key-output mapping
      file. If None and key are provided, this combiner assumes the keys fit in
      memory and will not store the result in a file. If empty string, a file
      name will be chosen based on the current scope. If not an empty string,
      should be unique within a given preprocessing function.

  Returns:
    Either:
    (A) Three `Tensor`s. The first is the key vocab of type tf.string, and the
        second two have same type as `x` (if key_vocabulary_filename is None).
    (B) The filename where the key-value mapping is stored (if
        key_vocabulary_filename is not None).
    NaNs and infinite input values are ignored.
  """
  if output_dtype is None:
    output_dtype = _FLOAT_OUTPUT_DTYPE_MAP.get(x.dtype)
    if output_dtype is None:
      raise TypeError('Tensor type %r is not supported' % x.dtype)

  if key is None:
    raise ValueError('A non-None key is required for _mean_and_var_per_key')

  if not reduce_instance_dims:
    raise NotImplementedError('Per-key elementwise reduction not supported')

  with tf.compat.v1.name_scope('mean_and_var_per_key'):
    x = tf.cast(x, output_dtype)

    key_vocab, key_counts, key_means, key_variances = (
        tf_utils.reduce_batch_count_mean_and_var_per_key(
            x, key, reduce_instance_dims=reduce_instance_dims))
    output_shape = ()

    combine_inputs = _WeightedMeanAndVarAccumulator(
        count=key_counts,
        mean=key_means,
        variance=key_variances,
        weight=tf.zeros_like(key_means, tf.float32))

    combiner = WeightedMeanAndVarCombiner(output_dtype.as_numpy_dtype,
                                          output_shape)

  if key_vocabulary_filename is not None:
    key_vocabulary_filename = _maybe_get_per_key_vocab_filename(
        key_vocabulary_filename)
    return _apply_cacheable_combiner_per_key_large(
        combiner, key_vocabulary_filename, key_vocab, *combine_inputs)

  key, key_mean, key_var = _apply_cacheable_combiner_per_key(
      combiner, key_vocab, *combine_inputs)

  return key, key_mean, key_var


class _WeightedMeanAndVarAccumulator(
    tfx_namedtuple.namedtuple('WeightedMeanAndVarAccumulator',
                              ['count', 'mean', 'variance', 'weight'])):
  """Container for WeightedMeanAndVarCombiner intermediate values."""

  @classmethod
  def make_nan_to_num(cls,
                      counts,
                      means,
                      variances,
                      weights,
                      compute_variance=False,
                      compute_weighted=True):
    """Util function to replace NaN with 0 and inf with large finite numbers."""
    if compute_variance:
      variances = np.nan_to_num(variances, copy=True)

    if compute_weighted:
      weights = np.nan_to_num(weights, copy=True)

    return cls(
        np.array(counts), np.nan_to_num(means, copy=True), variances, weights)


class WeightedMeanAndVarCombiner(analyzer_nodes.Combiner):
  """Combines a PCollection of accumulators to compute mean and variance."""

  accumulator_class = _WeightedMeanAndVarAccumulator

  def __init__(self,
               output_numpy_dtype,
               output_shape: Optional[Collection[Optional[int]]] = None,
               compute_variance: bool = True,
               compute_weighted: bool = False):
    """Init method for WeightedMeanAndVarCombiner.

    Args:
      output_numpy_dtype: A numpy dtype that the outputs are cast to.
      output_shape: The shape of the resulting Tensors.
      compute_variance: A bool indicating whether or not a variance should be
        calculated and returned.
      compute_weighted: A bool indicating whether or not weights are provided
        and all calculations should be weighted.
    """
    self._output_numpy_dtype = output_numpy_dtype
    self._output_shape = output_shape
    self._compute_variance = compute_variance
    self._compute_weighted = compute_weighted

    if self._compute_variance and self._compute_weighted:
      raise ValueError(
          'WeightedMeanAndVarCombiner does not yet support weighted variance')
    if self._output_shape is None:
      raise ValueError('An output_shape must be provided.')

  def create_accumulator(self) -> _WeightedMeanAndVarAccumulator:
    """Create an accumulator with all zero entries."""
    # TODO(b/131325061): Determine whether counts/weights should always be
    # scalars or if we want to continue supporting multi-dimensional arrays.
    initial_count, initial_weight = np.array(0), np.array(0.)
    # If we know the exact shape, initialize accumulator values with zeros of
    # the exact shape. For unknown dimensions, initialize with a 1D 0 array.
    output_shape = [dim if dim is not None else 0 for dim in self._output_shape]
    initial_mean, initial_var = np.zeros(output_shape), np.zeros(output_shape)
    return _WeightedMeanAndVarAccumulator(initial_count, initial_mean,
                                          initial_var, initial_weight)

  def add_input(
      self, accumulator: _WeightedMeanAndVarAccumulator,
      batch_values: _WeightedMeanAndVarAccumulator
  ) -> _WeightedMeanAndVarAccumulator:
    """Composes an accumulator from batch_values and calls merge_accumulators.

    Args:
      accumulator: The `_WeightedMeanAndVarAccumulator` computed so far.
      batch_values: A `_WeightedMeanAndVarAccumulator` for the current batch.

    Returns:
      A `_WeightedMeanAndVarAccumulator` which is accumulator and batch_values
        combined.
    """
    new_accumulator = _WeightedMeanAndVarAccumulator(*batch_values)
    return self._combine_mean_and_var_accumulators(accumulator, new_accumulator)

  def merge_accumulators(
      self, accumulators: List[_WeightedMeanAndVarAccumulator]
  ) -> _WeightedMeanAndVarAccumulator:
    """Merges several `_WeightedMeanAndVarAccumulator`s to a single accumulator.

    Args:
      accumulators: A list of `_WeightedMeanAndVarAccumulator`s.

    Returns:
      The sole merged `_WeightedMeanAndVarAccumulator`.
    """
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result = self._combine_mean_and_var_accumulators(result, accumulator)
    return result

  def extract_output(
      self, accumulator: _WeightedMeanAndVarAccumulator
  ) -> Union[Tuple[float, float], _WeightedMeanAndVarAccumulator]:
    """Converts an accumulator into the output accumulator or (mean, var) tuple.

    Args:
      accumulator: the final `_WeightedMeanAndVarAccumulator` value.

    Returns:
      A _WeightedMeanAndVarAccumulator or a 2-tuple composed of (mean, var).
    """

    if self._compute_variance and not self._compute_weighted:
      return (self._output_numpy_dtype(accumulator.mean),
              self._output_numpy_dtype(accumulator.variance))
    else:
      return _WeightedMeanAndVarAccumulator(
          np.int64(accumulator.count),
          self._output_numpy_dtype(accumulator.mean),
          self._output_numpy_dtype(accumulator.variance),
          self._output_numpy_dtype(accumulator.weight))

  def output_tensor_infos(self) -> List[analyzer_nodes.TensorInfo]:
    # The output is (mean, var).
    if self._compute_variance and not self._compute_weighted:
      return [
          analyzer_nodes.TensorInfo(
              tf.as_dtype(self._output_numpy_dtype), self._output_shape, None)
      ] * 2
    else:
      return [
          analyzer_nodes.TensorInfo(
              tf.as_dtype(np.int64), self._output_shape, None),
          analyzer_nodes.TensorInfo(
              tf.as_dtype(self._output_numpy_dtype), self._output_shape, None),
          analyzer_nodes.TensorInfo(
              tf.as_dtype(self._output_numpy_dtype), self._output_shape, None),
          analyzer_nodes.TensorInfo(
              tf.as_dtype(self._output_numpy_dtype), self._output_shape, None)
      ]

  def _combine_mean_and_var_accumulators(
      self, a: _WeightedMeanAndVarAccumulator,
      b: _WeightedMeanAndVarAccumulator) -> _WeightedMeanAndVarAccumulator:
    """Combines two mean and var accumulators.

    Args:
      a: A _WeightedMeanAndVarAccumulator.
      b: A _WeightedMeanAndVarAccumulator.

    Returns:
      A _WeightedMeanAndVarAccumulator computed as the combination of a and b.
    """
    # NaNs get preserved through division by a.count + b.count.
    a = _WeightedMeanAndVarAccumulator.make_nan_to_num(
        *a,
        compute_variance=self._compute_variance,
        compute_weighted=self._compute_weighted)
    b = _WeightedMeanAndVarAccumulator.make_nan_to_num(
        *b,
        compute_variance=self._compute_variance,
        compute_weighted=self._compute_weighted)

    # a.count >= b.count following this logic.
    if np.sum(a.count) < np.sum(b.count):
      a, b = b, a

    if np.sum(a.count) == 0:
      return b

    a_count, b_count = _pad_arrays_to_match(a.count, b.count)
    a_mean, b_mean = _pad_arrays_to_match(a.mean, b.mean)
    if self._compute_variance:
      a_variance, b_variance = _pad_arrays_to_match(a.variance, b.variance)
    if self._compute_weighted:
      a_weight, b_weight = _pad_arrays_to_match(a.weight, b.weight)

    combined_total = a_count + b_count

    # Mean and variance update formulas which are more numerically stable when
    # a and b vary in magnitude.
    if self._compute_weighted:
      combined_weights_mean = (
          a_weight + (b_count / combined_total) * (b_weight - a_weight))
      combined_mean = a_mean + (b_count * b_weight /
                                (combined_total * combined_weights_mean)) * (
                                    b_mean - a_mean)
    else:
      combined_weights_mean = np.ones(shape=combined_total.shape)
      combined_mean = a_mean + (b_count / combined_total * (b_mean - a_mean))

    if self._compute_variance:
      # TODO(zoyahav): Add an option for weighted variance if needed.
      assert not self._compute_weighted
      combined_variance = (
          a_variance + (b_count / combined_total) * (b_variance - a_variance +
                                                     ((b_mean - combined_mean) *
                                                      (b_mean - a_mean))))
    else:
      combined_variance = np.zeros(combined_mean.shape)

    return _WeightedMeanAndVarAccumulator(combined_total, combined_mean,
                                          combined_variance,
                                          combined_weights_mean)


# TODO(b/165020671): Optimize padding to save up to 15% computing resource.
def _pad_arrays_to_match(a, b):
  """Pad the ndarray values to match dimensions as needed.

  If the dimensions of the ndarrays values differ, we pad the smaller of the
  two arrays with zeros to be the same shape as the larger. In other words,
  the missing accumulator indices are assumed to be zero, and combining
  a = [1, 2, 3] with b = [1, 2] is equivalent t combining with b = [1, 2, 0].

  Args:
    a: NDarray to be matched in shaped with b
    b: NDarray to be matched in shaped with a

  Returns:
    a: a padded to same dimensions as b
    b: b padded to same dimensions as a
  """
  if a.shape == b.shape:
    return a, b
  padding_a, padding_b = [], []
  for a_dim, b_dim in zip(a.shape, b.shape):
    a_pad = b_pad = (0, 0)
    delta = a_dim - b_dim
    if delta > 0:
      b_pad = (0, abs(delta))
    elif delta < 0:
      a_pad = (0, abs(delta))
    padding_a.append(a_pad)
    padding_b.append(b_pad)
  if padding_a:
    a = np.pad(a, padding_a, mode='constant')
  if padding_b:
    b = np.pad(b, padding_b, mode='constant')
  return a, b


class _LMomentsAccumulator(
    tfx_namedtuple.namedtuple('LMomentsAccumulator', [
        'count_l1', 'count_l2', 'count_l3', 'count_l4', 'l1', 'l2', 'l3', 'l4'
    ])):
  """Container for _LMomentsCombiner intermediate values."""

  @classmethod
  def make_nan_to_num(cls, count_l1, count_l2, count_l3, count_l4,
                      l1, l2, l3, l4):
    return cls(
        np.array(count_l1), np.array(count_l2), np.array(count_l3),
        np.array(count_l4), np.nan_to_num(l1), np.nan_to_num(l2),
        np.nan_to_num(l3), np.nan_to_num(l4))

  def __reduce__(self):
    return self.__class__, tuple(self)


class _LMomentsCombiner(analyzer_nodes.Combiner):
  """Combines a PCollection of accumulators to compute L-moments."""

  accumulator_class = _LMomentsAccumulator

  def __init__(self, output_numpy_dtype, output_shape):
    """Init method for _LMomentsCombiner.

    Args:
      output_numpy_dtype: A numpy dtype that the outputs are cast to.
      output_shape: The shape of the resulting Tensors.
    """
    self._output_numpy_dtype = output_numpy_dtype
    self._output_shape = output_shape

  def create_accumulator(self):
    """Create an accumulator with all zero entries."""

    # If we know the exact shape, initialize accumulator values with zeros of
    # the exact shape. For unknown dimensions, initialize with a 1D 0 array
    # (this accumulator will be discarded by _combine_accumulators).
    output_shape = () if None in self._output_shape else self._output_shape
    initial_moment = np.zeros(output_shape, dtype=self._output_numpy_dtype)
    initial_count = np.zeros(output_shape, dtype=self._output_numpy_dtype)
    return _LMomentsAccumulator(
        initial_count, initial_count, initial_count, initial_count,
        initial_moment, initial_moment, initial_moment, initial_moment)

  def add_input(self, accumulator, batch_values):
    """Composes an accumulator from batch_values and calls merge_accumulators.

    Args:
      accumulator: The `_LMomentsAccumulator` computed so far.
      batch_values: A `_LMomentsAccumulator` for the current batch.

    Returns:
      A `_LMomentsAccumulator` which is accumulator and batch_values combined.
    """
    new_accumulator = _LMomentsAccumulator(*batch_values)
    return self._combine_accumulators(accumulator, new_accumulator)

  def merge_accumulators(self, accumulators):
    """Merges several `_LMomentsAccumulator`s to a single accumulator.

    Args:
      accumulators: A list of `_LMomentsAccumulator`s.

    Returns:
      The sole merged `_LMomentsAccumulator`.
    """
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result = self._combine_accumulators(result, accumulator)
    return result

  def extract_output(self, accumulator):
    """Converts an accumulator into the output (loc, scale, hl, hr) tuple.

    Estimates the parameters of a Tukey HH distribution, given estimates of the
    first four L-moments. The parameters are: location, scale, hl, and hr. If
    x is the input sample, then (x - location) / scale is distributed according
    to the Tukey HH distribution with parameters hl (left parameter) and hr
    (right parameter).

    Args:
      accumulator: the final `_LMomentsAccumulator` value.

    Returns:
      A 4-tuple composed of (location, scale, hl, hr).
    """

    # To compute kurtosis, we need positive scale and at least one quadruplet.
    # If this is not the case, L-kewness and L-kurtosis are set to zero, which
    # gives hl=0, hr=0 and samples are treated as in the Gaussian case.

    valid_scale = accumulator.l2 > 0.0
    valid_kurtosis = np.logical_and(valid_scale, accumulator.count_l4 > 0.0)

    l_skewness = np.true_divide(accumulator.l3, accumulator.l2,
                                where=valid_kurtosis,
                                out=np.zeros_like(accumulator.l3))

    l_kurtosis = np.true_divide(accumulator.l4, accumulator.l2,
                                where=valid_kurtosis,
                                out=np.zeros_like(accumulator.l4))
    l_skewness_and_kurtosis = np.stack((l_skewness, l_kurtosis), axis=0)
    h_params = np.apply_along_axis(
        gaussianization.compute_tukey_hh_params, 0, l_skewness_and_kurtosis)
    hh_l_mean, hh_l_scale = gaussianization.tukey_hh_l_mean_and_scale(h_params)

    scale = np.true_divide(accumulator.l2, hh_l_scale,
                           where=valid_scale, out=np.ones_like(accumulator.l2))
    loc = accumulator.l1 - scale * hh_l_mean
    hl = h_params[0, ...]
    hr = h_params[1, ...]
    return [self._output_numpy_dtype(x) for x in [loc, scale, hl, hr]]

  def output_tensor_infos(self):
    # The output is (loc, scale, hl, hr).
    return [
        analyzer_nodes.TensorInfo(
            tf.as_dtype(self._output_numpy_dtype), self._output_shape, None)
    ] * 4

  @property
  def accumulator_coder(self):
    # TODO(b/170510451): Re-enable caching for this Combiner.
    return None

  def _combine_accumulators(self, a, b):
    """Combines two accumulators.

    Args:
      a: A _LMomentsAccumulator.
      b: A _LMomentsAccumulator.

    Returns:
      A _LMomentsAccumulator computed as the combination of a and b.
    """
    # NaNs get preserved through division by a.count + b.count.
    a = _LMomentsAccumulator.make_nan_to_num(*a)
    b = _LMomentsAccumulator.make_nan_to_num(*b)

    # If one accumulator is empty return the other.
    if np.sum(a.count_l1) < np.sum(b.count_l1):
      a, b = b, a
    if np.sum(b.count_l1) == 0:
      return a

    a_count_l1, b_count_l1 = _pad_arrays_to_match(a.count_l1, b.count_l1)
    a_l1, b_l1 = _pad_arrays_to_match(a.l1, b.l1)
    a_count_l2, b_count_l2 = _pad_arrays_to_match(a.count_l2, b.count_l2)
    a_l2, b_l2 = _pad_arrays_to_match(a.l2, b.l2)
    a_count_l3, b_count_l3 = _pad_arrays_to_match(a.count_l3, b.count_l3)
    a_l3, b_l3 = _pad_arrays_to_match(a.l3, b.l3)
    a_count_l4, b_count_l4 = _pad_arrays_to_match(a.count_l4, b.count_l4)
    a_l4, b_l4 = _pad_arrays_to_match(a.l4, b.l4)

    combined_count_l1 = a_count_l1 + b_count_l1
    combined_count_l2 = a_count_l2 + b_count_l2
    combined_count_l3 = a_count_l3 + b_count_l3
    combined_count_l4 = a_count_l4 + b_count_l4

    combined_l1 = (a_l1 + np.true_divide(
        b_count_l1, combined_count_l1, where=combined_count_l1 > 0,
        out=np.zeros_like(a_l1)) * (b_l1 - a_l1))
    combined_l2 = (a_l2 + np.true_divide(
        b_count_l2, combined_count_l2, where=combined_count_l2 > 0,
        out=np.zeros_like(a_l2)) * (b_l2 - a_l2))
    combined_l3 = (a_l3 + np.true_divide(
        b_count_l3, combined_count_l3, where=combined_count_l3 > 0,
        out=np.zeros_like(a_l3)) * (b_l3 - a_l3))
    combined_l4 = (a_l4 + np.true_divide(
        b_count_l4, combined_count_l4, where=combined_count_l4 > 0,
        out=np.zeros_like(a_l4)) * (b_l4 - a_l4))

    return _LMomentsAccumulator(
        combined_count_l1, combined_count_l2, combined_count_l3,
        combined_count_l4, combined_l1, combined_l2, combined_l3, combined_l4)


def sanitized_vocab_filename(filename=None, prefix=None):
  """Generates a sanitized filename either from the given filename or the scope.

  If filename is specified, provide a sanitized version of the given filename.
  Otherwise generate a filename from the current scope.  Note that it is the
  callers responsibility to ensure that filenames are unique across calls within
  a given preprocessing function.

  Args:
    filename: A filename with non-alpha characters replaced with underscores and
      spaces to hyphens.
    prefix: Prefix to use for the name of the vocab file, if filename
      is not given.

  Returns:
    A valid filename.

  Raises:
    ValueError: If neither filename and prefix are specified, or if both
      are specified.
  """
  if filename is None and prefix is None:
    raise ValueError('Both filename and prefix cannot be None.')

  if filename is not None and prefix is not None:
    raise ValueError('Only one of filename or prefix can be specified.')

  if filename is None:
    filename = prefix + tf.compat.v1.get_default_graph().get_name_scope()
  # Replace non-alpha characters (excluding whitespaces) with '_'.
  filename = re.sub(r'[^\w\s-]', '_', filename).strip()
  # Replace whitespaces with '-'.
  return re.sub(r'[-\s]+', '-', filename)


def _get_vocab_filename(vocab_filename, store_frequency):
  """Returns a sanitized vocabulary filename with appropriate prefix applied.

  Args:
    vocab_filename: The file name for the vocabulary file. If none, the
      "vocabulary" scope name in the context of this graph will be used as the
      file name.
    store_frequency: A bool that is true when the vocabulary for which this
      generates a filename stores term frequency. False otherwise.

  Returns:
    A valid filename.
  """
  if vocab_filename is not None:
    prefix = None
  elif store_frequency:
    prefix = VOCAB_FREQUENCY_FILENAME_PREFIX
  else:
    prefix = VOCAB_FILENAME_PREFIX

  # Make the file name path safe.
  return sanitized_vocab_filename(vocab_filename, prefix=prefix)


def _maybe_get_per_key_vocab_filename(key_vocabulary_filename):
  if key_vocabulary_filename == '':  # pylint: disable=g-explicit-bool-comparison
    key_vocabulary_filename = _get_vocab_filename(vocab_filename=None,
                                                  store_frequency=False)
  return key_vocabulary_filename


# TODO(b/116308354): frequency_threshold is misleading since this threshold can
# be applied to mutual information rather than frequency.
def _get_top_k_and_frequency_threshold(top_k, frequency_threshold):
  """Validate `top_k` and `frequency_threshold` values and convert to number."""
  if top_k is not None:
    top_k = int(top_k)
    if top_k <= 0:
      raise ValueError('top_k must be positive, but got: %r' % top_k)

  if frequency_threshold is not None:
    frequency_threshold = float(frequency_threshold)
    if frequency_threshold < 0:
      raise ValueError(
          'frequency_threshold must be non-negative, but got: %r' %
          frequency_threshold)
    elif frequency_threshold <= 1:
      # Note: this warning is misleading in the context where tokens are ranked
      # based on mutual information rather than frequency.
      tf.compat.v1.logging.warn(
          'frequency_threshold %d <= 1 is a no-op, use None instead.',
          frequency_threshold)
  return top_k, frequency_threshold


class _VocabOrderingType:
  """Class for all vocab ordering types."""
  # Orders vocabulary based on the simple frequency of the token
  FREQUENCY = 1
  # Orders vocabulary based on the weighted frequency of the token
  WEIGHTED_FREQUENCY = 2
  # Orders vocabulary based on the weighted mutual
  # information of token with the label
  WEIGHTED_MUTUAL_INFORMATION = 3
  # Experimental
  WEIGHTED_LABELS = 4
  # Orders vocabulary based on the mutual information
  # of token with the label and without weight.
  MUTUAL_INFORMATION = 5


def register_vocab(sanitized_filename: str,
                   vocabulary_size: Optional[tf.Tensor] = None,
                   vocabulary_key: Optional[str] = None,
                   file_format: common_types
                   .VocabularyFileFormatType = DEFAULT_VOCABULARY_FILE_FORMAT):
  """Registers the specificed vocabulary within the asset map.

  Args:
    sanitized_filename: The santized filename of the vocabulary.
    vocabulary_size: The size of the vocabulary.
    vocabulary_key: The key of the vocabulary to use.
    file_format: The format of the vocabulary file (text or tfrecord_gzip).
  """
  if vocabulary_key is None:
    vocabulary_key = sanitized_filename
  filename = ('{}.tfrecord.gz'.format(sanitized_filename)
              if file_format == 'tfrecord_gzip' else sanitized_filename)
  annotators.annotate_asset(vocabulary_key, filename)
  if vocabulary_size is not None:
    annotators.annotate_vocab_size(vocabulary_key, vocabulary_size)


def get_empy_vocabulary_dummy_value(
    dtype: Union[tf.dtypes.DType, str]) -> Tuple[int, bytes]:
  """Returns a vocabulary entry to use in case of an empty vocabulary."""
  # TODO(b/62272023) remove this workaround if/when fixed on tensorflow.
  # If the vocabulary is empty add a dummy value with count one so
  # the tensorflow index operations don't fail to initialize with empty
  # tensors downstream.
  dummy_value = (b'49d0cd50-04bb-48c0-bc6f-5b575dce351a'
                 if tf.dtypes.as_dtype(dtype) == tf.string else b'-1')
  return (1, dummy_value)


# TODO(KesterTong): Once multiple outputs are supported, return indices too.
# TODO(b/117796748): Add coverage key feature input as alternative to `key_fn`.
# TODO(tensorflow/community) the experimental fingerprint_shuffle argument is a
# workaround for the inability to appropriately rebalance sharded variables on
# TF 1.0. The following TF 2.0 proposal should address this issue in the future
# https://github.com/tensorflow/community/blob/master/rfcs/20190116-embedding-partitioned-variable.md#goals
@common.log_api_use(common.ANALYZER_COLLECTION)
def vocabulary(
    x: common_types.TensorType,
    top_k: Optional[int] = None,
    frequency_threshold: Optional[int] = None,
    vocab_filename: Optional[str] = None,
    store_frequency: Optional[bool] = False,
    weights: Optional[tf.Tensor] = None,
    labels: Optional[tf.Tensor] = None,
    use_adjusted_mutual_info: bool = False,
    min_diff_from_avg: Optional[int] = None,
    coverage_top_k: Optional[int] = None,
    coverage_frequency_threshold: Optional[int] = None,
    key_fn: Optional[Callable[[Any], Any]] = None,
    fingerprint_shuffle: Optional[bool] = False,
    file_format: common_types
    .VocabularyFileFormatType = DEFAULT_VOCABULARY_FILE_FORMAT,
    name: Optional[str] = None) -> common_types.TemporaryAnalyzerOutputType:
  r"""Computes the unique values of a `Tensor` over the whole dataset.

  Computes The unique values taken by `x`, which can be a `Tensor` or
  `CompositeTensor` of any size.  The unique values will be aggregated over all
  dimensions of `x` and all instances.

  In case `file_format` is 'text' and one of the tokens contains the '\n' or
  '\r' characters or is empty it will be discarded.

  If an integer `Tensor` is provided, its semantic type should be categorical
  not a continuous/numeric, since computing a vocabulary over a continuous
  feature is not appropriate.

  The unique values are sorted by decreasing frequency and then reverse
  lexicographical order (e.g. [('a', 5), ('c', 3), ('b', 3)]). This is true even
  if `x` is numerical dtype (e.g. [('3', 5), ('2', 3), ('111', 3)]).

  For large datasets it is highly recommended to either set frequency_threshold
  or top_k to control the size of the output, and also the run time of this
  operation.

  When labels are provided, we filter the vocabulary based on the relationship
  between the token's presence in a record and the label for that record, using
  (possibly adjusted) Mutual Information. Note: If labels are provided, the x
  input must be a unique set of per record, as the semantics of the mutual
  information calculation depend on a multi-hot representation of the input.
  Having unique input tokens per row is advisable but not required for a
  frequency-based vocabulary.

  WARNING: The following is experimental and is still being actively worked on.

  Supply `key_fn` if you would like to generate a vocabulary with coverage over
  specific keys.

  A "coverage vocabulary" is the union of two vocabulary "arms". The "standard
  arm" of the vocabulary is equivalent to the one generated by the same function
  call with no coverage arguments. Adding coverage only appends additional
  entries to the end of the standard vocabulary.

  The "coverage arm" of the vocabulary is determined by taking the
  `coverage_top_k` most frequent unique terms per key. A term's key is obtained
  by applying `key_fn` to the term. Use `coverage_frequency_threshold` to lower
  bound the frequency of entries in the coverage arm of the vocabulary.

  Note this is currently implemented for the case where the key is contained
  within each vocabulary entry (b/117796748).

  Args:
    x: A categorical/discrete input `Tensor` or `CompositeTensor` with dtype
      tf.string or tf.int[8|16|32|64]. The inputs should generally be unique per
      row (i.e. a bag of words/ngrams representation).
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      absolute frequency is >= to the supplied threshold. If set to None, the
      full vocabulary is generated.  Absolute frequency means the number of
      occurrences of the element in the dataset, as opposed to the proportion of
      instances that contain that element.
    vocab_filename: The file name for the vocabulary file. If None, a file name
      will be chosen based on the current scope. If not None, should be unique
      within a given preprocessing function. NOTE To make your pipelines
      resilient to implementation details please set `vocab_filename` when you
      are using the vocab_filename on a downstream component.
    store_frequency: If True, frequency of the words is stored in the vocabulary
      file. In the case labels are provided, the mutual information is stored in
      the file instead. Each line in the file will be of the form
      'frequency word'. NOTE: if this is True then the computed vocabulary
      cannot be used with `tft.apply_vocabulary` directly, since frequencies are
      added to the beginning of each row of the vocabulary, which the mapper
      will not ignore.
    weights: (Optional) Weights `Tensor` for the vocabulary. It must have the
      same shape as x.
    labels: (Optional) Labels dense `Tensor` for the vocabulary. If provided,
      the vocabulary is calculated based on mutual information with the label,
      rather than frequency. The labels must have the same batch dimension as x.
      If x is sparse, labels should be a 1D tensor reflecting row-wise labels.
      If x is dense, labels can either be a 1D tensor of row-wise labels, or a
      dense tensor of the identical shape as x (i.e. element-wise labels).
      Labels should be a discrete integerized tensor (If the label is numeric,
      it should first be bucketized; If the label is a string, an integer
      vocabulary should first be applied). Note: `CompositeTensor` labels are
      not yet supported (b/134931826). WARNING: When labels are provided, the
      frequency_threshold argument functions as a mutual information
      threshold, which is a float. TODO(b/116308354): Fix confusing naming.
    use_adjusted_mutual_info: If true, and labels are provided, calculate
      vocabulary using adjusted rather than raw mutual information.
    min_diff_from_avg: MI (or AMI) of a feature x label will be adjusted to zero
      whenever the difference between count and the expected (average) count is
      lower than min_diff_from_average. This can be thought of as a regularizing
      parameter that pushes small MI/AMI values to zero. If None, a default
      parameter will be selected based on the size of the dataset (see
      calculate_recommended_min_diff_from_avg).
    coverage_top_k: (Optional), (Experimental) The minimum number of elements
      per key to be included in the vocabulary.
    coverage_frequency_threshold: (Optional), (Experimental) Limit the coverage
      arm of the vocabulary only to elements whose absolute frequency is >= this
      threshold for a given key.
    key_fn: (Optional), (Experimental) A fn that takes in a single entry of `x`
      and returns the corresponding key for coverage calculation. If this is
      `None`, no coverage arm is added to the vocabulary.
    fingerprint_shuffle: (Optional), (Experimental) Whether to sort the
      vocabularies by fingerprint instead of counts. This is useful for load
      balancing on the training parameter servers. Shuffle only happens while
      writing the files, so all the filters above (top_k, frequency_threshold,
      etc) will still take effect.
    file_format: (Optional) A str. The format of the resulting vocabulary file.
      Accepted formats are: 'tfrecord_gzip', 'text'. 'tfrecord_gzip' requires
        tensorflow>=2.4. The default value is 'text'.
    name: (Optional) A name for this operation.

  Returns:
    The path name for the vocabulary file containing the unique values of `x`.

  Raises:
    ValueError: If `top_k` or `frequency_threshold` is negative.
      If `coverage_top_k` or `coverage_frequency_threshold` is negative.
      If either `coverage_top_k` or `coverage_frequency_threshold` is specified
        and `key_fn` is not.
      If `key_fn` is specified and neither `coverage_top_k`, nor
  """
  top_k, frequency_threshold = _get_top_k_and_frequency_threshold(
      top_k, frequency_threshold)

  if (coverage_top_k or coverage_frequency_threshold) and not key_fn:
    raise ValueError('You must specify `key_fn` if you specify `coverage_top_k'
                     ' or `coverage_frequency_threshold` in `vocabulary`.')

  if key_fn and not (coverage_top_k or coverage_frequency_threshold):
    raise ValueError('You must specify `coverage_top_k`  or '
                     '`coverage_frequency_threshold` if you specify `key_fn` in'
                     ' `vocabulary`.')

  if file_format not in ALLOWED_VOCABULARY_FILE_FORMATS:
    raise ValueError(
        '"{}" is not an accepted file_format. It should be one of: {}'.format(
            file_format, ALLOWED_VOCABULARY_FILE_FORMATS))

  coverage_top_k, coverage_frequency_threshold = (
      _get_top_k_and_frequency_threshold(
          coverage_top_k, coverage_frequency_threshold))

  if x.dtype != tf.string and not x.dtype.is_integer:
    raise ValueError('expected tf.string or integer but got %r' % x.dtype)

  if labels is not None and not labels.dtype.is_integer:
    raise ValueError('expected integer labels but got %r' % labels.dtype)

  if (frequency_threshold is None and labels is None and key_fn is None and
      not fingerprint_shuffle and top_k is not None and
      top_k <= LARGE_VOCAB_TOP_K):
    logging.info('If the number of unique tokens is smaller than the provided '
                 'top_k or approximation error is acceptable, consider using '
                 'tft.experimental.approximate_vocabulary for a potentially '
                 'more efficient implementation.')

  with tf.compat.v1.name_scope(name, 'vocabulary'):
    vocabulary_key = vocab_filename
    vocab_filename = _get_vocab_filename(vocab_filename, store_frequency)
    informativeness_threshold = float('-inf')
    coverage_informativeness_threshold = float('-inf')
    if labels is not None:
      if weights is not None:
        vocab_ordering_type = _VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION
      else:
        vocab_ordering_type = _VocabOrderingType.MUTUAL_INFORMATION
      # Correct for the overloaded `frequency_threshold` API.
      if frequency_threshold is not None:
        informativeness_threshold = frequency_threshold
      frequency_threshold = 0.0
      if coverage_frequency_threshold is not None:
        coverage_informativeness_threshold = coverage_frequency_threshold
      coverage_frequency_threshold = 0.0
    elif weights is not None:
      vocab_ordering_type = _VocabOrderingType.WEIGHTED_FREQUENCY
    else:
      vocab_ordering_type = _VocabOrderingType.FREQUENCY
    analyzer_inputs = _get_vocabulary_analyzer_inputs(
        vocab_ordering_type=vocab_ordering_type,
        x=x,
        file_format=file_format,
        labels=labels,
        weights=weights)
    return _vocabulary_analyzer_nodes(
        analyzer_inputs=analyzer_inputs,
        input_dtype=x.dtype.name,
        vocab_ordering_type=vocab_ordering_type,
        vocab_filename=vocab_filename,
        top_k=top_k,
        frequency_threshold=frequency_threshold or 0,
        informativeness_threshold=informativeness_threshold,
        use_adjusted_mutual_info=use_adjusted_mutual_info,
        min_diff_from_avg=min_diff_from_avg,
        fingerprint_shuffle=fingerprint_shuffle,
        store_frequency=store_frequency,
        key_fn=key_fn,
        coverage_top_k=coverage_top_k,
        coverage_frequency_threshold=coverage_frequency_threshold or 0,
        coverage_informativeness_threshold=coverage_informativeness_threshold,
        file_format=file_format,
        vocabulary_key=vocabulary_key)


def _get_vocabulary_analyzer_inputs(
    vocab_ordering_type: int,
    x: common_types.TensorType,
    file_format: common_types.VocabularyFileFormatType,
    labels: Optional[tf.Tensor] = None,
    weights: Optional[tf.Tensor] = None):
  """Helper for constructing analyzer inputs from tensors.

  Args:
    vocab_ordering_type: VocabOrderingType specifying how to select vocabulary.
    x: Tensor to compute vocabulary over.
    file_format: The format of the resulting vocabulary file. Accepted formats
      are 'tfrecord_gzip', 'text'. 'tfrecord_gzip' requires tensorflow>=2.4.
    labels: Optional tensor of integerized labels.
    weights: Optional tensor of weights.

  Returns:
    A list of batch-reduced tensors to feed to vocabulary analysis.
  """
  filter_regex = get_vocab_newline_characters_regex(x.dtype, file_format)
  if vocab_ordering_type == _VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION:
    labels = tf.reshape(labels, [-1])
    reduced_batch = tf_utils.reduce_batch_weighted_cooccurrences(
        x, labels, weights, filter_regex=filter_regex)
    return [
        reduced_batch.unique_x, reduced_batch.summed_weights_per_x,
        reduced_batch.summed_positive_per_x_and_y, reduced_batch.counts_per_x
    ]
  elif vocab_ordering_type == _VocabOrderingType.MUTUAL_INFORMATION:
    labels = tf.reshape(labels, [-1])
    reduced_batch = tf_utils.reduce_batch_weighted_cooccurrences(
        x, labels, weights, filter_regex=filter_regex)
    return [
        reduced_batch.unique_x, reduced_batch.summed_positive_per_x_and_y,
        reduced_batch.counts_per_x
    ]
  elif vocab_ordering_type == _VocabOrderingType.WEIGHTED_FREQUENCY:
    reduced_batch = tf_utils.reduce_batch_weighted_counts(
        x, weights, filter_regex=filter_regex)
    assert reduced_batch.summed_positive_per_x_and_y is None
    assert reduced_batch.counts_per_x is None
    return [reduced_batch.unique_x, reduced_batch.summed_weights_per_x]
  else:
    reduced_batch = tf_utils.reduce_batch_weighted_counts(
        x, filter_regex=filter_regex)
    assert reduced_batch.summed_weights_per_x is None
    assert reduced_batch.summed_positive_per_x_and_y is None
    assert reduced_batch.counts_per_x is None
    return [reduced_batch.unique_x]


def get_vocab_newline_characters_regex(
    input_dtype: tf.dtypes.DType,
    file_format: common_types.VocabularyFileFormatType) -> Optional[str]:
  if input_dtype == tf.string and file_format == 'text':
    return _EMPTY_STRING_OR_NEWLINE_CHARS_REGEX
  else:
    return None


def _vocabulary_analyzer_nodes(
    analyzer_inputs: Collection[tf.Tensor],
    input_dtype: tf.dtypes.DType,
    vocab_ordering_type: int,
    vocab_filename: str,
    top_k: Optional[int] = None,
    frequency_threshold: int = 0,
    informativeness_threshold: float = float('-inf'),
    use_adjusted_mutual_info: bool = False,
    min_diff_from_avg: Optional[int] = None,
    fingerprint_shuffle: bool = False,
    store_frequency: bool = False,
    key_fn: Optional[Callable[[Any], Any]] = None,
    coverage_top_k: Optional[int] = None,
    coverage_frequency_threshold: float = 0.0,
    coverage_informativeness_threshold: float = float('-inf'),
    file_format: common_types
    .VocabularyFileFormatType = DEFAULT_VOCABULARY_FILE_FORMAT,
    vocabulary_key: Optional[str] = None
) -> common_types.TemporaryAnalyzerOutputType:
  """Internal helper for analyzing vocab. See `vocabulary` doc string."""
  if (file_format == 'tfrecord_gzip' and
      not tf_utils.is_vocabulary_tfrecord_supported()):
    raise ValueError(
        'Vocabulary file_format "tfrecord_gzip" not yet supported for '
        f'{tf.version.VERSION}.')

  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      analyzer_inputs)

  accumulate_output_value_node = nodes.apply_operation(
      analyzer_nodes.VocabularyAccumulate,
      input_values_node,
      vocab_ordering_type=vocab_ordering_type,
      input_dtype=input_dtype)

  merge_output_value_node = nodes.apply_operation(
      analyzer_nodes.VocabularyMerge,
      accumulate_output_value_node,
      use_adjusted_mutual_info=use_adjusted_mutual_info,
      min_diff_from_avg=min_diff_from_avg,
      vocab_ordering_type=vocab_ordering_type)

  filtered_value_node = nodes.apply_operation(
      analyzer_nodes.VocabularyPrune,
      merge_output_value_node,
      coverage_top_k=coverage_top_k,
      coverage_frequency_threshold=coverage_frequency_threshold,
      coverage_informativeness_threshold=coverage_informativeness_threshold,
      key_fn=key_fn,
      top_k=top_k,
      frequency_threshold=frequency_threshold,
      informativeness_threshold=informativeness_threshold,
      input_dtype=input_dtype)

  vocab_filename_node = nodes.apply_operation(
      analyzer_nodes.VocabularyOrderAndWrite,
      filtered_value_node,
      vocab_filename=vocab_filename,
      store_frequency=store_frequency,
      fingerprint_shuffle=fingerprint_shuffle,
      input_dtype=input_dtype,
      file_format=file_format,
      # LINT.IfChange(input_is_sorted)
      input_is_sorted=(top_k is not None and key_fn is None and
                       not fingerprint_shuffle)
      # LINT.ThenChange(beam/analyzer_impls.py:top_k_impl)
  )

  scope = tf.compat.v1.get_default_graph().get_name_scope()
  unfiltered_vocab_size_node = nodes.apply_operation(
      analyzer_nodes.VocabularyCount,
      merge_output_value_node,
      label=f'VocabularyCountUnfiltered[{scope}]')
  unfiltered_vocab_size = analyzer_nodes.bind_future_as_tensor(
      unfiltered_vocab_size_node,
      analyzer_nodes.TensorInfo(tf.int64, [], None),
      name=f'{vocab_filename}_unpruned_vocab_size')
  filtered_vocab_size_node = nodes.apply_operation(
      analyzer_nodes.VocabularyCount,
      filtered_value_node,
      label=f'VocabularyCountFiltered[{scope}]')
  filtered_vocab_size = analyzer_nodes.bind_future_as_tensor(
      filtered_vocab_size_node,
      analyzer_nodes.TensorInfo(tf.int64, [], None),
      name=f'{vocab_filename}_pruned_vocab_size')

  _maybe_annotate_vocab_metadata(vocab_filename, unfiltered_vocab_size,
                                 filtered_vocab_size)

  register_vocab(
      vocab_filename,
      vocabulary_size=filtered_vocab_size,
      vocabulary_key=vocabulary_key,
      file_format=file_format)
  return analyzer_nodes.wrap_as_tensor(vocab_filename_node)


def calculate_recommended_min_diff_from_avg(dataset_size: int) -> int:
  """Calculates a recommended min_diff_from_avg argument to tft.vocabulary.

  Computes a default min_diff_from_average parameter based on the size of the
  dataset. The MI (or AMI) of a token x label will be pushed to zero whenever
  the difference between the observed and the expected (average) cooccurrence
  with the label is < min_diff_from_average. This can be thought of as a
  regularization parameter for mutual information based vocabularies.

  Args:
    dataset_size: The number of recods in the dataset. The bigger the dataset,
      the higher the min_diff_from_average will be.

  Returns:
    An integer that is recomended to use as the min_diff_from_avg parameter of
    `vocabulary`.
  """
  # The minimum and maximum min_diff_from_avg parameter to use.
  min_value, max_value = 2, 25
  # Heuristics for a "small" and "large" dataset. The selected parameter will
  # be between min_value and max_value depending on where the dataset_size falls
  # relative to these values.
  small_dataset_size, large_dataset_size = 10000, 1000000
  return int(
      builtin_min(
          max_value,
          builtin_max(min_value, (dataset_size - small_dataset_size) /
                      (large_dataset_size - small_dataset_size) *
                      (max_value - min_value) + min_value)))


# Code related to this class is performance sensitive, so (micro-)benchmarks
# should be run when it is updated.
class QuantilesCombiner(analyzer_nodes.Combiner):
  """Computes quantiles on the PCollection.

  This implementation is based on go/squawd.
  For additional details on the algorithm, such as streaming and summary,
  see also http://web.cs.ucla.edu/~weiwang/paper/SSDBM07_2.pdf
  """

  def __init__(self,
               num_quantiles,
               epsilon,
               bucket_numpy_dtype,
               has_weights=False,
               output_shape=None,
               include_max_and_min=False,
               feature_shape=None):
    self._num_quantiles = num_quantiles
    self._epsilon = epsilon
    # Expected upper bound on the total number of input elements per feature.
    # Theoretical error bound is guaranteed to be <= epsilon as long as the
    # number of input elements is <= max_num_values.
    self._max_num_values = 1 << 32
    self._bucket_numpy_dtype = bucket_numpy_dtype
    self._has_weights = has_weights
    self._include_max_and_min = include_max_and_min
    num_outputs = (num_quantiles +
                   1) if include_max_and_min else (num_quantiles - 1)
    if feature_shape is None:
      feature_shape = []
    elif isinstance(feature_shape, int):
      feature_shape = [feature_shape]
    if output_shape is None:
      self._output_shape = list(feature_shape) + [num_outputs]
    else:
      self._output_shape = output_shape
    self._num_features = np.prod(feature_shape, dtype=np.int64).item()

  def create_accumulator(self):
    return sketches.QuantilesSketch(self._epsilon, self._max_num_values,
                                    self._num_features)

  def add_input(self, accumulator, next_input):
    # Flattened input array will be split on inputs for each feature.
    # C-contiguous order of flattened array is required.
    flat_values = pa.array(np.ravel(next_input[0]))
    if self._has_weights:
      flat_weights = pa.array(np.ravel(next_input[1]))
      accumulator.AddValues(flat_values, flat_weights)
    else:
      accumulator.AddValues(flat_values)
    return accumulator

  def merge_accumulators(self, accumulators):
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result.Merge(accumulator)
    return result

  def compact(self, accumulator):
    accumulator.Compact()
    return accumulator

  def extract_output(self, accumulator):
    result = accumulator.GetQuantiles(self._num_quantiles).to_pylist()
    if not result:
      return [np.zeros(self._output_shape, self._bucket_numpy_dtype)]
    result = np.array(result, self._bucket_numpy_dtype)
    # Trim elementwise results if max and min should be excluded.
    if not self._include_max_and_min:
      result = result[:, 1:-1]
    return [np.reshape(result, self._output_shape)]

  def output_tensor_infos(self):
    return [
        analyzer_nodes.TensorInfo(
            tf.as_dtype(self._bucket_numpy_dtype), self._output_shape, None)
    ]

  @property
  def accumulator_coder(self):
    return _QuantilesSketchCacheCoder()


class _QuantilesSketchCacheCoder(analyzer_nodes.CacheCoder):
  """Cache coder for the quantiles accumulator."""

  def encode_cache(self, accumulator):
    # TODO(b/174549940): Consider exposing and calling
    # `QuantilesSketch::Serialize` directly.
    # TODO(b/37788560): Should we be "intelligently" choosing the 'protocol'
    # argument for 'dumps'?
    return pickle.dumps(accumulator)

  def decode_cache(self, encoded_accumulator):
    return pickle.loads(encoded_accumulator)


@common.log_api_use(common.ANALYZER_COLLECTION)
def quantiles(x: tf.Tensor,
              num_buckets: int,
              epsilon: float,
              weights: Optional[tf.Tensor] = None,
              reduce_instance_dims: bool = True,
              name: Optional[str] = None) -> tf.Tensor:
  """Computes the quantile boundaries of a `Tensor` over the whole dataset.

  Quantile boundaries are computed using approximate quantiles,
  and error tolerance is specified using `epsilon`. The boundaries divide the
  input tensor into approximately equal `num_buckets` parts.
  See go/squawd for details, and how to control the error due to approximation.
  NaN input values and values with NaN weights are ignored.

  Args:
    x: An input `Tensor`.
    num_buckets: Values in the `x` are divided into approximately equal-sized
      buckets, where the number of buckets is `num_buckets`. The number of
      returned quantiles is `num_buckets` - 1.
    epsilon: Error tolerance, typically a small fraction close to zero (e.g.
      0.01). Higher values of epsilon increase the quantile approximation, and
      hence result in more unequal buckets, but could improve performance,
      and resource consumption.  Some measured results on memory consumption:
        For epsilon = 0.001, the amount of memory for each buffer to hold the
        summary for 1 trillion input values is ~25000 bytes. If epsilon is
        relaxed to 0.01, the buffer size drops to ~2000 bytes for the same input
        size. The buffer size also determines the amount of work in the
        different stages of the beam pipeline, in general, larger epsilon
        results in fewer and smaller stages, and less time. For more performance
        trade-offs see also http://web.cs.ucla.edu/~weiwang/paper/SSDBM07_2.pdf
    weights: (Optional) Weights tensor for the quantiles. Tensor must have the
      same batch size as x.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single output vector. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    The bucket boundaries represented as a list, with num_bucket-1 elements,
    unless reduce_instance_dims is False, which results in a Tensor of
    shape x.shape + [num_bucket-1].
    See code below for discussion on the type of bucket boundaries.
  """
  # Quantile ops convert input values to double under the hood. Keep bucket
  # boundaries as float for all numeric types.
  bucket_dtype = tf.float32
  with tf.compat.v1.name_scope(name, 'quantiles'):
    if weights is None:
      analyzer_inputs = [x]
      has_weights = False
    else:
      analyzer_inputs = [x, weights]
      has_weights = True
    feature_shape = [] if reduce_instance_dims else x.get_shape().as_list()[1:]
    output_shape = (feature_shape if feature_shape else [1]) + [num_buckets - 1]
    combiner = QuantilesCombiner(
        num_buckets,
        epsilon,
        bucket_dtype.as_numpy_dtype,
        has_weights=has_weights,
        output_shape=output_shape,
        feature_shape=feature_shape)
    (quantile_boundaries,) = _apply_cacheable_combiner(combiner,
                                                       *analyzer_inputs)
    return quantile_boundaries


def _quantiles_per_key(
    x: tf.Tensor,
    key: tf.Tensor,
    num_buckets: int,
    epsilon: float,
    weights: Optional[tf.Tensor] = None,
    name: Optional[str] = None
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, int]:
  """Like quantiles but per-key.

  For private use in tf.Transform implementation only.

  Args:
    x: An input `Tensor`.
    key: An input `Tensor` with rank 1 and size same as the fist dimension of
      `x`.  All values of `x` will be aggregated according to the corresponding
      value of `key`.
    num_buckets: See `quantiles`.
    epsilon: See `quantiles`.
    weights: See `quantiles`.
    name: (Optional) A name for this operation.

  Returns:
    A 4-tuple of (boundaries, scale, shift, num_buckets).
    The returned boundaries is a 1-d Tensor of size:
    ((num_buckets - 2) * num_keys) + 1

    And the returned scale and shift 1-d Tensors can be used to transform a
    value before applying bucketization and shift the resulting bucket.
    So the transformation of each input x before computing its bucket should be:
    F(x, key) = x * scale_factor_per_key[key] + shift_per_key[key]

    For example, if there are 2 keys, and the following boundaries are computed
    for them: [[0, 1, 2], [0, 1, 2]], this will return:
    boundaries: [0, 0.5, 1, 1.5, 2]
    scale_factor_per_key: [0.5, 0.5]
    shift_per_key: [0, 1]
    num_buckets: 4

  Raises:
    ValueError: If key has wrong dtype.
  """
  if key.dtype != tf.string:
    raise ValueError('key must have type tf.string')
  # Quantile ops convert input values to double under the hood. Keep bucket
  # boundaries as float for all numeric types.
  bucket_dtype = tf.float32
  with tf.compat.v1.name_scope(name, 'quantiles_by_key'):
    combiner = QuantilesCombiner(
        num_buckets,
        epsilon,
        bucket_dtype.as_numpy_dtype,
        has_weights=weights is not None,
        output_shape=(num_buckets - 1,))

    input_values_node = analyzer_nodes.get_input_tensors_value_nodes((
        key, x) if weights is None else (key, x, weights))

    accumulate_outputs_value_nodes = nodes.apply_multi_output_operation(
        analyzer_nodes.CacheableCombinePerKeyAccumulate,
        input_values_node,
        combiner=combiner)

    merge_output_value_node = nodes.apply_operation(
        analyzer_nodes.CacheableCombinePerKeyMerge,
        *accumulate_outputs_value_nodes,
        combiner=combiner)

    key_value_node, bucket_boundaries = nodes.apply_multi_output_operation(
        analyzer_nodes.CacheableCombinePerKeyFormatKeys,
        merge_output_value_node,
        combiner=combiner)

    boundaries, scale_factor, shift, num_buckets_node = (
        nodes.apply_multi_output_operation(
            analyzer_nodes.ScaleAndFlattenPerKeyBucketBouandaries,
            bucket_boundaries,
            output_tensor_dtype=bucket_dtype))

    return tuple(
        map(analyzer_nodes.wrap_as_tensor,
            [key_value_node, boundaries, scale_factor, shift, num_buckets_node
            ]))


class CovarianceCombiner(analyzer_nodes.Combiner):
  """Combines the PCollection to compute the biased covariance matrix."""

  def __init__(self, output_shape, numpy_dtype=np.float64):
    """Store the dtype and shape for np arrays/matrices for precision."""
    self._output_shape = output_shape
    self._numpy_dtype = numpy_dtype

  def create_accumulator(self):
    """Create an accumulator with all zero entries."""
    return [
        np.zeros((self._output_shape[0], self._output_shape[0]),
                 self._numpy_dtype),
        np.zeros((self._output_shape[0],), self._numpy_dtype),
        np.zeros((), self._numpy_dtype)
    ]

  def add_input(self, accumulator, batch_values):
    """Compute sum of input cross-terms, sum of inputs, and count.

    The cross terms for a numeric 1d array x are given by the set:
    {z_ij = x_i * x_j for all indices i and j}. This is stored as a 2d array.
    Since next_input is an array of 1d numeric arrays (i.e. a 2d array),
    matmul(transpose(next_input), next_input) will automatically sum up
    the cross terms of each 1d array in next_input.

    Args:
      accumulator: running sum of cross terms, input vectors, and count
      batch_values: entries from the pipeline, which must be single element list
          containing a 2d array
      representing multiple 1d arrays

    Returns:
      An accumulator with next_input considered in its running list of
      sum_product, sum_vectors, and count of input rows.
    """
    # Expect a single input representing the batch for the input tensor.
    batch_value, = batch_values

    assert len(np.shape(batch_value)) == 2

    batch_cross_terms = np.matmul(
        np.transpose(batch_value),
        batch_value
    ).astype(self._numpy_dtype)

    batch_sum = np.array(np.sum(batch_value, axis=0), self._numpy_dtype)
    batch_count = np.shape(batch_value)[0]

    sum_product, sum_vectors, count = accumulator

    return [
        sum_product + batch_cross_terms, sum_vectors + batch_sum,
        count + batch_count
    ]

  def merge_accumulators(self, accumulators):
    """Sums values in each accumulator entry."""
    # TODO(b/215378946): Consider updating accumulators[0] in place.
    products, vectors, counts = zip(*accumulators)
    return [
        np.sum(products, axis=0),
        np.sum(vectors, axis=0),
        np.sum(counts, axis=0)
    ]

  def extract_output(self, accumulator):
    """Run covariance logic on sum_product, sum of input vectors, and count.

    The formula used to compute the covariance is cov(x) = E(xx^T) - uu^T,
    where x is the original input to the combiner, and u = mean(x).
    E(xx^T) is computed by dividing sum of cross terms (index 0) by count
    (index 2). u is computed by taking the sum of rows (index 1) and dividing by
    the count (index 2).

    Args:
      accumulator: final accumulator as a list of the sum of cross-terms matrix,
        sum of input vectors, and count.

    Returns:
      A list containing a single 2d ndarray, the covariance matrix.
    """

    sum_product, sum_vectors, count = accumulator
    if count == 0:
      return [np.zeros(self._output_shape, self._numpy_dtype)]
    expected_cross_terms = sum_product / count
    expected_terms = sum_vectors / count
    return [
        np.ndarray.astype(  # TODO(b/64987151): # pytype: disable=attribute-error
            expected_cross_terms - np.outer(expected_terms, expected_terms),
            self._numpy_dtype)
    ]

  def output_tensor_infos(self):
    return [
        analyzer_nodes.TensorInfo(
            tf.as_dtype(self._numpy_dtype), self._output_shape, None)
    ]


@common.log_api_use(common.ANALYZER_COLLECTION)
def covariance(x: tf.Tensor,
               dtype: tf.DType,
               name: Optional[str] = None) -> tf.Tensor:
  """Computes the covariance matrix over the whole dataset.

  The covariance matrix M is defined as follows:
  Let x[:j] be a tensor of the jth element of all input vectors in x, and let
  u_j = mean(x[:j]). The entry M[i,j] = E[(x[:i] - u_i)(x[:j] - u_j)].
  Notice that the diagonal entries correspond to variances of individual
  elements in the vector, i.e. M[i,i] corresponds to the variance of x[:i].

  Args:
    x: A rank-2 `Tensor`, 0th dim are rows, 1st dim are indices in each input
      vector.
    dtype: Tensorflow dtype of entries in the returned matrix.
    name: (Optional) A name for this operation.

  Raises:
    ValueError: if input is not a rank-2 Tensor.

  Returns:
    A rank-2 (matrix) covariance `Tensor`
  """

  if not isinstance(x, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x)

  with tf.compat.v1.name_scope(name, 'covariance'):
    x.shape.assert_has_rank(2)

    input_dim = x.shape.as_list()[1]
    shape = (input_dim, input_dim)

    (result,) = _apply_cacheable_combiner(
        CovarianceCombiner(shape, dtype.as_numpy_dtype), x)
    return result


class PCACombiner(CovarianceCombiner):
  """Compute PCA of accumulated data using the biased covariance matrix."""

  def __init__(self, output_shape, output_dim=None, numpy_dtype=np.float64):
    """Store pca output dimension, shape and dtype for precision."""
    super().__init__(output_shape, numpy_dtype=numpy_dtype)
    self._output_dim = output_dim

  def extract_output(self, accumulator):
    """Compute PCA of the accumulated data using the biased covariance matrix.

    Following the covariance computation in CovarianceCombiner, this method runs
    eigenvalue decomposition on the covariance matrix, sorts eigenvalues in
    decreasing order, and returns the first output_dim corresponding
    eigenvectors (principal components) as a matrix.

    Args:
      accumulator: final accumulator as a list of the sum of cross-terms matrix,
        sum of input vectors, and count.

    Returns:
      A list containing a matrix of shape (input_dim, output_dim).
    """
    sum_product, sum_vectors, count = accumulator
    if count == 0:
      # In this case all eigenvalues==0 and we output (possibly truncated) basis
      # vectors. Note that if _output_dim is None, then M is set to N in np.eye.
      return [np.eye(N=self._output_shape[0], M=self._output_dim,
                     dtype=self._numpy_dtype)]
    expected_cross_terms = sum_product / count
    expected_terms = sum_vectors / count
    cov = np.ndarray.astype(  # TODO(b/64987151): # pytype: disable=attribute-error
        expected_cross_terms - np.outer(expected_terms, expected_terms),
        self._numpy_dtype)
    vals, vecs = np.linalg.eigh(cov)
    sorted_vecs = vecs[:, np.argsort(vals)[::-1]]
    if self._output_dim is None:
      return [sorted_vecs]
    else:
      return [sorted_vecs[:, :self._output_dim]]


@common.log_api_use(common.ANALYZER_COLLECTION)
def pca(x: tf.Tensor,
        output_dim: int,
        dtype: tf.DType,
        name: Optional[str] = None) -> tf.Tensor:
  """Computes PCA on the dataset using biased covariance.

  The PCA analyzer computes output_dim orthonormal vectors that capture
  directions/axes corresponding to the highest variances in the input vectors of
  `x`. The output vectors are returned as a rank-2 tensor with shape
  `(input_dim, output_dim)`, where the 0th dimension are the components of each
  output vector, and the 1st dimension are the output vectors representing
  orthogonal directions in the input space, sorted in order of decreasing
  variances.

  The output rank-2 tensor (matrix) serves a useful transform purpose. Formally,
  the matrix can be used downstream in the transform step by multiplying it to
  the input tensor `x`. This transform reduces the dimension of input vectors to
  output_dim in a way that retains the maximal variance.

  NOTE: To properly use PCA, input vector components should be converted to
  similar units of measurement such that the vectors represent a Euclidean
  space. If no such conversion is available (e.g. one element represents time,
  another element distance), the canonical approach is to first apply a
  transformation to the input data to normalize numerical variances, i.e.
  `tft.scale_to_z_score()`. Normalization allows PCA to choose output axes that
  help decorrelate input axes.

  Below are a couple intuitive examples of PCA.

  Consider a simple 2-dimensional example:

  Input x is a series of vectors `[e, e]` where `e` is Gaussian with mean 0,
  variance 1. The two components are perfectly correlated, and the resulting
  covariance matrix is

  ```
  [[1 1],
   [1 1]].
  ```

  Applying PCA with `output_dim = 1` would discover the first principal
  component `[1 / sqrt(2), 1 / sqrt(2)]`. When multipled to the original
  example, each vector `[e, e]` would be mapped to a scalar `sqrt(2) * e`. The
  second principal component would be `[-1 / sqrt(2), 1 / sqrt(2)]` and would
  map `[e, e]` to 0, which indicates that the second component captures no
  variance at all. This agrees with our intuition since we know that the two
  axes in the input are perfectly correlated and can be fully explained by a
  single scalar `e`.

  Consider a 3-dimensional example:

  Input `x` is a series of vectors `[a, a, b]`, where `a` is a zero-mean, unit
  variance Gaussian and `b` is a zero-mean, variance 4 Gaussian and is
  independent of `a`. The first principal component of the unnormalized vector
  would be `[0, 0, 1]` since `b` has a much larger variance than any linear
  combination of the first two components. This would map `[a, a, b]` onto `b`,
  asserting that the axis with highest energy is the third component. While this
  may be the desired output if `a` and `b` correspond to the same units, it is
  not statistically desireable when the units are irreconciliable. In such a
  case, one should first normalize each component to unit variance first, i.e.
  `b := b / 2`. The first principal component of a normalized vector would yield
  `[1 / sqrt(2), 1 / sqrt(2), 0]`, and would map `[a, a, b]` to `sqrt(2) * a`.
  The second component would be `[0, 0, 1]` and map `[a, a, b]` to `b`. As can
  be seen, the benefit of normalization is that PCA would capture highly
  correlated components first and collapse them into a lower dimension.

  Args:
    x: A rank-2 `Tensor`, 0th dim are rows, 1st dim are indices in row vectors.
    output_dim: The PCA output dimension (number of eigenvectors to return).
    dtype: Tensorflow dtype of entries in the returned matrix.
    name: (Optional) A name for this operation.

  Raises:
    ValueError: if input is not a rank-2 Tensor.

  Returns:
    A 2D `Tensor` (matrix) M of shape (input_dim, output_dim).
  """

  if not isinstance(x, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x)

  with tf.compat.v1.name_scope(name, 'pca'):
    x.shape.assert_has_rank(2)

    input_dim = x.shape.as_list()[1]
    shape = (input_dim, output_dim)

    (result,) = _apply_cacheable_combiner(
        PCACombiner(shape, output_dim, dtype.as_numpy_dtype), x)
    return result


def _maybe_annotate_vocab_metadata(vocab_filename: str,
                                   unfiltered_vocabulary_size: tf.Tensor,
                                   filtered_vocabulary_size: tf.Tensor):
  """Annotates a bucketized tensor with the boundaries that were applied.

  Creates a deferred annotation for the specified tensor.

  Args:
    vocab_filename: The name of the vocabulary.
    unfiltered_vocabulary_size: A tf.int64 tensor containing the unfiltered
      vocab size.
    filtered_vocabulary_size: A tf.int64 tensor containing the filtered vocab
      size.
  """
  if not common.IS_ANNOTATIONS_PB_AVAILABLE:
    return

  from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top
  message_type = annotations_pb2.VocabularyMetadata.DESCRIPTOR.full_name
  unfiltered_vocabulary_size = tf.expand_dims(unfiltered_vocabulary_size, 0)
  filtered_vocabulary_size = tf.expand_dims(filtered_vocabulary_size, 0)
  file_name = tf.convert_to_tensor([vocab_filename])
  descriptor_source = descriptor_pb2.FileDescriptorSet()
  annotations_pb2.VocabularyMetadata.DESCRIPTOR.file.CopyToProto(
      descriptor_source.file.add())
  descriptor_source_str = b'bytes://' + descriptor_source.SerializeToString()
  message_proto = tf_utils._encode_proto(  # pylint: disable=protected-access
      {
          'unfiltered_vocabulary_size': unfiltered_vocabulary_size,
          'filtered_vocabulary_size': filtered_vocabulary_size,
          'file_name': file_name,
      }, message_type, descriptor_source=descriptor_source_str)
  assert message_proto.shape == [1]
  message_proto = message_proto[0]

  # Note: we annotate globally here (tied to a vocabulary by filename) rather
  # than attaching to a tensor, because this annotation is tied to an analysis
  # output not a final tensor produced by a mapper.
  type_url = os.path.join(common.ANNOTATION_PREFIX_URL, message_type)
  schema_inference.annotate(type_url, message_proto)
