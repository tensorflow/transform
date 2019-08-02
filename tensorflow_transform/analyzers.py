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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pickle
import re

# GOOGLE-INITIALIZATION
import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import nodes
from tensorflow_transform import tf_utils


from tensorflow.contrib.boosted_trees.python.ops import gen_quantile_ops
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops
from tensorflow.python.ops import resources
from tensorflow.python.util import deprecation

__all__ = [
    'covariance',
    'max',
    'mean',
    'min',
    'pca',
    'ptransform_analyzer',
    'quantiles',
    'size',
    'sum',
    'uniques',
    'var',
    'vocabulary',
]

# This module defines max and min functions that override the builtins.
builtin_max = max
builtin_min = min


VOCAB_FILENAME_PREFIX = 'vocab_'
VOCAB_FREQUENCY_FILENAME_PREFIX = 'vocab_frequency_'

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

_MEAN_OUTPUT_DTYPE_MAP = {
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


def apply_analyzer(analyzer_def_cls, *tensor_inputs, **analyzer_def_kwargs):
  """Applies the analyzer over the whole dataset.

  Args:
    analyzer_def_cls: A class inheriting from analyzer_nodes.AnalyzerDef that
      should be applied.
    *tensor_inputs: A list of input `Tensor`s or `SparseTensor`s.
    **analyzer_def_kwargs: KW arguments to use when constructing
      analyzer_def_cls.

  Returns:
    A list of `Tensor`s representing the values of the analysis result.
  """
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)
  output_value_nodes = nodes.apply_multi_output_operation(
      analyzer_def_cls,
      input_values_node,
      **analyzer_def_kwargs)
  return tuple(map(analyzer_nodes.wrap_as_tensor, output_value_nodes))


def _apply_cacheable_combiner(combiner, *tensor_inputs):
  """Applies the combiner over the whole dataset possibly utilizing cache."""
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)

  accumulate_outputs_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombineAccumulate,
      input_values_node,
      combiner=combiner)

  outputs_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombineMerge,
      *accumulate_outputs_value_nodes,
      combiner=combiner)

  return tuple(map(analyzer_nodes.wrap_as_tensor, outputs_value_nodes))


def _apply_cacheable_combiner_per_key(combiner, *tensor_inputs):
  """Similar to _apply_cacheable_combiner but this is computed per key."""
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)

  accumulate_outputs_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombinePerKeyAccumulate,
      input_values_node,
      combiner=combiner)

  output_value_nodes = nodes.apply_multi_output_operation(
      analyzer_nodes.CacheableCombinePerKeyMerge,
      *accumulate_outputs_value_nodes,
      combiner=combiner)

  return tuple(map(analyzer_nodes.wrap_as_tensor, output_value_nodes))


class NumPyCombiner(analyzer_nodes.Combiner):
  """Combines the PCollection only on the 0th dimension using nparray.

  Args:
    fn: The numpy function representing the reduction to be done.
    output_dtypes: The numpy dtype to cast each output to.
    output_shapes: The shapes of the outputs.
  """

  def __init__(self, fn, output_dtypes, output_shapes):
    self._fn = fn
    self._output_dtypes = output_dtypes
    self._output_shapes = output_shapes

  # TODO(b/34792459): merge_accumulators and extract_output assume that not all
  # accumulator(s) are None.  This only works when .without_defaults() is
  # used but even in that case it is an implementation detail of Beam that we
  # should not be relying on.  Instead we should use 0 or +-inf depending on the
  # accumulator. Invoking self._fn(()) might also be a good way of determining
  # the default (works for some but not all fns).
  def create_accumulator(self):
    return None

  def add_input(self, accumulator, batch_values):
    # TODO(b/112414577): Go back to accepting only a single input.
    # See comment in _numeric_combine.
    reduced_values = batch_values
    if accumulator is None:
      return reduced_values
    else:
      return [
          self._fn((sub_accumulator, reduced_value), axis=0)
          for sub_accumulator, reduced_value
          in zip(accumulator, reduced_values)]

  def merge_accumulators(self, accumulators):
    non_empty_accumulators = [
        accumulator for accumulator in accumulators if accumulator is not None
    ]
    if non_empty_accumulators:
      return [
          # numpy's sum, min, max, etc functions operate on array-like objects,
          # but not arbitrary iterables. Convert the provided sub_accumulators
          # into a list.
          self._fn(list(sub_accumulators), axis=0)
          for sub_accumulators in zip(*non_empty_accumulators)]
    else:
      return None

  def extract_output(self, accumulator):
    if accumulator is None:
      return None
    else:
      # For each output, cast that output to the specified type.  Note there
      # will be one output for each input tensor to the analyzer.
      return [sub_accumulator.astype(output_dtype)
              for sub_accumulator, output_dtype
              in zip(accumulator, self._output_dtypes)]

  def output_tensor_infos(self):
    return [
        analyzer_nodes.TensorInfo(tf.as_dtype(dtype), shape, False)
        for dtype, shape in zip(self._output_dtypes, self._output_shapes)
    ]


def _get_output_shape_from_input(x):
  if isinstance(x, tf.SparseTensor):
    return x.get_shape()[1:]

  # When reducing over batch dimensions, with known shape, the result will be
  # the same shape as the input, but without the batch.
  if x.shape.dims is not None:
    return x.shape.as_list()[1:]
  return (None,)


# TODO(b/112414577): Go back to accepting only a single input.
# Currently we accept multiple inputs so that we can implement min and max
# with a single combiner.
def _numeric_combine(inputs,
                     fn,
                     reduce_instance_dims=True,
                     output_dtypes=None,
                     key=None):
  """Apply a reduction, defined by a numpy function to multiple inputs.

  Args:
    inputs: A list of tensors, which will be independently reduced.
    fn: A function to reduce tensors across instances/batches, to get a single
        output.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    output_dtypes: (Optional) A list of dtypes of the output tensors. If None,
        the output tensor has the same type as the input one.
    key: (Optional) Apply the same operation, but on a per-key basis.

  Returns:
     A list of tensors with the same length as `inputs`, representing the
         input tensors that have been reduced by `fn` across instances and
         batches.
  """
  for x in inputs:
    if not isinstance(x, tf.Tensor):
      raise TypeError('Expected a Tensor, but got %r' % x)

  if output_dtypes is None:
    output_dtypes = [x.dtype for x in inputs]
  if reduce_instance_dims:
    # If reducing over all dimensions, result is scalar.
    output_shapes = [() for _ in inputs]
  else:
    # Reducing over batch dimensions.
    output_shapes = [x.get_shape() for x in inputs]
  combiner = NumPyCombiner(
      fn, [dtype.as_numpy_dtype for dtype in output_dtypes], output_shapes)
  if key is None:
    return _apply_cacheable_combiner(combiner, *inputs)
  return _apply_cacheable_combiner_per_key(combiner, key, *inputs)


def min(x, reduce_instance_dims=True, name=None):  # pylint: disable=redefined-builtin
  """Computes the minimum of the values of a `Tensor` over the whole dataset.

  In the case of a `SparseTensor` missing values will be used in return value:
  for float, NaN is used and for other dtypes the max is used.

  Args:
    x: A `Tensor` or `SparseTensor`.
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


def max(x, reduce_instance_dims=True, name=None):  # pylint: disable=redefined-builtin
  """Computes the maximum of the values of a `Tensor` over the whole dataset.

  In the case of a `SparseTensor` missing values will be used in return value:
  for float, NaN is used and for other dtypes the min is used.

  Args:
    x: A `Tensor` or `SparseTensor`.
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


def _min_and_max(x, reduce_instance_dims=True, name=None):
  """Computes the min and max of the values of a `Tensor` or `SparseTensor`.

  In the case of a `SparseTensor` missing values will be used in return value:
  for float, NaN is used and for other dtypes the min is used.

  Args:
    x: A `Tensor` or `SparseTensor`.
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
    combine_fn = np.max
    if (not reduce_instance_dims and isinstance(x, tf.SparseTensor) and
        x.dtype.is_floating):
      combine_fn = np.nanmax

    output_dtype = x.dtype

    x_batch_minus_min, x_batch_max = tf_utils.reduce_batch_minus_min_and_max(
        x, reduce_instance_dims)

    minus_x_min, x_max = _numeric_combine(  # pylint: disable=unbalanced-tuple-unpacking
        [x_batch_minus_min, x_batch_max], combine_fn, reduce_instance_dims)
    return tf.cast(0 - minus_x_min, output_dtype), tf.cast(x_max, output_dtype)


def _min_and_max_per_key(x, key, reduce_instance_dims=True, name=None):
  """Computes the min and max of the values of a `Tensor` or `SparseTensor`.

  In the case of a `SparseTensor` missing values will be used in return value:
  for float, NaN is used and for other dtypes the min is used.

  Args:
    x: A `Tensor` or `SparseTensor`.
    key: A Tensor or `SparseTensor` of dtype tf.string.  If `x` is
      a `SparseTensor`, `key` must exactly match `x` in everything except
      values.
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
    combine_fn = np.max
    if (not reduce_instance_dims and isinstance(x, tf.SparseTensor) and
        x.dtype.is_floating):
      combine_fn = np.nanmax

    output_dtype = x.dtype

    key_vocab, x_batch_minus_min, x_batch_max = (
        tf_utils.reduce_batch_minus_min_and_max_per_key(x, key))

    key, minus_x_min, x_max = _numeric_combine(  # pylint: disable=unbalanced-tuple-unpacking
        [x_batch_minus_min, x_batch_max],
        combine_fn,
        reduce_instance_dims,
        key=key_vocab)
    return (
        key,
        tf.cast(0 - minus_x_min, output_dtype),
        tf.cast(x_max, output_dtype))


def _sum_combine_fn_and_dtype(input_dtype):
  output_dtype = _SUM_OUTPUT_DTYPE_MAP.get(input_dtype)
  if output_dtype is None:
    raise TypeError('Tensor type %r is not supported' % input_dtype)

  def sum_fn_with_dtype(a, axis=None):
    return np.sum(a, axis=axis, dtype=output_dtype.as_numpy_dtype)

  return output_dtype, sum_fn_with_dtype


def sum(x, reduce_instance_dims=True, name=None):  # pylint: disable=redefined-builtin
  """Computes the sum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor` or `SparseTensor`. Its type must be floating point
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
      if isinstance(x, tf.SparseTensor):
        x = x.values
      x = tf.reduce_sum(input_tensor=x)
    elif isinstance(x, tf.SparseTensor):
      if x.dtype == tf.uint8 or x.dtype == tf.uint16:
        x = tf.cast(x, tf.int64)
      elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
        TypeError('Data type %r is not supported' % x.dtype)
      x = tf.sparse.reduce_sum(x, axis=0)
    else:
      x = tf.reduce_sum(input_tensor=x, axis=0)
    output_dtype, sum_fn = _sum_combine_fn_and_dtype(x.dtype)
    return _numeric_combine([x], sum_fn, reduce_instance_dims,
                            [output_dtype])[0]


def size(x, reduce_instance_dims=True, name=None):
  """Computes the total size of instances in a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor` or `SparseTensor`.
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


def mean(x, reduce_instance_dims=True, name=None, output_dtype=None):
  """Computes the mean of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor` or `SparseTensor`. Its type must be floating point
        (float{16|32|64}), or integral ([u]int{8|16|32|64}).
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` containing the mean. If `x` is floating point, the mean will have
    the same type as `x`. If `x` is integral, the output is cast to float32.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'mean'):
    return _mean_and_var(x, reduce_instance_dims, output_dtype)[0]


def var(x, reduce_instance_dims=True, name=None, output_dtype=None):
  """Computes the variance of the values of a `Tensor` over the whole dataset.

  Uses the biased variance (0 delta degrees of freedom), as given by
  (x - mean(x))**2 / length(x).

  Args:
    x: `Tensor` or `SparseTensor`. Its type must be floating point
        (float{16|32|64}), or integral ([u]int{8|16|32|64}).
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` containing the variance. If `x` is floating point, the variance
    will have the same type as `x`. If `x` is integral, the output is cast to
    float32.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  with tf.compat.v1.name_scope(name, 'var'):
    return _mean_and_var(x, reduce_instance_dims, output_dtype)[1]


def _mean_and_var(x, reduce_instance_dims=True, output_dtype=None):
  """More efficient combined `mean` and `var`.  See `var`."""
  if output_dtype is None:
    output_dtype = _MEAN_OUTPUT_DTYPE_MAP.get(x.dtype)
    if output_dtype is None:
      raise TypeError('Tensor type %r is not supported' % x.dtype)

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


def _mean_and_var_per_key(x, key, reduce_instance_dims=True, output_dtype=None):
  """`mean_and_var` by group, specified by key."""
  if output_dtype is None:
    output_dtype = _MEAN_OUTPUT_DTYPE_MAP.get(x.dtype)
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

    key, key_mean, key_var = _apply_cacheable_combiner_per_key(
        WeightedMeanAndVarCombiner(output_dtype.as_numpy_dtype, output_shape),
        key_vocab,
        *combine_inputs)

  return key, key_mean, key_var


class _WeightedMeanAndVarAccumulator(
    collections.namedtuple('WeightedMeanAndVarAccumulator',
                           ['count', 'mean', 'variance', 'weight'])):
  """Container for WeightedMeanAndVarCombiner intermediate values."""

  @classmethod
  def make_nan_to_num(cls, counts, means, variances, weights):
    return cls(
        np.array(counts), np.nan_to_num(means), np.nan_to_num(variances),
        np.nan_to_num(weights))

  def __reduce__(self):
    return self.__class__, tuple(self)


class WeightedMeanAndVarCombiner(analyzer_nodes.Combiner):
  """Combines a PCollection of accumulators to compute mean and variance."""

  accumulator_class = _WeightedMeanAndVarAccumulator

  def __init__(self,
               output_numpy_dtype,
               output_shape=None,
               compute_variance=True,
               compute_weighted=False):
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

  def create_accumulator(self):
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

  def add_input(self, accumulator, batch_values):
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

  def merge_accumulators(self, accumulators):
    """Merges several `_WeightedMeanAndVarAccumulator`s to a single accumulator.

    Args:
      accumulators: A list of `_WeightedMeanAndVarAccumulator`s.

    Returns:
      The sole merged `_WeightedMeanAndVarAccumulator`.
    """
    result = self.create_accumulator()
    for accumulator in accumulators:
      result = self._combine_mean_and_var_accumulators(result, accumulator)
    return result

  def extract_output(self, accumulator):
    """Converts an accumulator into the output (mean, var) tuple.

    Args:
      accumulator: the final `_WeightedMeanAndVarAccumulator` value.

    Returns:
      A 2-tuple composed of (mean, var).
    """
    if self._compute_variance and not self._compute_weighted:
      return (self._output_numpy_dtype(accumulator.mean),
              self._output_numpy_dtype(accumulator.variance))
    else:
      return accumulator

  def output_tensor_infos(self):
    # The output is (mean, var).
    return [
        analyzer_nodes.TensorInfo(tf.as_dtype(self._output_numpy_dtype),
                                  self._output_shape,
                                  False)
    ] * 2

  def compute_running_update(self, total_count, current_count, update):
    """Numerically stable way of computing a streaming batched update."""
    return (current_count / total_count) * update

  def _combine_mean_and_var_accumulators(self, a, b):
    """Combines two mean and var accumulators.

    Args:
      a: A _WeightedMeanAndVarAccumulator.
      b: A _WeightedMeanAndVarAccumulator.

    Returns:
      A _WeightedMeanAndVarAccumulator computed as the combination of a and b.
    """
    # NaNs get preserved through division by a.count + b.count.
    a = _WeightedMeanAndVarAccumulator.make_nan_to_num(*a)
    b = _WeightedMeanAndVarAccumulator.make_nan_to_num(*b)

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
    else:
      combined_weights_mean = np.ones(shape=combined_total.shape)
      b_weight = np.ones(shape=b_mean.shape)

    combined_mean = a_mean + (b_count * b_weight /
                              (combined_total * combined_weights_mean)) * (
                                  b_mean - a_mean)
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
      "uniques" scope name in the context of this graph will be used as the file
      name.
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


# TODO(b/116308354): frequency_threshold is misleading since this threshold can
# be applied to mutual information rather than frequency.
def _get_top_k_and_frequency_threshold(top_k, frequency_threshold):
  """Validate `top_k` and `frequency_threshold` values and convert to number."""
  if top_k is not None:
    top_k = int(top_k)
    if top_k < 0:
      raise ValueError('top_k must be non-negative, but got: %r' % top_k)

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


class _VocabOrderingType(object):
  # Orders vocabulary based on the simple frequency of the token
  FREQUENCY = 1
  # Orders vocabulary based on the weighted frequency of the token
  WEIGHTED_FREQUENCY = 2
  # Orders vocabulary based on the mutual information of token with the label
  WEIGHTED_MUTUAL_INFORMATION = 3


# TODO(KesterTong): Once multiple outputs are supported, return indices too.
# TODO(b/117796748): Add coverage key feature input as alternative to `key_fn`.
# TODO(b/116308354): rename store_frequency to store_importance because it now
# can return mutual information.
# TODO(tensorflow/community) the experimental fingerprint_shuffle argument is a
# workaround for the inability to appropriately rebalance sharded variables on
# TF 1.0. The following TF 2.0 proposal should address this issue in the future
# https://github.com/tensorflow/community/blob/master/rfcs/20190116-embedding-partitioned-variable.md#goals
def vocabulary(x,
               top_k=None,
               frequency_threshold=None,
               vocab_filename=None,
               store_frequency=False,
               weights=None,
               labels=None,
               use_adjusted_mutual_info=False,
               min_diff_from_avg=None,
               coverage_top_k=None,
               coverage_frequency_threshold=None,
               key_fn=None,
               fingerprint_shuffle=False,
               name=None):
  r"""Computes the unique values of a `Tensor` over the whole dataset.

  Computes The unique values taken by `x`, which can be a `Tensor` or
  `SparseTensor` of any size.  The unique values will be aggregated over all
  dimensions of `x` and all instances.

  In case one of the tokens contains the '\n' or '\r' characters or is empty it
  will be discarded since we are currently writing the vocabularies as text
  files. This behavior will likely be fixed/improved in the future.

  If an integer `Tensor` is provided, its semantic type should be categorical
  not a continuous/numeric, since computing a vocabulary over a continuous
  feature is not appropriate.

  The unique values are sorted by decreasing frequency and then reverse
  lexicographical order (e.g. [('a', 5), ('c', 3), ('b', 3)]).

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
    x: A categorical/discrete input `Tensor` or `SparseTensor` with dtype
      tf.string or tf.int[8|16|32|64]. The inputs should generally be unique per
      row (i.e. a bag of words/ngrams representation).
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      absolute frequency is >= to the supplied threshold. If set to None, the
      full vocabulary is generated.  Absolute frequency means the number of
      occurrences of the element in the dataset, as opposed to the proportion of
      instances that contain that element.
    vocab_filename: The file name for the vocabulary file. If none, the
      "uniques" scope name in the context of this graph will be used as the file
      name. If not None, should be unique within a given preprocessing function.
      NOTE To make your pipelines resilient to implementation details please
      set `vocab_filename` when you are using the vocab_filename on a downstream
      component.
    store_frequency: If True, frequency of the words is stored in the
      vocabulary file. In the case labels are provided, the mutual
      information is stored in the file instead. Each line in the file
      will be of the form 'frequency word'. NOTE: if this is True then the
      computed vocabulary cannot be used with `tft.apply_vocabulary` directly,
      since frequencies are added to the beginning of each row of the
      vocabulary, which the mapper will not ignore.
    weights: (Optional) Weights `Tensor` for the vocabulary. It must have the
      same shape as x.
    labels: (Optional) Labels dense `Tensor` for the vocabulary. If provided,
      the vocabulary is calculated based on mutual information with the label,
      rather than frequency. The labels must have the same batch dimension as x.
      If x is sparse, labels should be a 1D tensor reflecting row-wise labels.
      If x is dense, labels can either be a 1D tensor of row-wise labels, or
      a dense tensor of the identical shape as x (i.e. element-wise labels).
      Labels should be a discrete integerized tensor (If the label is numeric,
      it should first be bucketized; If the label is a string, an integer
      vocabulary should first be applied). Note: `SparseTensor` labels are not
      yet supported (b/134931826).
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

  coverage_top_k, coverage_frequency_threshold = (
      _get_top_k_and_frequency_threshold(
          coverage_top_k, coverage_frequency_threshold))

  if x.dtype != tf.string and not x.dtype.is_integer:
    raise ValueError('expected tf.string or integer but got %r' % x.dtype)

  if labels is not None and not labels.dtype.is_integer:
    raise ValueError('expected integer labels but got %r' % labels.dtype)

  with tf.compat.v1.name_scope(name, 'vocabulary'):
    vocab_filename = _get_vocab_filename(vocab_filename, store_frequency)

    if labels is not None:
      labels = tf.reshape(labels, [-1])
      vocab_ordering_type = _VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION
      reduced_batch = tf_utils.reduce_batch_weighted_cooccurrences(
          x, labels, weights)
      analyzer_inputs = [
          reduced_batch.unique_x, reduced_batch.summed_weights_per_x,
          reduced_batch.summed_positive_per_x_and_y, reduced_batch.counts_per_x
      ]
    elif weights is not None:
      vocab_ordering_type = _VocabOrderingType.WEIGHTED_FREQUENCY
      reduced_batch = tf_utils.reduce_batch_weighted_counts(x, weights)
      assert reduced_batch.summed_positive_per_x_and_y is None
      assert reduced_batch.counts_per_x is None
      analyzer_inputs = [
          reduced_batch.unique_x, reduced_batch.summed_weights_per_x
      ]
    else:
      vocab_ordering_type = _VocabOrderingType.FREQUENCY
      reduced_batch = tf_utils.reduce_batch_weighted_counts(x)
      assert reduced_batch.summed_weights_per_x is None
      assert reduced_batch.summed_positive_per_x_and_y is None
      assert reduced_batch.counts_per_x is None
      analyzer_inputs = [reduced_batch.unique_x]

    input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
        analyzer_inputs)

    accumulate_output_value_node = nodes.apply_operation(
        analyzer_nodes.VocabularyAccumulate,
        input_values_node,
        vocab_ordering_type=vocab_ordering_type,
        input_dtype=x.dtype.name)

    merge_output_value_node = nodes.apply_operation(
        analyzer_nodes.VocabularyMerge, accumulate_output_value_node,
        use_adjusted_mutual_info=use_adjusted_mutual_info,
        min_diff_from_avg=min_diff_from_avg,
        vocab_ordering_type=vocab_ordering_type)

    filtered_value_node = nodes.apply_operation(
        analyzer_nodes.VocabularyOrderAndFilter,
        merge_output_value_node,
        coverage_top_k=coverage_top_k,
        coverage_frequency_threshold=coverage_frequency_threshold,
        key_fn=key_fn,
        top_k=top_k,
        frequency_threshold=frequency_threshold)

    vocab_filename_node = nodes.apply_operation(
        analyzer_nodes.VocabularyWrite,
        filtered_value_node,
        vocab_filename=vocab_filename,
        store_frequency=store_frequency,
        fingerprint_shuffle=fingerprint_shuffle,
        input_dtype=x.dtype.name)

    vocab_filename = analyzer_nodes.wrap_as_tensor(vocab_filename_node)
    return vocab_filename


def calculate_recommended_min_diff_from_avg(dataset_size):
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


@deprecation.deprecated(None, 'Use `tft.vocabulary()` instead.')
def uniques(x,
            top_k=None,
            frequency_threshold=None,
            vocab_filename=None,
            store_frequency=False,
            weights=None,
            labels=None,
            name=None):
  r"""See `tft.vocabulary`."""
  return vocabulary(
      x=x,
      top_k=top_k,
      frequency_threshold=frequency_threshold,
      vocab_filename=vocab_filename,
      store_frequency=store_frequency,
      weights=weights,
      labels=labels,
      name=name)


# TODO(b/65627483): Make this an instantiation of a generic CombineFn based on
# TF ops.
#
# TODO(KesterTong): It seems like QuantilesCombiner is using the state of the
# current object as the accumulator (as opposed to using a bonafide
# accumulator). We should change that to ensure correctness (I believe we
# currently rely on runner implementation details), portability and
# ease of understanding of the code. The fact that a summary/accumulator can be
# None is also confusing.
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
               always_return_num_quantiles=False,
               has_weights=False,
               output_shape=None,
               include_max_and_min=False,
               feature_shape=None):
    self._num_quantiles = num_quantiles
    self._epsilon = epsilon
    self._bucket_numpy_dtype = bucket_numpy_dtype
    self._always_return_num_quantiles = always_return_num_quantiles
    self._has_weights = has_weights
    self._output_shape = output_shape
    self._include_max_and_min = include_max_and_min
    self._feature_shape = [] if feature_shape is None else feature_shape
    if isinstance(self._feature_shape, int):
      self._feature_shape = [self._feature_shape]
    self._num_features = np.prod(self._feature_shape, dtype=np.int64)
    if not self._always_return_num_quantiles and self._num_features > 1:
      raise NotImplementedError(
          'Elementwise quantiles requires same boundary count.')

  # TODO(b/69566045): Move initialization to start_bundle() or follow the
  # _start_bundle() approach that TFMA has taken and get rid of the __reduce__
  # override below.
  def initialize_local_state(self, tf_config=None):
    """Called by the CombineFnWrapper's __init__ method.

    This can be used to set non-pickleable local state.  It is used in
    conjunction with overriding __reduce__ so this state is not pickled.  This
    method must be called prior to any other method.

    Args:
      tf_config: (optional) A tf.ConfigProto
    """
    # Create a new session with a new graph for quantile ops.
    with tf.Graph().as_default() as graph:

      # stamp_token is used to commit the state of each qaccumulator. In
      # this case, the qaccumulator state is completely returned and stored
      # as part of quantile_state/summary in the combiner fn (i.e the summary is
      # extracted and stored outside the qaccumulator). So we don't use
      # the timestamp mechanism to signify progress in the qaccumulator state.
      stamp_token = 0

      self._session = tf.compat.v1.Session(graph=graph)

      qaccumulators = []
      for idx in range(self._num_features):
        qaccumulators.append(quantile_ops.QuantileAccumulator(
            init_stamp_token=stamp_token,
            num_quantiles=self._num_quantiles,
            epsilon=self._epsilon,
            name='qaccumulator_{}'.format(idx),
            generate_quantiles=self._always_return_num_quantiles))
      qaccumulators = tuple(qaccumulators)

      self._session.run(
          resources.initialize_resources(resources.shared_resources()))

      self._add_input_callable = self._make_add_input_callable(
          qaccumulators, stamp_token)
      self._merge_inputs_callable = self._make_add_summary_callable(
          qaccumulators, stamp_token)
      self._get_buckets_callable = self._make_get_buckets_callable(
          qaccumulators, stamp_token)

      # Create op to flush summaries and return a list representing the
      # summaries that were added to all accumulators so far.
      self._flush_summary_callable = self._session.make_callable(
          fetches=[qacc.flush_summary(stamp_token=stamp_token,
                                      next_stamp_token=stamp_token)
                   for qacc in qaccumulators])

      graph.finalize()

    # We generate an empty summary by calling self._flush_summary_op.
    # We cache this as some implementations may call create_accumulator for
    # every input, and it can be cached since it will always be the same and
    # immutable.
    self._empty_summary = self._flush_summary_callable()[0]

  def __reduce__(self):
    return QuantilesCombiner, (self._num_quantiles,
                               self._epsilon,
                               self._bucket_numpy_dtype,
                               self._always_return_num_quantiles,
                               self._has_weights,
                               self._output_shape,
                               self._include_max_and_min,
                               self._feature_shape)

  def _make_add_input_callable(self, qaccumulators, stamp_token):
    # Create placeholders for add_inputs_callable.  These placeholders will
    # be used to provide prebuilt summary, input and weights to the
    # QuantileAccumulator.
    # inputs and weights need to have shapes (1, None) as this is what the
    # QuantileAccumulator accepts.
    prebuilt_summary = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[self._num_features], name='summaries')
    inputs = tf.compat.v1.placeholder(
        dtype=self._bucket_numpy_dtype, shape=[self._num_features, None],
        name='inputs')
    feed_list = [prebuilt_summary, inputs]
    if self._has_weights:
      weights = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None],
                                         name='weights')
      feed_list.append(weights)
    else:
      weights = tf.expand_dims(tf.ones_like(inputs[0, :]), axis=0)

    # TODO(b/68277922): Investigate add_inputs() to efficiently handle multiple
    # batches of inputs.
    # This is where we can most parallelize the operation, so we should
    # refrain from using accumulators until necessary to merge
    next_summaries = gen_quantile_ops.make_quantile_summaries(
        dense_float_features=[[data] for data in tf.unstack(inputs, axis=0)],
        sparse_float_feature_indices=[],
        sparse_float_feature_values=[],
        sparse_float_feature_shapes=[],
        example_weights=weights,
        epsilon=self._epsilon / 2).dense_summaries

    summaries = []
    for summary_ix in range(self._num_features):
      add_prebuilt_summary_op = qaccumulators[summary_ix].add_prebuilt_summary(
          stamp_token=stamp_token, summary=prebuilt_summary[summary_ix])

      with tf.control_dependencies([add_prebuilt_summary_op]):
        # Create op to update the accumulator with new input fed from
        # inputs_placeholder.

        add_summary_op = qaccumulators[summary_ix].add_prebuilt_summary(
            stamp_token=stamp_token, summary=next_summaries[summary_ix])

      with tf.control_dependencies([add_summary_op]):
        # After the flush_summary, qaccumulators will not contain any
        # uncommitted information that represents the input. Instead all the
        # digested information is returned as 'summary'. Many such summaries
        # will be combined by merge_accumulators().
        summaries.append(qaccumulators[summary_ix].flush_summary(
            stamp_token=stamp_token, next_stamp_token=stamp_token))

    return self._session.make_callable(fetches=summaries, feed_list=feed_list)

  def _make_add_summary_callable(self, qaccumulators, stamp_token):
    summary = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[self._num_features])

    add_merge_prebuilt_summary_op = [
        qaccumulators[summary_ix].add_prebuilt_summary(
            stamp_token=stamp_token, summary=summary[summary_ix])
        for summary_ix in range(self._num_features)]

    return self._session.make_callable(
        fetches=add_merge_prebuilt_summary_op,
        feed_list=[summary])

  def _make_get_buckets_callable(self, qaccumulators, stamp_token):
    def final_buckets(accumulator):
      _, buckets = accumulator.get_buckets(stamp_token=stamp_token)
      return buckets

    final_summary = tf.compat.v1.placeholder(dtype=tf.string,
                                             shape=[self._num_features])

    add_final_summary_op = [
        qaccumulators[summary_ix].add_prebuilt_summary(
            stamp_token=stamp_token, summary=final_summary[summary_ix])
        for summary_ix in range(self._num_features)
    ]

    # Create ops to flush the accumulator and return approximate boundaries.
    with tf.control_dependencies(add_final_summary_op):
      flush_op = [qacc.flush(stamp_token=stamp_token,
                             next_stamp_token=stamp_token)
                  for qacc in qaccumulators]

    with tf.control_dependencies(flush_op):
      bucket_lists = [final_buckets(qacc) for qacc in qaccumulators]

    return self._session.make_callable(
        fetches=bucket_lists, feed_list=[final_summary])

  def create_accumulator(self):
    return None

  def add_input(self, summary, next_input):
    # next_input is a list of tensors each one representing a batch for its
    # respective input.  In this case a single input should be
    # reshaped to (num_features, ?).
    flattened_input = np.reshape(next_input[0],
                                 newshape=(-1, self._num_features,))

    if summary is None:
      if flattened_input.size == 0:
        return None
      summary = [self._empty_summary] * self._num_features

    # This conversion is necessary to maintain compatibility with previous
    # versions, and will be removed in future (0.15).
    # TODO(b/127336397): Remove this.
    if isinstance(summary, (six.binary_type, six.text_type)):
      summary = [summary]

    if self._has_weights:
      flattened_weights = np.reshape(next_input[1], newshape=(1, -1))
      if flattened_input.size != flattened_weights.size * self._num_features:
        # We can only accept one dimension of weights; different size is ok.
        raise ValueError(
            'Values and weights contain incompatible sizes ({} vs {})'
            .format(flattened_input.size, flattened_weights.size))
      callable_args = [summary, flattened_input.T, flattened_weights]
    else:
      callable_args = [summary, flattened_input.T]

    return self._add_input_callable(*callable_args)

  def merge_accumulators(self, summary_sets):
    found_a_summary = False
    for summaries in summary_sets:
      if summaries is None:
        continue
      found_a_summary = True
      # TODO(b/127336397): Remove this check.
      self._merge_inputs_callable(
          [summaries] if isinstance(summaries, (six.binary_type, six.text_type))
          else summaries)

    if found_a_summary:
      return self._flush_summary_callable()
    return None

  def extract_output(self, summary):
    num_buckets = (
        self._num_quantiles - 1 if self._always_return_num_quantiles else 0)
    output_shape = tuple(self._feature_shape + [num_buckets])
    if summary is None:
      return [np.zeros(output_shape, np.float32)]
    output_shape = tuple(self._feature_shape + [-1])

    bucket_lists = self._get_buckets_callable(summary)

    def prune_buckets(buckets):  # pylint: disable=missing-docstring
      # If always_return_num_quantiles is set to True, the number of elements in
      # buckets is always equal to num_quantiles + 1. Hence we trim the min and
      # max quantile boundaries to return the internal boundaries.
      if self._always_return_num_quantiles:
        return buckets[1:-1]
      # If always_return_num_quantiles is set to False, the approximate quantile
      # library can return less or more than requested number of quantiles.
      # The max value can be same as the last internal boundary, due to removal
      # of duplicates. Below, the min and/or max quantile boundaries are trimmed
      # depending on the actual boundaries returned by the library.
      elif buckets.size >= (self._num_quantiles + 1):
        # Trim min/max.
        return buckets[1:-1]
      elif buckets.size == self._num_quantiles:
        return buckets[1:]
      # Do not trim min/max, these are part of requested boundaries.
      return buckets

    if not self._include_max_and_min:
      bucket_lists = list(map(prune_buckets, bucket_lists))

    return [np.reshape(np.stack(bucket_lists, axis=0), output_shape)]

  def output_tensor_infos(self):
    return [
        analyzer_nodes.TensorInfo(
            tf.as_dtype(self._bucket_numpy_dtype), self._output_shape, False)
    ]

  @property
  def accumulator_coder(self):
    return _QuantilesAccumulatorCacheCoder()


class _QuantilesAccumulatorCacheCoder(analyzer_nodes.CacheCoder):
  """The quantiles accumulator is a list of already encoded bytes.

  It needs to be pickled into a cacheable form.
  """

  def encode_cache(self, accumulator):
    return pickle.dumps(accumulator)

  def decode_cache(self, encoded_accumulator):
    return pickle.loads(encoded_accumulator)


def quantiles(x, num_buckets, epsilon, weights=None, reduce_instance_dims=True,
              always_return_num_quantiles=False, name=None):
  """Computes the quantile boundaries of a `Tensor` over the whole dataset.

  quantile boundaries are computed using approximate quantiles,
  and error tolerance is specified using `epsilon`. The boundaries divide the
  input tensor into approximately equal `num_buckets` parts.
  See go/squawd for details, and how to control the error due to approximation.

  Args:
    x: An input `Tensor`.
    num_buckets: Values in the `x` are divided into approximately equal-sized
      buckets, where the number of buckets is num_buckets. This is a hint. The
      actual number of buckets computed can be less or more than the requested
      number. Use the generated metadata to find the computed number of buckets.
    epsilon: Error tolerance, typically a small fraction close to zero (e.g.
      0.01). Higher values of epsilon increase the quantile approximation, and
      hence result in more unequal buckets, but could improve performance,
      and resource consumption.  Some measured results on memory consumption:
        For epsilon = 0.001, the amount of memory for each buffer to hold the
        summary for 1 trillion input values is ~25000 bytes. If epsilon is
        relaxed to 0.01, the buffer size drops to ~2000 bytes for the same input
        size. If we use a strict epsilon value of 0, the buffer size is same
        size as the input, because the intermediate stages have to remember
        every input and the quantile boundaries can be found only after an
        equivalent to a full sorting of input. The buffer size also determines
        the amount of work in the different stages of the beam pipeline, in
        general, larger epsilon results in fewer and smaller stages, and less
        time. For more performance
        trade-offs see also http://web.cs.ucla.edu/~weiwang/paper/SSDBM07_2.pdf
    weights: (Optional) Weights tensor for the quantiles. Tensor must have the
      same batch size as x.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single output vector. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    always_return_num_quantiles: (Optional) A bool that determines whether the
      exact num_buckets should be returned (defaults to False for now, but will
      be changed to True in an imminent update).
    name: (Optional) A name for this operation.

  Returns:
    The bucket boundaries represented as a list, with num_bucket-1 elements,
    unless reduce_instance_dims is False, which results in a Tensor of
    shape x.shape + [num_bucket-1].
    See code below for discussion on the type of bucket boundaries.
  """
  # TODO(b/137963802): Make always_return_num_quantiles default to True
  # TODO(b/64039847): quantile ops only support float bucket boundaries as this
  # triggers an assertion in MakeQuantileSummaries().
  # The restriction does not apply to inputs, which can be of any integral
  # dtype including tf.int32, tf.int64, tf.flost64 and tf.double.
  bucket_dtype = tf.float32
  with tf.compat.v1.name_scope(name, 'quantiles'):
    if weights is None:
      analyzer_inputs = [x]
      has_weights = False
    else:
      analyzer_inputs = [x, weights]
      has_weights = True
    combiner = QuantilesCombiner(
        num_buckets,
        epsilon,
        bucket_dtype,
        always_return_num_quantiles=(
            not reduce_instance_dims or always_return_num_quantiles),
        has_weights=has_weights,
        output_shape=(None,) if reduce_instance_dims else tuple(
            x.get_shape().as_list()[1:] + [None]),
        feature_shape=None if reduce_instance_dims else (
            x.get_shape().as_list()[1:])
        )
    (quantile_boundaries,) = _apply_cacheable_combiner(combiner,
                                                       *analyzer_inputs)
    if quantile_boundaries.get_shape().ndims < 2:
      return tf.expand_dims(quantile_boundaries, axis=0)
    return quantile_boundaries


def _quantiles_per_key(x, key, num_buckets, epsilon, name=None):
  """Like quantiles but per-key.

  For private use in tf.Transform implementation only.

  Args:
    x: An input `Tensor`.
    key: An input `Tensor` with rank 1 and size same as the fist dimension of
      `x`.  All values of `x` will be aggregated according to the corresponding
      value of `key`.
    num_buckets: See `quantiles`.
    epsilon: See `quantiles`.
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
  # TODO(b/64039847): quantile ops only support float bucket boundaries as this
  # triggers an assertion in MakeQuantileSummaries().
  # The restriction does not apply to inputs, which can be of any integral
  # dtype including tf.int32, tf.int64, tf.flost64 and tf.double.
  bucket_dtype = tf.float32
  with tf.compat.v1.name_scope(name, 'quantiles_by_key'):
    combiner = QuantilesCombiner(
        num_buckets,
        epsilon,
        bucket_dtype,
        always_return_num_quantiles=True,
        output_shape=(None,))

    input_values_node = analyzer_nodes.get_input_tensors_value_nodes((key, x))

    accumulate_outputs_value_nodes = nodes.apply_multi_output_operation(
        analyzer_nodes.CacheableCombinePerKeyAccumulate,
        input_values_node,
        combiner=combiner)

    key_value_node, bucket_boundaries = nodes.apply_multi_output_operation(
        analyzer_nodes.CacheableCombinePerKeyMerge,
        *accumulate_outputs_value_nodes,
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

  def __init__(self, numpy_dtype=np.float64, output_shape=None):
    """Store the dtype for np arrays/matrices for precision."""
    self._output_shape = output_shape
    self._numpy_dtype = numpy_dtype

  def create_accumulator(self):
    """Create an accumulator with all zero entries."""
    return None

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

    if accumulator is None:
      return [batch_cross_terms, batch_sum, batch_count]
    else:
      sum_product, sum_vectors, count = accumulator
      return [sum_product + batch_cross_terms,
              sum_vectors + batch_sum,
              count + batch_count]

  def merge_accumulators(self, accumulators):
    """Sums values in each accumulator entry."""
    accumulators = [
        accumulator for accumulator in accumulators if accumulator is not None
    ]
    if accumulators:
      # Because each accumulator contains multiple arrays of different
      # dimensions, the np.sum operation must be explicitly used across the
      # entries within each accumulator. np.sum(list(accumulators)) does not
      # work.
      sum_product = np.sum(
          [accumulator[0] for accumulator in accumulators], axis=0)
      sum_vectors = np.sum(
          [accumulator[1] for accumulator in accumulators], axis=0)
      count = np.sum([accumulator[2] for accumulator in accumulators], axis=0)
      return [sum_product, sum_vectors, count]
    else:
      return None

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
    expected_cross_terms = sum_product / count
    expected_terms = sum_vectors / count

    return [expected_cross_terms - np.outer(expected_terms, expected_terms)]

  def output_tensor_infos(self):
    return [
        analyzer_nodes.TensorInfo(
            tf.as_dtype(self._numpy_dtype), self._output_shape, False)
    ]


def covariance(x, dtype, name=None):
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
        CovarianceCombiner(dtype.as_numpy_dtype, shape), x)
    return result


class PCACombiner(CovarianceCombiner):
  """Compute PCA of accumulated data using the biased covariance matrix."""

  def __init__(self, output_dim=None, numpy_dtype=np.float64,
               output_shape=None):
    """Store pca output dimension, and dtype for precision."""
    super(PCACombiner, self).__init__(
        numpy_dtype=numpy_dtype, output_shape=output_shape)
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
    expected_cross_terms = sum_product / count
    expected_terms = sum_vectors / count
    cov = expected_cross_terms - np.outer(expected_terms, expected_terms)
    vals, vecs = np.linalg.eigh(cov)
    sorted_vecs = vecs[:, np.argsort(vals)[::-1]]
    if self._output_dim is None:
      return [sorted_vecs]
    else:
      return [sorted_vecs[:, :self._output_dim]]


def pca(x, output_dim, dtype, name=None):
  """Computes pca on the dataset using biased covariance.

  The pca analyzer computes output_dim orthonormal vectors that capture
  directions/axes corresponding to the highest variances in the input vectors of
  x. The output vectors are returned as a rank-2 tensor with shape
  (input_dim, output_dim), where the 0th dimension are the components of each
  output vector, and the 1st dimension are the output vectors representing
  orthogonal directions in the input space, sorted in order of decreasing
  variances.

  The output rank-2 tensor (matrix) serves a useful transform purpose. Formally,
  the matrix can be used downstream in the transform step by multiplying it to
  the input tensor x. This transform reduces the dimension of input vectors to
  output_dim in a way that retains the maximal variance.

  NOTE: To properly use PCA, input vector components should be converted to
  similar units of measurement such that the vectors represent a Euclidean
  space. If no such conversion is available (e.g. one element represents time,
  another element distance), the canonical approach is to first apply a
  transformation to the input data to normalize numerical variances, i.e.
  tft.scale_to_z_score(). Normalization allows PCA to choose output axes that
  help decorrelate input axes.

  Below are a couple intuitive examples of PCA.

  Consider a simple 2-dimensional example:

  Input x is a series of vectors [e, e] where e is Gaussian with mean 0,
  variance 1. The two components are perfectly correlated, and the resulting
  covariance matrix is
  [[1 1],
   [1 1]].
  Applying PCA with output_dim = 1 would discover the first principal component
  [1 / sqrt(2), 1 / sqrt(2)]. When multipled to the original example, each
  vector [e, e] would be mapped to a scalar sqrt(2) * e. The second principal
  component would be [-1 / sqrt(2), 1 / sqrt(2)] and would map [e, e] to 0,
  which indicates that the second component captures no variance at all. This
  agrees with our intuition since we know that the two axes in the input are
  perfectly correlated and can be fully explained by a single scalar e.

  Consider a 3-dimensional example:

  Input x is a series of vectors [a, a, b], where a is a zero-mean, unit
  variance Gaussian. b is a zero-mean, variance 4 Gaussian and is independent of
  a. The first principal component of the unnormalized vector would be [0, 0, 1]
  since b has a much larger variance than any linear combination of the first
  two components. This would map [a, a, b] onto b, asserting that the axis with
  highest energy is the third component. While this may be the desired
  output if a and b correspond to the same units, it is not statistically
  desireable when the units are irreconciliable. In such a case, one should
  first normalize each component to unit variance first, i.e. b := b / 2.
  The first principal component of a normalized vector would yield
  [1 / sqrt(2), 1 / sqrt(2), 0], and would map [a, a, b] to sqrt(2) * a. The
  second component would be [0, 0, 1] and map [a, a, b] to b. As can be seen,
  the benefit of normalization is that PCA would capture highly correlated
  components first and collapse them into a lower dimension.

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
        PCACombiner(output_dim, dtype.as_numpy_dtype, shape), x)
    return result


def ptransform_analyzer(inputs, output_dtypes, output_shapes, ptransform,
                        name=None):
  """Applies a user-provided PTransform over the whole dataset.

  WARNING: This is experimental.

  Note that in order to have asset files copied correctly, any outputs that
  represent asset filenames must be added to the `tf.GraphKeys.ASSET_FILEPATHS`
  collection by the caller.

  Args:
    inputs: A list of input `Tensor`s.
    output_dtypes: The list of TensorFlow dtypes of the output of the analyzer.
    output_shapes: The list of shapes of the output of the analyzer.  Must have
      the same length as output_dtypes.
    ptransform: A Beam PTransform that accepts a Beam PCollection where each
      element is a list of `ndarray`s.  Each element in the list contains a
      batch of values for the corresponding input tensor of the analyzer.  It
      returns a tuple of `PCollection`, each containing a single element which
      is an `ndarray`.
    name: (Optional) Similar to a TF op name.  Used to define a unique scope for
      this analyzer, which can be used for debugging info.

  Returns:
    A list of output `Tensor`s.  These will have `dtype` and `shape` as
      specified by `output_dtypes` and `output_shapes`.

  Raises:
    ValueError: If output_dtypes and output_shapes have different lengths.
  """
  if len(output_dtypes) != len(output_shapes):
    raise ValueError('output_dtypes ({}) and output_shapes ({}) had different'
                     ' lengths'.format(output_dtypes, output_shapes))
  with tf.compat.v1.name_scope(name, 'ptransform'):
    output_tensor_infos = [
        analyzer_nodes.TensorInfo(dtype, shape, False)
        for dtype, shape in zip(output_dtypes, output_shapes)
    ]
    return apply_analyzer(
        analyzer_nodes.PTransform,
        *inputs,
        ptransform=ptransform,
        output_tensor_info_list=output_tensor_infos)
