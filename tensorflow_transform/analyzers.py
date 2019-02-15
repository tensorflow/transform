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
import re

# GOOGLE-INITIALIZATION
import numpy as np
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import nodes
from tensorflow_transform import tf_utils

from tensorflow.contrib.boosted_trees.python.ops import quantile_ops
from tensorflow.python.ops import resources
from tensorflow.python.util import deprecation


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
  # the same shape as the input, but without the batch.  If reducing over batch
  # dimensions, with unknown shape, the result will also have unknown shape.
  return x.shape.as_list()[1:] if x.shape.dims is not None else None


# TODO(b/112414577): Go back to accepting only a single input.
# Currently we accept multiple inputs so that we can implement min and max
# with a single combiner.
def _numeric_combine(inputs,
                     fn,
                     reduce_instance_dims=True,
                     output_dtypes=None):
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
  return _apply_cacheable_combiner(combiner, *inputs)


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
  with tf.name_scope(name, 'min'):
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
  with tf.name_scope(name, 'max'):
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
  with tf.name_scope(name, 'min_and_max'):
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
  with tf.name_scope(name, 'sum'):
    if reduce_instance_dims:
      if isinstance(x, tf.SparseTensor):
        x = x.values
      x = tf.reduce_sum(x)
    elif isinstance(x, tf.SparseTensor):
      if x.dtype == tf.uint8 or x.dtype == tf.uint16:
        x = tf.cast(x, tf.int64)
      elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
        TypeError('Data type %r is not supported' % x.dtype)
      x = tf.sparse_reduce_sum(x, axis=0)
    else:
      x = tf.reduce_sum(x, axis=0)
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
  with tf.name_scope(name, 'size'):
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
  with tf.name_scope(name, 'mean'):
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
  with tf.name_scope(name, 'var'):
    return _mean_and_var(x, reduce_instance_dims, output_dtype)[1]


def _mean_and_var(x, reduce_instance_dims=True, output_dtype=None):
  """More efficient combined `mean` and `var`.  See `var`."""
  if output_dtype is None:
    output_dtype = _MEAN_OUTPUT_DTYPE_MAP.get(x.dtype)
    if output_dtype is None:
      raise TypeError('Tensor type %r is not supported' % x.dtype)

  with tf.name_scope('mean_and_var'):

    x = tf.cast(x, output_dtype)

    x_count, x_mean, x_variance = (
        tf_utils.reduce_batch_count_mean_and_var(x, reduce_instance_dims))

    combine_inputs = _MeanAndVarAccumulator(
        count=x_count, mean=x_mean, variance=x_variance)

    output_shape = ()
    if not reduce_instance_dims:
      # We need to use tf.expand_dims to artificially add a batch dimension.
      output_shape = _get_output_shape_from_input(
          tf.expand_dims(x_count, axis=0))

    x_mean, x_var = _apply_cacheable_combiner(
        MeanAndVarCombiner(output_dtype.as_numpy_dtype, output_shape),
        *combine_inputs)

  return x_mean, x_var


class _MeanAndVarAccumulator(
    collections.namedtuple('MeanAndVarAccumulator',
                           ['count', 'mean', 'variance'])):
  """Container for MeanAndVarCombiner intermediate values."""

  @classmethod
  def make_nan_to_num(cls, counts, means, variances):
    return cls(counts, np.nan_to_num(means), np.nan_to_num(variances))

  def __reduce__(self):
    return self.__class__, tuple(self)


class MeanAndVarCombiner(analyzer_nodes.Combiner):
  """Combines a PCollection of accumulators to compute mean and variance."""

  def __init__(self, output_numpy_dtype, output_shape=None):
    self._output_numpy_dtype = output_numpy_dtype
    self._output_shape = output_shape

  def create_accumulator(self):
    """Create an accumulator with all zero entries."""
    return _MeanAndVarAccumulator(0, 0., 0.)

  def add_input(self, accumulator, batch_values):
    """Composes an accumulator from batch_values and calls merge_accumulators.

    Args:
      accumulator: The `_MeanAndVarAccumulator` computed so far.
      batch_values: A `_MeanAndVarAccumulator` for the current batch.

    Returns:
      A `_MeanAndVarAccumulator` which is accumulator and batch_values combined.
    """
    new_accumulator = _MeanAndVarAccumulator(*batch_values)
    return self._combine_mean_and_var_accumulators(accumulator, new_accumulator)

  def merge_accumulators(self, accumulators):
    """Merges several `_MeanAndVarAccumulator`s to a single accumulator.

    Args:
      accumulators: A list of `_MeanAndVarAccumulator`s and/or Nones.

    Returns:
      The sole merged `_MeanAndVarAccumulator`.
    """
    non_empty_accumulators = [
        accumulator for accumulator in accumulators if accumulator is not None
    ]
    if not non_empty_accumulators:
      return self.create_accumulator()

    result = non_empty_accumulators[0]

    for accumulator in non_empty_accumulators[1:]:
      result = self._combine_mean_and_var_accumulators(result, accumulator)

    return result

  def extract_output(self, accumulator):
    """Converts an accumulator into the output (mean, var) tuple.

    Args:
      accumulator: the final `_MeanAndVarAccumulator` value.

    Returns:
      A 2-tuple composed of (mean, var) or None if accumulator is None.
    """
    if accumulator is None:
      return None
    else:
      return (self._output_numpy_dtype(accumulator.mean),
              self._output_numpy_dtype(accumulator.variance))

  def output_tensor_infos(self):
    # The output is (mean, var).
    return [
        analyzer_nodes.TensorInfo(self._output_numpy_dtype, self._output_shape,
                                  False)
    ] * 2

  def _combine_mean_and_var_accumulators(self, a, b):
    """Combines two mean and var accumulators.

    Args:
      a: A _MeanAndVarAccumulator.
      b: A _MeanAndVarAccumulator.

    Returns:
      A _MeanAndVarAccumulator computed as the combination of a and b.
    """
    # NaNs get preserved through division by a.count + b.count.
    a = _MeanAndVarAccumulator.make_nan_to_num(*a)
    b = _MeanAndVarAccumulator.make_nan_to_num(*b)

    # a.count >= b.count following this logic.
    if np.sum(a.count) < np.sum(b.count):
      a, b = b, a

    if np.sum(a.count) == 0:
      return b

    combined_total = a.count + b.count

    # Mean and variance update formulas which are more numerically stable when
    # a and b vary in magnitude.
    combined_mean = a.mean + (b.count / combined_total) * (b.mean - a.mean)

    combined_variance = (
        a.variance + (b.count / combined_total) * (b.variance + (
            (b.mean - combined_mean) * (b.mean - a.mean)) - a.variance))

    return _MeanAndVarAccumulator(combined_total, combined_mean,
                                  combined_variance)


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
    filename = prefix + tf.get_default_graph().get_name_scope()
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


def _get_top_k_and_frequency_threshold(top_k, frequency_threshold):
  """Validate `top_k` and `frequency_threshold` values and convert to int."""
  if top_k is not None:
    top_k = int(top_k)
    if top_k < 0:
      raise ValueError('top_k must be non-negative, but got: %r' % top_k)

  if frequency_threshold is not None:
    frequency_threshold = int(frequency_threshold)
    if frequency_threshold < 0:
      raise ValueError(
          'frequency_threshold must be non-negative, but got: %r' %
          frequency_threshold)
    elif frequency_threshold <= 1:
      tf.logging.warn(
          'frequency_threshold %d <= 1 is a no-op, use None instead.',
          frequency_threshold)
  return top_k, frequency_threshold


# TODO(KesterTong): Once multiple outputs are supported, return indices too.
# TODO(b/117796748): Add coverage key feature input as alternative to `key_fn`.
# TODO(b/116308354): rename store_frequency to store_importance because it now
# can return mutual information.
def vocabulary(
    x,
    top_k=None,
    frequency_threshold=None,
    vocab_filename=None,
    store_frequency=False,
    weights=None,
    labels=None,
    use_adjusted_mutual_info=False,
    min_diff_from_avg=0.0,
    coverage_top_k=None,
    coverage_frequency_threshold=None,
    key_fn=None,
    name=None):
  r"""Computes the unique values of a `Tensor` over the whole dataset.

  Computes The unique values taken by `x`, which can be a `Tensor` or
  `SparseTensor` of any size.  The unique values will be aggregated over all
  dimensions of `x` and all instances.

  In case one of the tokens contains the '\n' or '\r' characters or is empty it
  will be discarded since we are currently writing the vocabularies as text
  files. This behavior will likely be fixed/improved in the future.

  The unique values are sorted by decreasing frequency and then reverse
  lexicographical order (e.g. [('a', 5), ('c', 3), ('b', 3)]).

  For large datasets it is highly recommended to either set frequency_threshold
  or top_k to control the size of the output, and also the run time of this
  operation.

  When labels are provided, we filter the vocabulary based on how correlated the
  unique value is with a positive label (Mutual Information).


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
    x: An input `Tensor` or `SparseTensor` with dtype tf.string.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      absolute frequency is >= to the supplied threshold. If set to None, the
      full vocabulary is generated.  Absolute frequency means the number of
      occurences of the element in the dataset, as opposed to the proportion of
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
      will be of the form 'frequency word'.
    weights: (Optional) Weights `Tensor` for the vocabulary. It must have the
      same shape as x.
    labels: (Optional) Labels `Tensor` for the vocabulary. It must have dtype
      int64, have values 0 or 1, and have the same shape as x.
    use_adjusted_mutual_info: If true, use adjusted mutual information.
    min_diff_from_avg: Mutual information of a feature will be adjusted to zero
      whenever the difference between count of the feature with any label and
      its expected count is lower than min_diff_from_average.
    coverage_top_k: (Optional), (Experimental) The minimum number of elements
      per key to be included in the vocabulary.
    coverage_frequency_threshold: (Optional), (Experimental) Limit the coverage
      arm of the vocabulary only to elements whose absolute frequency is >= this
      threshold for a given key.
    key_fn: (Optional), (Experimental) A fn that takes in a single entry of `x`
      and returns the corresponding key for coverage calculation. If this is
      `None`, no coverage arm is added to the vocabulary.
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

  if isinstance(x, tf.SparseTensor):
    x = x.values

  if x.dtype != tf.string:
    raise ValueError('expected tf.string but got %r' % x.dtype)

  with tf.name_scope(name, 'vocabulary'):
    vocab_filename = _get_vocab_filename(vocab_filename, store_frequency)

    if labels is not None:
      vocab_ordering_type = (
          tf_utils.VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION)
      (unique_inputs, sum_total,
       sum_positive, counts) = tf_utils.reduce_batch_vocabulary(
           x, vocab_ordering_type, weights, labels)
      analyzer_inputs = [unique_inputs, sum_total, sum_positive, counts]

    elif weights is not None:
      vocab_ordering_type = tf_utils.VocabOrderingType.WEIGHTED_FREQUENCY
      (unique_inputs, sum_weights,
       none_sum, none_counts) = tf_utils.reduce_batch_vocabulary(
           x, vocab_ordering_type, weights)
      assert none_sum is None
      assert none_counts is None
      analyzer_inputs = [unique_inputs, sum_weights]
    else:
      vocab_ordering_type = tf_utils.VocabOrderingType.FREQUENCY
      (unique_inputs, none_weights, none_sum,
       none_counts) = tf_utils.reduce_batch_vocabulary(x, vocab_ordering_type)
      assert none_sum is None
      assert none_weights is None
      assert none_counts is None
      analyzer_inputs = [unique_inputs]

    input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
        analyzer_inputs)

    accumulate_output_value_node = nodes.apply_operation(
        analyzer_nodes.VocabularyAccumulate, input_values_node,
        vocab_ordering_type=vocab_ordering_type)

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
        store_frequency=store_frequency)

    vocab_filename = analyzer_nodes.wrap_as_tensor(vocab_filename_node)
    return vocab_filename


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
               include_max_and_min=False):
    self._num_quantiles = num_quantiles
    self._epsilon = epsilon
    self._bucket_numpy_dtype = bucket_numpy_dtype
    self._always_return_num_quantiles = always_return_num_quantiles
    self._has_weights = has_weights
    self._output_shape = output_shape
    self._include_max_and_min = include_max_and_min

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

      # stamp_token is used to commit the state of the qaccumulator. In
      # this case, the qaccumulator state is completely returned and stored
      # as part of quantile_state/summary in the combiner fn (i.e the summary is
      # extracted and stored outside the qaccumulator). So we don't use
      # the timestamp mechanism to signify progress in the qaccumulator state.
      stamp_token = 0

      self._session = tf.Session(graph=graph)

      qaccumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=stamp_token,
          num_quantiles=self._num_quantiles,
          epsilon=self._epsilon,
          name='qaccumulator',
          generate_quantiles=self._always_return_num_quantiles)
      self._session.run(
          resources.initialize_resources(resources.shared_resources()))

      self._add_input_callable = self._make_add_input_callable(
          qaccumulator, stamp_token)
      self._merge_inputs_callable = self._make_add_summary_callable(
          qaccumulator, stamp_token)
      self._get_buckets_callable = self._make_get_buckets_callable(
          qaccumulator, stamp_token)

      # Create op to flush summaries and return a summary representing the
      # summaries that were added the accumulator so far.
      self._flush_summary_callable = self._session.make_callable(
          fetches=qaccumulator.flush_summary(
              stamp_token=stamp_token, next_stamp_token=stamp_token))

      graph.finalize()

    # We generate an empty summary by calling self._flush_summary_op.
    # We cache this as some implementations may call create_accumulator for
    # every input, and it can be cached since it will always be the same and
    # immutable.
    self._empty_summary = self._flush_summary_callable()

  def __reduce__(self):
    return QuantilesCombiner, (self._num_quantiles, self._epsilon,
                               self._bucket_numpy_dtype,
                               self._always_return_num_quantiles,
                               self._has_weights,
                               self._output_shape,
                               self._include_max_and_min)

  def _make_add_input_callable(self, qaccumulator, stamp_token):
    # Create placeholders for add_inputs_callable.  These placeholders will
    # be used to provide prebuilt summary, input and weights to the
    # QuantileAccumulator.
    # inputs and weights need to have shapes (1, None) as this is what the
    # QuantileAccumulator accepts.
    prebuilt_summary = tf.placeholder(dtype=tf.string, shape=[])
    inputs = tf.placeholder(dtype=self._bucket_numpy_dtype, shape=[1, None])
    feed_list = [prebuilt_summary, inputs]
    if self._has_weights:
      weights = tf.placeholder(dtype=tf.float32, shape=[1, None])
      feed_list.append(weights)
    else:
      weights = tf.ones_like(inputs)

    # TODO(b/68277922): Investigate add_inputs() to efficiently handle multiple
    # batches of inputs.
    add_prebuilt_summary_op = qaccumulator.add_prebuilt_summary(
        stamp_token=stamp_token, summary=prebuilt_summary)

    with tf.control_dependencies([add_prebuilt_summary_op]):
      # Create op to update the accumulator with new input fed from
      # inputs_placeholder.
      add_summary_op = qaccumulator.add_summary(
          stamp_token=stamp_token, column=inputs, example_weights=weights)

    with tf.control_dependencies([add_summary_op]):
      # After the flush_summary, qaccumulator will not contain any
      # uncommitted information that represents the input. Instead all the
      # digested information is returned as 'summary'. Many such summaries
      # will be combined by merge_accumulators().
      summary = qaccumulator.flush_summary(
          stamp_token=stamp_token, next_stamp_token=stamp_token)

    return self._session.make_callable(fetches=summary, feed_list=feed_list)

  def _make_add_summary_callable(self, qaccumulator, stamp_token):
    merge_prebuilt_summary = tf.placeholder(dtype=tf.string, shape=[])

    add_merge_prebuilt_summary_op = qaccumulator.add_prebuilt_summary(
        stamp_token=stamp_token, summary=merge_prebuilt_summary)

    return self._session.make_callable(
        fetches=add_merge_prebuilt_summary_op,
        feed_list=[merge_prebuilt_summary])

  def _make_get_buckets_callable(self, qaccumulator, stamp_token):
    final_summary = tf.placeholder(dtype=tf.string, shape=[])

    add_final_summary_op = qaccumulator.add_prebuilt_summary(
        stamp_token=stamp_token, summary=final_summary)

    # Create ops to flush the accumulator and return approximate boundaries.
    with tf.control_dependencies([add_final_summary_op]):
      flush_op = qaccumulator.flush(
          stamp_token=stamp_token, next_stamp_token=stamp_token)

    with tf.control_dependencies([flush_op]):
      _, buckets = qaccumulator.get_buckets(stamp_token=stamp_token)

    return self._session.make_callable(
        fetches=buckets, feed_list=[final_summary])

  def create_accumulator(self):
    return None

  def add_input(self, summary, next_input):
    # next_input is a list of tensors each one representing a batch for its
    # respective input.  In this case we have a single input, which we reshape
    # to (1,?).
    flattened_input = np.reshape(next_input[0], newshape=(1, -1))

    if summary is None and flattened_input.size == 0:
      return None

    callable_args = [summary or self._empty_summary, flattened_input]
    if self._has_weights:
      flattened_weights = np.reshape(next_input[1], newshape=(1, -1))
      if len(flattened_input) != len(flattened_weights):
        raise ValueError(
            'Values and weights contained different number of values ({} vs {})'
            .format(len(flattened_input), len(flattened_weights)))
      callable_args.append(flattened_weights)

    return self._add_input_callable(*callable_args)

  def merge_accumulators(self, summaries):
    found_a_summary = False
    for summary in summaries:
      if summary is not None:
        found_a_summary = True
        self._merge_inputs_callable(summary)

    if found_a_summary:
      return self._flush_summary_callable()
    else:
      return None

  def extract_output(self, summary):
    if summary is None:
      num_buckets = (
          self._num_quantiles - 1 if self._always_return_num_quantiles else 0)
      return [np.zeros((num_buckets,), np.float32)]

    buckets = self._get_buckets_callable(summary)

    if not self._include_max_and_min:
      # If always_return_num_quantiles is set to True, the number of elements in
      # buckets is always equal to num_quantiles + 1. Hence we trim the min and
      # max quantile boundaries to return the internal boundaries.
      if self._always_return_num_quantiles:
        buckets = buckets[1:-1]
      # If always_return_num_quantiles is set to False, the approximate quantile
      # library can return less or more than requested number of quantiles.
      # The max value can be same as the last internal boundary, due to removal
      # of duplicates. Below, the min and/or max quantile boundaries are trimmed
      # depending on the actual boundaries returned by the library.
      elif buckets.size >= (self._num_quantiles + 1):
        # Trim min/max.
        buckets = buckets[1:-1]
      elif buckets.size == self._num_quantiles:
        buckets = buckets[1:]
      else:
        # Do not trim min/max, these are part of requested boundaries.
        pass

    return [buckets]

  def output_tensor_infos(self):
    return [
        analyzer_nodes.TensorInfo(
            tf.as_dtype(self._bucket_numpy_dtype), self._output_shape, False)
    ]

  @property
  def accumulator_coder(self):
    return _QuantilesAccumulatorCacheCoder()


class _QuantilesAccumulatorCacheCoder(analyzer_nodes.CacheCoder):
  """The quantiles accumulator is already encoded."""

  def encode_cache(self, accumulator):
    return accumulator

  def decode_cache(self, encoded_accumulator):
    return encoded_accumulator


def quantiles(x, num_buckets, epsilon, weights=None, name=None):
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
      same shape as x.
    name: (Optional) A name for this operation.

  Returns:
    The bucket boundaries represented as a list, with num_bucket-1 elements
    See code below for discussion on the type of bucket boundaries.
  """
  # TODO(b/64039847): quantile ops only support float bucket boundaries as this
  # triggers an assertion in MakeQuantileSummaries().
  # The restriction does not apply to inputs, which can be of any integral
  # dtype including tf.int32, tf.int64, tf.flost64 and tf.double.
  bucket_dtype = tf.float32
  with tf.name_scope(name, 'quantiles'):
    if weights is None:
      analyzer_inputs = [x]
      has_weights = False
      always_return_num_quantiles = False
    else:
      x = tf_utils.assert_same_shape(x, weights)
      analyzer_inputs = [x, weights]
      has_weights = True
      always_return_num_quantiles = True
    combiner = QuantilesCombiner(
        num_buckets,
        epsilon,
        bucket_dtype.as_numpy_dtype,
        always_return_num_quantiles=always_return_num_quantiles,
        has_weights=has_weights,
        output_shape=(None,))
    (quantile_boundaries,) = _apply_cacheable_combiner(combiner,
                                                       *analyzer_inputs)
    return tf.expand_dims(quantile_boundaries, axis=0)


def _quantiles_per_key(x, key, num_buckets, epsilon, name=None):
  """Like quantiles but per-key.

  For private use in tf.Transform implemenation only.

  Args:
    x: An input `Tensor`.
    key: An input `Tensor` with rank 1 and size same as the fist dimension of
      `x`.  All values of `x` will be aggregated according to the corresponding
      value of `key`.
    num_buckets: See `quantiles`.
    epsilon: See `quantiles`.
    name: (Optional) A name for this operation.

  Returns:
    A pair (key_vocab, quantiles) where `key_vocab` is a sorted vocabulary of
    all elements in the input `key` and `quantiles` is a rank 2 tensor
    containing quantile boundaries for each key, where boundaries are for the
    corresponding element of `key_vocab`.

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
  with tf.name_scope(name, 'quantiles_by_key'):
    combiner = QuantilesCombiner(
        num_buckets,
        epsilon,
        bucket_dtype.as_numpy_dtype,
        always_return_num_quantiles=True,
        output_shape=(None,))
    key, bucket_boundaries = _apply_cacheable_combiner_per_key(combiner, key, x)
    return key, bucket_boundaries


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

  with tf.name_scope(name, 'covariance'):
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

  with tf.name_scope(name, 'pca'):
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
    output_dtypes: The list of dtypes of the output of the analyzer.
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
  with tf.name_scope(name, 'ptransform'):
    output_tensor_infos = [
        analyzer_nodes.TensorInfo(dtype, shape, False)
        for dtype, shape in zip(output_dtypes, output_shapes)
    ]
    return apply_analyzer(
        analyzer_nodes.PTransform,
        *inputs,
        ptransform=ptransform,
        output_tensor_info_list=output_tensor_infos)
