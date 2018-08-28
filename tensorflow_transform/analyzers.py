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

import numpy as np
import tensorflow as tf

from tensorflow_transform import tf_utils
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops

from tensorflow.python.ops import resources
from tensorflow.python.util import deprecation


ANALYZER_COLLECTION = 'tft_analyzers'
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


# NOTE: this code is designed so that Analyzer is pickleable, and in particular
# does not try to pickle a tf.Graph or tf.Tensor which may not be pickleable.
# This is due to https://issues.apache.org/jira/browse/BEAM-3812.  Until that
# issue is fixed, anything that is a member variable of a Beam PTransform may
# end up getting pickled.  Instances of Analyzer do end up as member variables
# of a PTransform in our implementation of tf.Transform on Beam currently, so
# we must avoid directly putting `Tensor`s inside `Analyzer`, and instead use
# tensor names.
#
# Due to these pickling issues and also logical separation of TensorFlow and
# numpy code, the spec should also not contain TensorFlow dtypes but rather
# their numpy equivalent.
class Analyzer(object):
  """A class representing computation that will be done by Beam.

  An Analyzer is like a tf.Operation except that it requires computation over
  the full dataset.  E.g. sum(my_tensor) will compute the sum of the value of
  my_tensor over all instances in the dataset.  The Analyzer class contains the
  inputs to this computation, and placeholders which will later be converted to
  constants during a call to AnalyzeDataset.

  Analyzer implementations write some files to disk in a temporary location and
  return tensors that contain the filename.  These outputs must be added to the
  tf.GraphKeys.ASSET_FILEPATHS collection.  Doing so will ensure a few things
  happen:
  * the tensor will be removed from the collection prior to writing the
    SavedModel (since the tensor will be replaced)
  * when the tensor is replaced, the replacement will be added to the
    tf.GraphKeys.ASSET_FILEPATHS collection
  * This in turn causes the underlying file to be added to the SavedModel's
    `assets` directory when the model is saved

  Args:
    inputs: The `Tensor`s that are used to create inputs to this analyzer,
    outputs: The `Tensor`s whose values will be replaced by the result of the
        analyzer.
    spec: An object that will be used to determine how the analyzer is
        implemented by Beam.
    name: The name of this analyzer, typically a TensorFlow scope.
  """

  def __init__(self, inputs, outputs, spec, name):
    for index, tensor in enumerate(inputs):
      if not isinstance(tensor, tf.Tensor):
        raise ValueError(
            'In analyzer {}, the {}th input ({}) was not a Tensor'.format(
                name, index, tensor))
    for index, tensor in enumerate(outputs):
      if not isinstance(tensor, tf.Tensor):
        raise ValueError(
            'In analyzer {}, the {}th output ({}) was not a Tensor'.format(
                name, index, tensor))
    self._inputs = inputs
    self._outputs = outputs
    self._spec = spec
    self._name = name

  @property
  def spec(self):
    return self._spec

  @property
  def name(self):
    return self._name

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def control_inputs(self):
    return []


class CombinerSpec(object):
  """Analyze using combiner function.

  This object mirrors a beam.CombineFn, that will receive a beam PCollection
  representing the batched input tensors.
  """

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

  def extract_output(self, accumulator):
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: the final accumulator value.

    Returns: A list of ndarrays representing the result of this combiner.
    """
    raise NotImplementedError

  def num_outputs(self):
    """Return the number of outputs that are produced by extract_output.

    Returns: The number of outputs extract_output will produce.
    """
    raise NotImplementedError


class _CombinePerKeySpec(object):
  """A wrapper for per-key combining.

  For private use in tf.Transform implemenation only.

  All outputs returned by extract_output must be the same shape across keys so
  they can be stacked.

  Args:
    combiner_spec: A `CombinerSpec` that will be used to reduce values for each
        key.
  """

  def __init__(self, combiner_spec):
    self._combiner_spec = combiner_spec

  @property
  def combiner_spec(self):
    return self._combiner_spec


def combine_analyzer(inputs, output_dtypes, output_shapes, combiner_spec, name):
  """Applies the combiner over the whole dataset.

  Args:
    inputs: A list of input `Tensor`s or `SparseTensor`s.
    output_dtypes: The list of dtypes of the output of the analyzer.
    output_shapes: The list of shapes of the output of the analyzer.  Must have
      the same length as output_dtypes.
    combiner_spec: A subclass of CombinerSpec.
    name: Similar to a TF op name.  Used to define a unique scope for this
      analyzer, which can be used for debugging info.

  Returns:
    A list of `Tensor`s representing the combined values.  These will have
        `dtype` and `shape` given by `output_dtypes` and `output_shapes`.  These
        dtypes and shapes must be compatible with the combiner_spec.

  Raises:
    ValueError: If output_dtypes and output_shapes have different lengths.
  """
  if len(output_dtypes) != len(output_shapes):
    raise ValueError('output_dtypes ({}) and output_shapes ({}) had different'
                     ' lengths'.format(output_dtypes, output_shapes))
  with tf.name_scope(name) as scope:
    outputs = [tf.placeholder(dtype, shape)
               for dtype, shape in zip(output_dtypes, output_shapes)]
    tf.add_to_collection(
        ANALYZER_COLLECTION, Analyzer(inputs, outputs, combiner_spec, scope))
    return outputs


class _NumPyCombinerSpec(CombinerSpec):
  """Combines the PCollection only on the 0th dimension using nparray.

  Args:
    fn: The numpy function representing the reduction to be done.
    reduce_instance_dims: Whether to reduce across non-batch dimensions.
    output_dtypes: The numpy dtype to cast each output to.
  """

  def __init__(self, fn, output_dtypes):
    self._fn = fn
    self._output_dtypes = output_dtypes

  def create_accumulator(self):
    return None

  def add_input(self, accumulator, batch_values):
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

  def num_outputs(self):
    return len(self._output_dtypes)


def _get_output_shape_from_input(x):
  if isinstance(x, tf.SparseTensor):
    return x.get_shape()[1:]

  # When reducing over batch dimensions, with known shape, the result will be
  # the same shape as the input, but without the batch.  If reducing over batch
  # dimensions, with unknown shape, the result will also have unknown shape.
  return x.shape.as_list()[1:] if x.shape.dims is not None else None


def _numeric_combine(inputs,
                     fn,
                     reduce_instance_dims=True,
                     name=None,
                     output_dtypes=None):
  """Apply a reduction, defined by a numpy function to multiple inputs.

  Args:
    inputs: A list of tensors, which will be independently reduced.
    fn: A function to reduce tensors across instances/batches, to get a single
        output.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.
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
    shapes = [() for _ in inputs]
  else:
    # Reducing over batch dimensions.
    shapes = [x.get_shape() for x in inputs]
  spec = _NumPyCombinerSpec(fn,
                            [dtype.as_numpy_dtype for dtype in output_dtypes])
  return combine_analyzer(inputs, output_dtypes, shapes, spec, name
                          if name is not None else fn.__name__)


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
  return _min_and_max(x, reduce_instance_dims, name)[1]


def _sparse_minus_reduce_min_and_reduce_max(x):
  """Computes the -min and max of a tensor x.

  It differs from sparse_reduce_max in that sparse_reduce_max returns 0 when all
  elements are missing along axis 0.
  We replace the 0 with NaN when x's dtype is float and dtype.min+1 when it's
  int.

  Args:
    x: A `SparseTensor`.

  Returns:
    Two `Tensors' which are the -min and max.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  if not isinstance(x, tf.SparseTensor):
    raise TypeError('Expected a SparseTensor, but got %r' % x)
  minus_x = tf.SparseTensor(
      indices=x.indices, values=0 - x.values, dense_shape=x.dense_shape)
  x_count = tf_utils.reduce_batch_count(x, reduce_instance_dims=False)
  batch_has_no_values = tf.equal(x_count, tf.constant(0, dtype=tf.int64))
  x_batch_max = tf.sparse_reduce_max(x, axis=0)
  x_batch_minus_min = tf.sparse_reduce_max(minus_x, axis=0)

  if x.dtype.is_floating:
    missing_value = tf.constant(np.nan, x.dtype)
  else:
    missing_value = tf.constant(x.dtype.min + 1, x.dtype)

  x_batch_max = tf.where(batch_has_no_values,
                         tf.fill(tf.shape(x_batch_max), missing_value),
                         x_batch_max)
  x_batch_minus_min = tf.where(batch_has_no_values,
                               tf.fill(tf.shape(x_batch_minus_min),
                                       missing_value),
                               x_batch_minus_min)
  return x_batch_minus_min, x_batch_max


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
    output_dtype = x.dtype

    if x.dtype == tf.uint8 or x.dtype == tf.uint16:
      x = tf.cast(x, tf.int32)

    elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
      raise TypeError('Tensor type %r is not supported' % x.dtype)

    if reduce_instance_dims:
      if isinstance(x, tf.SparseTensor):
        x = x.values

      x_batch_max = tf.reduce_max(x)
      x_batch_minus_min = tf.reduce_max(0 - x)
    elif isinstance(x, tf.SparseTensor):
      x_batch_minus_min, x_batch_max = (
          _sparse_minus_reduce_min_and_reduce_max(x))
      if x.dtype.is_floating:
        combine_fn = np.nanmax
    else:
      x_batch_max = tf.reduce_max(x, axis=0)
      x_batch_minus_min = tf.reduce_max(0 - x, axis=0)

    def inf_to_nan(tensor):
      if tensor.dtype.is_floating:
        nan = tf.constant(np.nan, output_dtype)
        return tf.where(tf.is_inf(tensor), tensor + nan, tensor)
      return tensor

    minus_x_min, x_max = _numeric_combine(  # pylint: disable=unbalanced-tuple-unpacking
        [inf_to_nan(x_batch_minus_min),
         inf_to_nan(x_batch_max)], combine_fn, reduce_instance_dims)
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
  return _numeric_combine([x], sum_fn, reduce_instance_dims, name,
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
    return _mean_and_var(x, reduce_instance_dims, name, output_dtype)[0]


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
    return _mean_and_var(x, reduce_instance_dims, name, output_dtype)[1]


def _mean_and_var(x, reduce_instance_dims=True, name=None, output_dtype=None):
  """More efficient combined `mean` and `var`.  See `var`."""
  if output_dtype is None:
    output_dtype = _MEAN_OUTPUT_DTYPE_MAP.get(x.dtype)
    if output_dtype is None:
      raise TypeError('Tensor type %r is not supported' % x.dtype)

  with tf.name_scope(name, 'mean_and_var'):

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

    x_mean, x_var = combine_analyzer(
        inputs=combine_inputs,
        output_dtypes=[output_dtype] * 2,
        output_shapes=[output_shape] * 2,
        combiner_spec=_MeanAndVarCombinerSpec(output_dtype.as_numpy_dtype),
        name=name if name is not None else 'mean_and_var')

  return x_mean, x_var


class _MeanAndVarAccumulator(
    collections.namedtuple('MeanAndVarAccumulator',
                           ['count', 'mean', 'variance'])):
  """Container for _MeanAndVarCombinerSpec intermediate values."""

  @classmethod
  def make_nan_to_num(cls, counts, means, variances):
    return cls(counts, np.nan_to_num(means), np.nan_to_num(variances))

  def __reduce__(self):
    return self.__class__, tuple(self)


class _MeanAndVarCombinerSpec(CombinerSpec):
  """Combines a PCollection of accumulators to compute mean and variance."""

  def __init__(self, output_numpy_dtype):
    self._output_numpy_dtype = output_numpy_dtype

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

  def num_outputs(self):
    # The output is (mean, var).
    return 2

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


class _VocabularySpec(object):
  """Operation to compute unique values."""

  def __init__(self, top_k, frequency_threshold, vocab_filename,
               store_frequency, has_weights):
    self._top_k = top_k
    self._frequency_threshold = frequency_threshold
    self._vocab_filename = vocab_filename
    self._store_frequency = store_frequency
    self._has_weights = has_weights

  @property
  def top_k(self):
    return self._top_k

  @property
  def frequency_threshold(self):
    return self._frequency_threshold

  @property
  def vocab_filename(self):
    return self._vocab_filename

  @property
  def store_frequency(self):
    return self._store_frequency

  @property
  def has_weights(self):
    return self._has_weights


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


def vocabulary(x,
               top_k=None,
               frequency_threshold=None,
               vocab_filename=None,
               store_frequency=False,
               weights=None,
               name=None):
  r"""Computes the unique values of a `Tensor` over the whole dataset.

  Computes The unique values taken by `x`, which can be a `Tensor` or
  `SparseTensor` of any size.  The unique values will be aggregated over all
  dimensions of `x` and all instances.

  In case one of the tokens contains the '\n' or '\r' characters or is empty it
  will be discarded since we are currently writing the vocabularies as text
  files. This behavior will likely be fixed/improved in the future.

  The unique values are sorted by decreasing frequency and then decreasing
  lexicographical order.

  For large datasets it is highly recommended to either set frequency_threshold
  or top_k to control the size of the output, and also the run time of this
  operation.

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
      vocabulary file. Each line in the file will be of the form
      'frequency word\n'.
    weights: (Optional) Weights tensor for the vocabulary. Tensor must have the
      same shape as x.
      name: (Optional) A name for this operation.

  Returns:
    The path name for the vocabulary file containing the unique values of `x`.

  Raises:
    ValueError: If `top_k` or `frequency_threshold` is negative.
  """
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

  if isinstance(x, tf.SparseTensor):
    x = x.values

  if x.dtype != tf.string:
    raise ValueError('expected tf.string but got %r' % x.dtype)

  with tf.name_scope(name, 'vocabulary') as scope:
    if vocab_filename is not None:
      prefix = None
    elif store_frequency:
      prefix = VOCAB_FREQUENCY_FILENAME_PREFIX
    else:
      prefix = VOCAB_FILENAME_PREFIX

    # Make the file name path safe.
    vocab_filename = sanitized_vocab_filename(vocab_filename, prefix=prefix)

    spec = _VocabularySpec(top_k, frequency_threshold, vocab_filename,
                           store_frequency, weights is not None)

    x = tf.reshape(x, [-1])

    if weights is None:
      analyzer_inputs = [x]
    else:
      # Reducing in TF first.
      x = tf_utils.assert_same_shape(x, weights)
      unique = tf.unique(x, out_idx=tf.int64)

      weights = tf.reshape(weights, [-1])
      summed_weights = tf.unsorted_segment_sum(weights, unique.idx,
                                               tf.size(unique.y))
      analyzer_inputs = [unique.y, summed_weights]

    result = tf.placeholder(tf.string, [])
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, result)
    tf.add_to_collection(ANALYZER_COLLECTION,
                         Analyzer(analyzer_inputs, [result], spec, scope))
    return result


@deprecation.deprecated(None, 'Use `tft.vocabulary()` instead.')
def uniques(x,
            top_k=None,
            frequency_threshold=None,
            vocab_filename=None,
            store_frequency=False,
            weights=None,
            name=None):
  r"""See `tft.vocabulary`."""
  return vocabulary(
      x=x,
      top_k=top_k,
      frequency_threshold=frequency_threshold,
      vocab_filename=vocab_filename,
      store_frequency=store_frequency,
      name=name,
      weights=weights)


class _QuantilesCombinerSpec(CombinerSpec):
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
               has_weights=False):
    self._num_quantiles = num_quantiles
    self._epsilon = epsilon
    self._bucket_numpy_dtype = bucket_numpy_dtype
    self._always_return_num_quantiles = always_return_num_quantiles
    self._has_weights = has_weights

  def initialize_local_state(self, tf_config=None):
    """Called by the CombineFnWrapper's __init__ method.

    This can be used to set non-pickleable local state.  It is used in
    conjunction with overriding __reduce__ so this state is not pickled.  This
    method must be called prior to any other method.

    Args:
      tf_config: (optional) A tf.ConfigProto
    """
    # stamp_token is used to commit the state of the qaccumulator. In
    # this case, the qaccumulator state is completely returned and stored
    # as part of quantile_state/summary in the combiner fn (i.e the summary is
    # extracted and stored outside the qaccumulator). So we don't use
    # the timestamp mechanism to signify progress in the qaccumulator state.
    stamp_token = 0

    # Create a new session with a new graph for quantile ops.
    self._session = tf.Session(graph=tf.Graph(), config=tf_config)
    with self._session.graph.as_default():
      with self._session.as_default():
        self._qaccumulator = quantile_ops.QuantileAccumulator(
            init_stamp_token=stamp_token,
            num_quantiles=self._num_quantiles,
            epsilon=self._epsilon,
            name='qaccumulator',
            generate_quantiles=self._always_return_num_quantiles)
        resources.initialize_resources(resources.shared_resources()).run()

        # Create placeholders that will be used to provide input and weights to
        # the QuantileAccumulator.
        # They need to have shapes (1, None) as this is what the
        # QuantileAccumulator accepts.
        self._add_summary_input = tf.placeholder(
            dtype=self._bucket_numpy_dtype, shape=[1, None])
        if self._has_weights:
          self._add_summary_weights = tf.placeholder(
              dtype=tf.float32, shape=[1, None])
        else:
          self._add_summary_weights = tf.ones_like(self._add_summary_input)

        # Create op to update the accumulator with new input fed from
        # self._add_summary_input.
        self._add_summary_op = self._qaccumulator.add_summary(
            stamp_token=stamp_token,
            column=self._add_summary_input,
            example_weights=self._add_summary_weights)

        # Create op to add a prebuilt summary to the accumulator, and a
        # placeholder tensor to provide the input for this op.
        self._prebuilt_summary_input = tf.placeholder(
            dtype=tf.string, shape=[])
        self._add_prebuilt_summary_op = self._qaccumulator.add_prebuilt_summary(
            stamp_token=stamp_token,
            summary=self._prebuilt_summary_input)

        # Create op to flush summaries and return a summary representing the
        # summaries that were added the accumulator so far.
        self._flush_summary_op = self._qaccumulator.flush_summary(
            stamp_token=stamp_token,
            next_stamp_token=stamp_token)

        # Create ops to flush the accumulator and return approximate boundaries.
        self._flush_op = self._qaccumulator.flush(
            stamp_token=stamp_token,
            next_stamp_token=stamp_token)
        _, self._buckets_op = self._qaccumulator.get_buckets(
            stamp_token=stamp_token)

    # We generate an empty summary by calling self._flush_summary_op.
    # We cache this as some implementations may call create_accumulator for
    # every input, and it can be cached since it will always be the same and
    # immutable.
    self._empty_summary = self._session.run(self._flush_summary_op)

  def __reduce__(self):
    return _QuantilesCombinerSpec, (self._num_quantiles, self._epsilon,
                                    self._bucket_numpy_dtype,
                                    self._always_return_num_quantiles,
                                    self._has_weights)

  def create_accumulator(self):
    return None

  def add_input(self, summary, next_input):
    # next_input is a list of tensors each one representing a batch for its
    # respective input.  In this case we have a single input, which we reshape
    # to (1,?).
    flattened_input = np.reshape(next_input[0], newshape=(1, -1))

    if summary is None and flattened_input.size == 0:
      return None

    if self._has_weights:
      flattened_weights = np.reshape(next_input[1], newshape=(1, -1))
      if len(flattened_input) != len(flattened_weights):
        raise ValueError(
            'Values and weights contained different number of values ({} vs {})'
            .format(len(flattened_input), len(flattened_weights)))
      add_summary_op_feed_dict = {
          self._add_summary_input: flattened_input,
          self._add_summary_weights: flattened_weights
      }
    else:
      add_summary_op_feed_dict = {self._add_summary_input: flattened_input}

    self._session.run(
        self._add_prebuilt_summary_op,
        feed_dict={
            self._prebuilt_summary_input: summary or self._empty_summary
        })

    self._session.run(self._add_summary_op, add_summary_op_feed_dict)

    # After the flush_summary, qaccumulator will not contain any
    # uncommitted information that represents the input. Instead all the
    # digested information is returned as 'summary'. Many such summaries
    # will be combined by merge_accumulators().
    return self._session.run(self._flush_summary_op)

  def merge_accumulators(self, summaries):
    summaries = [summary for summary in summaries if summary is not None]
    if not summaries:
      return None

    for summary in summaries:
      self._session.run(
          self._add_prebuilt_summary_op,
          feed_dict={self._prebuilt_summary_input: summary})

    # Compute new summary.
    # All relevant state about the input is captured by 'summary'
    # (see comment at the end of add_input()).
    return self._session.run(self._flush_summary_op)

  def extract_output(self, summary):
    if summary is None:
      num_buckets = (
          self._num_quantiles - 1 if self._always_return_num_quantiles else 0)
      return [np.zeros((num_buckets,), np.float32)]

    # All relevant state about the input is captured by 'summary'
    # (see comment in add_input() and merge_accumulators()).
    self._session.run(
        self._add_prebuilt_summary_op,
        feed_dict={self._prebuilt_summary_input: summary})
    self._session.run(self._flush_op)
    buckets = self._session.run(self._buckets_op)

    # Quantile boundaries is a list of the form
    #    [np.ndarrary(min, <internal-boundaries>, max)]
    # If always_return_num_quantiles is set to True, the number of elements in
    # buckets is always equal to num_quantiles + 1. Hence we trim the min and
    # max quantile boundaries to return the internal boundaries.
    if self._always_return_num_quantiles:
      return [buckets[1:-1]]

    # If always_return_num_quantiles is set to False, the approximate quantile
    # library can return less or more than requested number of quantiles.
    # The max value can be same as the last internal boundary, due to removal
    # of duplicates. Below, the min and/or max quantile boundaries are trimmed
    # depending on the actual boundaries returned by the library.
    if buckets.size >= (self._num_quantiles + 1):
      # Trim min/max.
      buckets = buckets[1:-1]
    elif buckets.size == self._num_quantiles:
      # Trim min only.
      buckets = buckets[1:]
    else:
      # Do not trim min/max, these are part of requested boundaries.
      pass

    return [buckets]

  def num_outputs(self):
    return 1


def quantiles(x, num_buckets, epsilon, weights=None, name=None):
  """Computes the quantile boundaries of a `Tensor` over the whole dataset.

  quantile boundaries are computed using approximate quantiles,
  and error tolerance is specified using `epsilon`. The boundaries divide the
  input tensor into approximately equal `num_buckets` parts.
  See go/squawd for details, and how to control the error due to approximation.

  Args:
    x: An input `Tensor`.
    num_buckets: Values in the `x` are divided into approximately
      equal-sized buckets, where the number of buckets is num_buckets.
      This is a hint. The actual number of buckets computed can be
      less or more than the requested number. Use the generated metadata to
      find the computed number of buckets.
    epsilon: Error tolerance, typically a small fraction close to zero
      (e.g. 0.01). Higher values of epsilon increase the quantile approximation,
      and hence result in more unequal buckets, but could improve performance,
      and resource consumption.  Some measured results on memory consumption:
      For epsilon = 0.001, the amount of memory for each buffer to hold the
      summary for 1 trillion input values is ~25000 bytes. If epsilon is
      relaxed to 0.01, the buffer size drops to ~2000 bytes for the same input
      size. If we use a strict epsilon value of 0, the buffer size is same size
      as the input, because the intermediate stages have to remember every input
      and the quantile boundaries can be found only after an equivalent to a
      full sorting of input. The buffer size also determines the amount of work
      in the different stages of the beam pipeline, in general, larger epsilon
      results in fewer and smaller stages, and less time. For more performance
      trade-offs see also http://web.cs.ucla.edu/~weiwang/paper/SSDBM07_2.pdf
    weights: (Optional) Weights tensor for the quantiles. Tensor must have the
      same shape as x.
    name: (Optional) A name for this operation.

  Returns:
    The bucket boundaries represented as a list, with num_bucket-1 elements
    See code below for discussion on the type of bucket boundaries.
  """

  with tf.name_scope(name, 'quantiles'):
    bucket_dtype = tf.float32
    if weights is None:
      analyzer_inputs = [x]
      has_weights = False
      always_return_num_quantiles = False
    else:
      x = tf_utils.assert_same_shape(x, weights)
      analyzer_inputs = [x, weights]
      has_weights = True
      always_return_num_quantiles = True
    combiner_spec = _QuantilesCombinerSpec(
        num_buckets,
        epsilon,
        bucket_dtype.as_numpy_dtype,
        always_return_num_quantiles=always_return_num_quantiles,
        has_weights=has_weights)
    quantile_boundaries = combine_analyzer(analyzer_inputs, [bucket_dtype],
                                           [(None,)], combiner_spec,
                                           'quantiles')[0]
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
  with tf.name_scope(name, 'quantiles_by_key'):
    if key.dtype != tf.string:
      raise ValueError('key must have type tf.string')
    bucket_dtype = tf.float32
    combiner_spec = _CombinePerKeySpec(
        _QuantilesCombinerSpec(num_buckets, epsilon,
                               bucket_dtype.as_numpy_dtype,
                               always_return_num_quantiles=True))
    return combine_analyzer([key, x], [tf.string, bucket_dtype],
                            [(None,), (None, None)], combiner_spec, 'quantiles')


class _CovarianceCombinerSpec(CombinerSpec):
  """Combines the PCollection to compute the biased covariance matrix."""

  def __init__(self, numpy_dtype=np.float64):
    """Store the dtype for np arrays/matrices for precision."""
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

  def num_outputs(self):
    return 1


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

  x.shape.assert_has_rank(2)

  input_dim = x.shape.as_list()[1]
  shape = (input_dim, input_dim)

  spec = _CovarianceCombinerSpec(dtype.as_numpy_dtype)
  return combine_analyzer(
      [x], [dtype], [shape], spec,
      name if name is not None else 'covariance')[0]


class _PCACombinerSpec(_CovarianceCombinerSpec):
  """Compute PCA of accumulated data using the biased covariance matrix."""

  def __init__(self, output_dim=None, numpy_dtype=np.float64):
    """Store pca output dimension, and dtype for precision."""
    super(_PCACombinerSpec, self).__init__(numpy_dtype=numpy_dtype)
    self._output_dim = output_dim

  def extract_output(self, accumulator):
    """Compute PCA of the accumulated data using the biased covariance matrix.

    Following the covariance computation in _CovarianceCombinerSpec,
    this method runs eigenvalue decomposition on the covariance matrix,
    sorts eigenvalues in decreasing order, and returns the first output_dim
    corresponding eigenvectors (principal components) as a matrix.

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

  def num_outputs(self):
    return 1


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

  x.shape.assert_has_rank(2)

  input_dim = x.shape.as_list()[1]
  shape = (input_dim, output_dim)

  spec = _PCACombinerSpec(output_dim, dtype.as_numpy_dtype)
  return combine_analyzer(
      [x], [dtype], [shape], spec,
      name if name is not None else 'pca')[0]


