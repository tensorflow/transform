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

import tensorflow as tf


ANALYZER_COLLECTION = 'tft_analyzers'


class Analyzer(object):
  """An operation-like class for full-pass analyses of data.

  An Analyzer is like a tf.Operation except that it requires computation over
  the full dataset.  E.g. sum(my_tensor) will compute the sum of the value of
  my_tensor over all instances in the dataset.  The Analyzer class contains the
  inputs to this computation, and placeholders which will later be converted to
  constants during a call to AnalyzeDataset.

  Args:
    inputs: The inputs to the analyzer.
    output_shapes_and_dtype: List of pairs of (dtype, shape) for each output.
    spec: A description of the computation to be done.

  Raises:
    ValueError: If the inputs are not all `Tensor`s.
  """

  def __init__(self, inputs, output_dtypes_and_shapes, spec):
    for tensor in inputs:
      if not isinstance(tensor, tf.Tensor):
        raise ValueError('Analyzers can only accept `Tensor`s as inputs')
    self._inputs = inputs
    self._outputs = [tf.placeholder(dtype, shape)
                     for dtype, shape in output_dtypes_and_shapes]
    self._spec = spec
    tf.add_to_collection(ANALYZER_COLLECTION, self)

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def spec(self):
    return self._spec


class NumericCombineSpec(object):
  """Operation to combine numeric values."""

  MIN = 'min'
  MAX = 'max'
  SUM = 'sum'

  def __init__(self, dtype, combiner_type, reduce_instance_dims):
    self._dtype = dtype
    self._combiner_type = combiner_type
    self._reduce_instance_dims = reduce_instance_dims

  @property
  def dtype(self):
    return self._dtype

  @property
  def combiner_type(self):
    return self._combiner_type

  @property
  def reduce_instance_dims(self):
    return self._reduce_instance_dims


def _numeric_combine(x, combiner_type, reduce_instance_dims=True):
  """Apply an analyzer with NumericCombineSpec to given input."""
  if not isinstance(x, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x)

  if reduce_instance_dims:
    # If reducing over all dimensions, result is scalar.
    shape = ()
  elif x.shape.dims is not None:
    # If reducing over batch dimensions, with known shape, the result will be
    # the same shape as the input, but without the batch.
    shape = x.shape.as_list()[1:]
  else:
    # If reducing over batch dimensions, with unknown shape, the result will
    # also have unknown shape.
    shape = None
  with tf.name_scope(combiner_type):
    spec = NumericCombineSpec(x.dtype, combiner_type, reduce_instance_dims)
    return Analyzer([x], [(x.dtype, shape)], spec).outputs[0]


def min(x, reduce_instance_dims=True):  # pylint: disable=redefined-builtin
  """Computes the minimum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a `Tensor` of the same shape as the input.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _numeric_combine(x, NumericCombineSpec.MIN, reduce_instance_dims)


def max(x, reduce_instance_dims=True):  # pylint: disable=redefined-builtin
  """Computes the maximum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the output.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _numeric_combine(x, NumericCombineSpec.MAX, reduce_instance_dims)


def sum(x, reduce_instance_dims=True):  # pylint: disable=redefined-builtin
  """Computes the sum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the output.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _numeric_combine(x, NumericCombineSpec.SUM, reduce_instance_dims)


def size(x, reduce_instance_dims=True):
  """Computes the total size of instances in a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the output.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with tf.name_scope('size'):
    # Note: Calling `sum` defined in this module, not the builtin.
    return sum(tf.ones_like(x), reduce_instance_dims)


def mean(x, reduce_instance_dims=True):
  """Computes the mean of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the output.

  Returns:
    A `Tensor` containing the mean. If `x` is floating point, the mean will
    have the same type as `x`. If `x` is integral, the output is cast to float32
    for int8 and int16 and float64 for int32 and int64 (similar to the behavior
    of tf.truediv).
  """
  with tf.name_scope('mean'):
    # Note: Calling `sum` defined in this module, not the builtin.
    return tf.divide(
        sum(x, reduce_instance_dims), size(x, reduce_instance_dims))


def var(x, reduce_instance_dims=True):
  """Computes the variance of the values of a `Tensor` over the whole dataset.

  Uses the biased variance (0 delta degrees of freedom), as given by
  (x - mean(x))**2 / length(x).

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the output.

  Returns:
    A `Tensor` containing the variance. If `x` is floating point, the variance
    will have the same type as `x`. If `x` is integral, the output is cast to
    float32 for int8 and int16 and float64 for int32 and int64 (similar to the
    behavior of tf.truediv).
  """
  with tf.name_scope('var'):
    # Note: Calling `mean`, `sum`, and `size` as defined in this module, not the
    # builtins.
    x_mean = mean(x, reduce_instance_dims)
    # x_mean will be float32 or float64, depending on type of x.
    squared_deviations = tf.square(tf.cast(x, x_mean.dtype) - x_mean)
    return mean(squared_deviations, reduce_instance_dims)


class UniquesSpec(object):
  """Operation to compute unique values."""

  def __init__(self, dtype, top_k, frequency_threshold):
    self._dtype = dtype
    self._top_k = top_k
    self._frequency_threshold = frequency_threshold

  @property
  def dtype(self):
    return self._dtype

  @property
  def top_k(self):
    return self._top_k

  @property
  def frequency_threshold(self):
    return self._frequency_threshold


def uniques(x, top_k=None, frequency_threshold=None):
  """Computes the unique values of a `Tensor` over the whole dataset.

  Computes The unique values taken by `x`, which can be a `Tensor` or
  `SparseTensor` of any size.  The unique values will be aggregated over all
  dimensions of `x` and all instances.

  The unique values are sorted by decreasing frequency and then decreasing
  value.

  Args:
    x: An input `Tensor` or `SparseTensor`.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      frequency is >= to the supplied threshold. If set to None, the full
      vocabulary is generated

  Returns:
    The unique values of `x`.

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
  if isinstance(x, tf.SparseTensor):
    x = x.values
  with tf.name_scope('uniques'):
    spec = UniquesSpec(x.dtype, top_k, frequency_threshold)
    return Analyzer([x], [(x.dtype, [None])], spec).outputs[0]
