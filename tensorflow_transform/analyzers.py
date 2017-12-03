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

import re

import numpy as np
import tensorflow as tf


ANALYZER_COLLECTION = 'tft_analyzers'
VOCAB_FILENAME_PREFIX = 'vocab_'
VOCAB_FREQUENCY_FILENAME_PREFIX = 'vocab_frequency_'


class Analyzer(object):
  """An operation-like class for full-pass analyses of data.

  An Analyzer is like a tf.Operation except that it requires computation over
  the full dataset.  E.g. sum(my_tensor) will compute the sum of the value of
  my_tensor over all instances in the dataset.  The Analyzer class contains the
  inputs to this computation, and placeholders which will later be converted to
  constants during a call to AnalyzeDataset.

  Args:
    inputs: The inputs to the analyzer.
    output_dtype_shape_and_is_asset: List of tuples of `(DType, Shape, bool)`
      for each output.  A tf.placeholder with the given DType and Shape will be
      constructed to represent the output of the analyzer, and this placeholder
      will eventually be replaced by the actual value of the analyzer.  The
      boolean value states whether this Tensor represents an asset filename or
      not.
    spec: A description of the computation to be done.
    name: Similar to a TF op name.  Used to define a unique scope for this
      analyzer, which can be used for debugging info.

  Raises:
    ValueError: If the inputs are not all `Tensor`s.
  """

  def __init__(self, inputs, output_dtype_shape_and_is_asset, spec, name):
    for tensor in inputs:
      if not isinstance(tensor, tf.Tensor):
        raise ValueError('Analyzers can only accept `Tensor`s as inputs')
    self._inputs = inputs
    self._outputs = []
    self._output_is_asset_map = {}
    with tf.name_scope(name) as scope:
      self._name = scope
      for dtype, shape, is_asset in output_dtype_shape_and_is_asset:
        output_tensor = tf.placeholder(dtype, shape)
        if is_asset and output_tensor.dtype != tf.string:
          raise ValueError(('Tensor {} cannot represent an asset, because it '
                            'is not a string.').format(output_tensor.name))
        self._outputs.append(output_tensor)
        self._output_is_asset_map[output_tensor] = is_asset
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

  @property
  def name(self):
    return self._name

  def output_is_asset(self, output_tensor):
    return self._output_is_asset_map[output_tensor]


class CombinerSpec(object):
  """Analyze using combiner function.

  This object mirrors a beam.CombineFn, that will receive a beam PCollection
  representing the batched input tensors.
  """

  def create_accumulator(self):
    """Return a fresh, empty accumulator.

    Returns: An empty accumulator.  This can be an Python value.
    """
    raise NotImplementedError

  def add_input(self, accumulator, element):
    """Return result of folding element into accumulator.

    Args:
      accumulator: the current accumulator
      element: the element to add, which will be an ndarray representing the
         value of the input for a batch.

    Returns: An accumulator that includes the additional element.
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
      accumulator: the final accumulator value.  Should be a list of ndarrays.

    Returns: An ndarray representing the result of this combiner.
    """
    raise NotImplementedError


def combine_analyzer(x, output_dtype, output_shape, combiner_spec, name):
  """Applies the combiner over the whole dataset.

  Args:
    x: An input `Tensor` or `SparseTensor`.
    output_dtype: The dtype of the output of the analyzer.
    output_shape: The shape of the output of the analyzer.
    combiner_spec: A subclass of CombinerSpec.
    name: Similar to a TF op name.  Used to define a unique scope for this
      analyzer, which can be used for debugging info.

  Returns:
    The combined values, which is a `Tensor` with type output_dtype and shape
    `output_shape`.  These must be compatible with the combiner_spec.
  """
  return Analyzer([x], [(output_dtype, output_shape, False)], combiner_spec,
                  name).outputs[0]


class _NumPyCombinerSpec(CombinerSpec):
  """Combines the PCollection only on the 0th dimension using nparray."""

  def __init__(self, fn, reduce_instance_dims):
    self._fn = fn
    self._reduce_instance_dims = reduce_instance_dims

  def create_accumulator(self):
    return None

  def add_input(self, accumulator, next_input):
    if self._reduce_instance_dims:
      batch = self._fn(next_input)
    else:
      batch = self._fn(next_input, axis=0)
    if accumulator is None:
      return batch
    else:
      return self._fn((accumulator, batch), axis=0)

  def merge_accumulators(self, accumulators):
    # numpy's sum, min, max, etc functions operate on array-like objects, but
    # not arbitrary iterables. Convert the provided accumulators into a list
    return self._fn(list(accumulators), axis=0)

  def extract_output(self, accumulator):
    return [accumulator]


def _numeric_combine(x, fn, reduce_instance_dims=True, name=None):
  """Apply an analyzer with _NumericCombineSpec to given input."""
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
  return combine_analyzer(
      x, x.dtype, shape, _NumPyCombinerSpec(fn, reduce_instance_dims),
      name if name is not None else fn.__name__)


def min(x, reduce_instance_dims=True, name=None):  # pylint: disable=redefined-builtin
  """Computes the minimum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a `Tensor` of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _numeric_combine(x, np.min, reduce_instance_dims, name)


def max(x, reduce_instance_dims=True, name=None):  # pylint: disable=redefined-builtin
  """Computes the maximum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _numeric_combine(x, np.max, reduce_instance_dims, name)


def sum(x, reduce_instance_dims=True, name=None):  # pylint: disable=redefined-builtin
  """Computes the sum of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  return _numeric_combine(x, np.sum, reduce_instance_dims, name)


def size(x, reduce_instance_dims=True, name=None):
  """Computes the total size of instances in a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with tf.name_scope(name, 'size'):
    # Note: Calling `sum` defined in this module, not the builtin.
    return sum(tf.ones_like(x), reduce_instance_dims)


def mean(x, reduce_instance_dims=True, name=None):
  """Computes the mean of the values of a `Tensor` over the whole dataset.

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the mean. If `x` is floating point, the mean will
    have the same type as `x`. If `x` is integral, the output is cast to float32
    for int8 and int16 and float64 for int32 and int64 (similar to the behavior
    of tf.truediv).
  """
  with tf.name_scope(name, 'mean'):
    # Note: Calling `sum` defined in this module, not the builtin.
    return tf.divide(
        sum(x, reduce_instance_dims), size(x, reduce_instance_dims))


def var(x, reduce_instance_dims=True, name=None):
  """Computes the variance of the values of a `Tensor` over the whole dataset.

  Uses the biased variance (0 delta degrees of freedom), as given by
  (x - mean(x))**2 / length(x).

  Args:
    x: A `Tensor`.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the variance. If `x` is floating point, the variance
    will have the same type as `x`. If `x` is integral, the output is cast to
    float32 for int8 and int16 and float64 for int32 and int64 (similar to the
    behavior of tf.truediv).
  """
  with tf.name_scope(name, 'var'):
    # Note: Calling `mean`, `sum`, and `size` as defined in this module, not the
    # builtins.
    x_mean = mean(x, reduce_instance_dims)
    # x_mean will be float32 or float64, depending on type of x.
    squared_deviations = tf.square(tf.cast(x, x_mean.dtype) - x_mean)
    return mean(squared_deviations, reduce_instance_dims)


class _UniquesSpec(object):
  """Operation to compute unique values."""

  def __init__(self, top_k, frequency_threshold,
               vocab_filename, store_frequency):
    self._top_k = top_k
    self._frequency_threshold = frequency_threshold
    self._vocab_filename = vocab_filename
    self._store_frequency = store_frequency

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


def uniques(x, top_k=None, frequency_threshold=None,
            vocab_filename=None, store_frequency=False, name=None):
  r"""Computes the unique values of a `Tensor` over the whole dataset.

  Computes The unique values taken by `x`, which can be a `Tensor` or
  `SparseTensor` of any size.  The unique values will be aggregated over all
  dimensions of `x` and all instances.

  In case one of the tokens contains the '\n' or '\r' characters or is empty it
  will be discarded since we are currently writing the vocabularies as text
  files. This behavior will likely be fixed/improved in the future.

  The unique values are sorted by decreasing frequency and then decreasing
  lexicographical order.

  Args:
    x: An input `Tensor` or `SparseTensor`.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      frequency is >= to the supplied threshold. If set to None, the full
      vocabulary is generated.
    vocab_filename: The file name for the vocabulary file. If none, the
      "uniques" scope name in the context of this graph will be used as the file
      name. If not None, should be unique within a given preprocessing function.
      NOTE To make your pipelines resilient to implementation details please
      set `vocab_filename` when you are using the vocab_filename on a downstream
      component.
    store_frequency: If True, frequency of the words is stored in the
      vocabulary file. Each line in the file will be of the form
      'frequency word\n'.
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

  if isinstance(x, tf.SparseTensor):
    x = x.values

  with tf.name_scope(name, 'uniques'):
    if vocab_filename is not None:
      prefix = None
    elif store_frequency:
      prefix = VOCAB_FREQUENCY_FILENAME_PREFIX
    else:
      prefix = VOCAB_FILENAME_PREFIX

    # Make the file name path safe.
    vocab_filename = sanitized_vocab_filename(vocab_filename, prefix=prefix)

    spec = _UniquesSpec(top_k, frequency_threshold, vocab_filename,
                        store_frequency)
    return Analyzer([x], [(tf.string, [], True)], spec, 'uniques').outputs[0]


class _QuantilesSpec(object):
  """Operation to compute quantile boundaries."""

  def __init__(self, epsilon, num_buckets):
    self._epsilon = epsilon
    self._num_buckets = num_buckets

  @property
  def epsilon(self):
    return self._epsilon

  @property
  def num_buckets(self):
    return self._num_buckets

  @property
  def bucket_dtype(self):
    return tf.float32


def quantiles(x, num_buckets, epsilon, name=None):
  """Computes the quantile boundaries of a `Tensor` over the whole dataset.

  quantile boundaries are computed using approximate quantiles,
  and error tolerance is specified using `epsilon`. The boundaries divide the
  input tensor into approximately equal `num_buckets` parts.
  See go/squawd for details, and how to control the error due to approximation.

  Args:
    x: An input `Tensor` or `SparseTensor`.
    num_buckets: Values in the `x` are divided into approximately equal-sized
      buckets, where the number of buckets is num_buckets.
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
    name: (Optional) A name for this operation.

  Returns:
    The bucket boundaries represented as a list, with num_bucket-1 elements
    See bucket_dtype() above for type of bucket boundaries.
  """

  with tf.name_scope(name, 'quantiles'):
    spec = _QuantilesSpec(epsilon, num_buckets)
    quantile_boundaries = Analyzer(
        [x], [(spec.bucket_dtype, [1, None], False)], spec,
        'quantiles').outputs[0]

    # quantile boundaries is of the form
    #    [nd.arrary(first, <num_buckets-1>, last)]
    # Drop the fist and last quantile boundaries, so that we end-up with
    # num_buckets-1 boundaries, and hence num_buckets buckets.
    return quantile_boundaries[0:1, 1:-1]
