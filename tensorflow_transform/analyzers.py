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
      accumulator: the final accumulator value.  Should be a list of ndarrays.

    Returns: A list of ndarrays representing the result of this combiner.
    """
    raise NotImplementedError


def combine_analyzer(inputs, output_dtypes, output_shapes, combiner_spec, name):
  """Applies the combiner over the whole dataset.

  Args:
    inputs: A list of input `Tensor`s or `SparseTensor`s.
    output_dtypes: The list of dtypes of the output of the analyzer.
    output_shapes: The list of dtypes of the output of the analyzer.  Must have
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
    raise ValueError('output_dtypes (%r) and output_shapes (%r) had different'
                     ' lengths' % output_dtypes, output_shapes)
  return Analyzer(
      inputs,
      [(output_dtype, output_shape, False)
       for output_dtype, output_shape in zip(output_dtypes, output_shapes)],
      combiner_spec,
      name).outputs


class _NumPyCombinerSpec(CombinerSpec):
  """Combines the PCollection only on the 0th dimension using nparray."""

  def __init__(self, fn, reduce_instance_dims):
    self._fn = fn
    self._reduce_instance_dims = reduce_instance_dims

  def create_accumulator(self):
    return None

  def add_input(self, accumulator, batch_values):
    if self._reduce_instance_dims:
      reduced_values = [self._fn(batch_value) for batch_value in batch_values]
    else:
      reduced_values = [self._fn(batch_value, axis=0)
                        for batch_value in batch_values]
    if accumulator is None:
      return reduced_values
    else:
      return [
          self._fn((sub_accumulator, reduced_value), axis=0)
          for sub_accumulator, reduced_value
          in zip(accumulator, reduced_values)]

  def merge_accumulators(self, accumulators):
    # numpy's sum, min, max, etc functions operate on array-like objects, but
    # not arbitrary iterables. Convert the provided accumulators into a list
    return [
        self._fn(list(sub_accumulators), axis=0)
        for sub_accumulators in zip(*accumulators)]

  def extract_output(self, accumulator):
    return accumulator


def _numeric_combine(inputs, fn, reduce_instance_dims=True, name=None):
  """Apply a reduction, defined by a numpy function to multiple inputs.

  Args:
    inputs: A list of tensors, which will be indpendently reduced.
    fn: A function to reduce tensors across instances/batches, to get a single
        output.
    reduce_instance_dims: By default collapses the batch and instance dimensions
        to arrive at a single scalar output. If False, only collapses the batch
        dimension and outputs a vector of the same shape as the input.
    name: (Optional) A name for this operation.

  Returns:
     A list of tensors with the same length as `inputs`, representing the
         input tensors that have been reduced by `fn` across instances and
         batches.
  """
  for x in inputs:
    if not isinstance(x, tf.Tensor):
      raise TypeError('Expected a Tensor, but got %r' % x)

  if reduce_instance_dims:
    # If reducing over all dimensions, result is scalar.
    shapes = [() for _ in inputs]
  else:
    # If reducing over batch dimensions, with known shape, the result will be
    # the same shape as the input, but without the batch.  If reducing over
    # batch dimensions, with unknown shape, the result will also have unknown
    # shape.
    shapes = [x.shape.as_list()[1:] if x.shape.dims is not None else None
              for x in inputs]
  return combine_analyzer(
      inputs,
      [x.dtype for x in inputs],
      shapes,
      _NumPyCombinerSpec(fn, reduce_instance_dims),
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
  return _numeric_combine([x], np.min, reduce_instance_dims, name)[0]


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
  return _numeric_combine([x], np.max, reduce_instance_dims, name)[0]


def _min_and_max(x, reduce_instance_dims=True, name=None):  # pylint: disable=redefined-builtin
  with tf.name_scope(name, 'min_and_max'):
    # Unary minus op doesn't support tf.int64, so use 0 - x instead of -x.
    minus_x_min, x_max = _numeric_combine(  # pylint: disable=unbalanced-tuple-unpacking
        [0 - x, x], np.max, reduce_instance_dims)
    return 0 - minus_x_min, x_max


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
  return _numeric_combine([x], np.sum, reduce_instance_dims, name)[0]


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
    # For now _numeric_combine will return a tuple with as many elements as the
    # input tuple.
    x_count, x_sum = _numeric_combine(  # pylint: disable=unbalanced-tuple-unpacking
        [tf.ones_like(x), x], np.sum, reduce_instance_dims)
    return tf.divide(x_sum, x_count)


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


def _mean_and_var(x, reduce_instance_dims=True, name=None):
  """More efficient combined `mean` and `var`.  See `var`."""
  with tf.name_scope(name, 'mean_and_var'):
    # Note: Calling `mean`, `sum`, and `size` as defined in this module, not the
    # builtins.
    x_mean = mean(x, reduce_instance_dims)
    # x_mean will be float32 or float64, depending on type of x.
    squared_deviations = tf.square(tf.cast(x, x_mean.dtype) - x_mean)
    x_var = mean(squared_deviations, reduce_instance_dims)
    return x_mean, x_var


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

    # The Analyzer returns a 2d matrix of 1*num_buckets.  Below, we remove
    # the first dimension and return the boundaries as a simple 1d list.
    return quantile_boundaries[0:1]


class _CovarianceCombinerSpec(CombinerSpec):
  """Combines the PCollection to compute the biased covariance matrix."""

  def __init__(self, dtype=tf.float64):
    """Store the dtype for np arrays/matrices for precision."""
    self._output_dtype = dtype
    self._np_dtype = dtype.as_numpy_dtype

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
    ).astype(self._np_dtype)

    batch_sum = np.array(np.sum(batch_value, axis=0), self._np_dtype)
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
    # Because each accumulator contains multiple arrays of different dimensions,
    # the np.sum operation must be explicitly used across the entries within
    # each accumulator. np.sum(list(accumulators)) does not work.

    sum_product = np.sum(
        [accumulator[0] for accumulator in accumulators], axis=0)
    sum_vectors = np.sum(
        [accumulator[1] for accumulator in accumulators], axis=0)
    count = np.sum([accumulator[2] for accumulator in accumulators], axis=0)
    return [sum_product, sum_vectors, count]

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
    dtype: numpy dtype of entries in the returned matrix.
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

  spec = _CovarianceCombinerSpec(dtype)
  return combine_analyzer(
      [x], [dtype], [shape], spec,
      name if name is not None else 'covariance')[0]


class _PCACombinerSpec(_CovarianceCombinerSpec):

  def __init__(self, output_dim=None, dtype=tf.float64):
    """Store pca output dimension, and dtype for precision."""
    super(_PCACombinerSpec, self).__init__(dtype=dtype)
    self._output_dim = output_dim

  def extract_output(self, accumulator):
    """Compute PCA the accumulated data using the biased covariance matrix.

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
    dtype: numpy dtype of entries in the returned matrix.
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

  spec = _PCACombinerSpec(output_dim, dtype)
  return combine_analyzer(
      [x], [dtype], [shape], spec,
      name if name is not None else 'pca')[0]
