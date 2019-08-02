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
"""The core public API of TFTransform.  Provide functions to transform tensors.

The core tf.Transform API requires a user to construct a
"preprocessing function" that accepts and returns `Tensor`s.  This function is
built by composing regular functions built from TensorFlow ops, as well as
special functions we refer to as `Analyzer`s.  `Analyzer`s behave similarly to
TensorFlow ops but require a full pass over the whole dataset to compute their
output value.  The analyzers are defined in analyzers.py, while this module
provides helper functions that call analyzers and then use the results of the
anaylzers to transform the original data.

The user-defined preprocessing function should accept and return `Tensor`s that
are batches from the dataset, whose batch size may vary.  For example the
following preprocessing function centers the input 'x' while returning 'y'
unchanged.

import tensorflow_transform as tft

def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']

  # Apply the `mean` analyzer to obtain the mean x.
  x_mean = tft.mean(x)

  # Subtract the mean.
  x_centered = x - mean

  # Return a new dictionary containing x_centered, and y unchanged
  return {
    'x_centered': x_centered,
    'y': y
  }

This user-defined function then must be run using an implementation based on
some distributed computation framework.  The canonical implementation uses
Apache Beam as the underlying framework.  See beam/impl.py for how to use the
Beam implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
# GOOGLE-INITIALIZATION
import six

import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import schema_inference
from tensorflow_transform import tf_utils

# pylint: disable=g-direct-tensorflow-import
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.util import deprecation
# pylint: enable=g-direct-tensorflow-import

# TODO(b/132098015): Schema annotations aren't yet supported in OSS builds.
# pylint: disable=g-import-not-at-top
try:
  from tensorflow_transform import annotations_pb2
except ImportError:
  pass
# pylint: enable=g-import-not-at-top


def sparse_tensor_to_dense_with_shape(x, shape, default_value=0):
  """Converts a `SparseTensor` into a dense tensor and sets its shape.

  Args:
    x: A `SparseTensor`.
    shape: The desired shape of the densified `Tensor`.
    default_value: (Optional) Value to set for indices not specified. Defaults
      to zero.

  Returns:
    A `Tensor` with the desired shape.

  Raises:
    ValueError: If input is not a `SparseTensor`.
  """
  if not isinstance(x, tf.SparseTensor):
    raise ValueError('input must be a SparseTensor')
  new_dense_shape = [
      x.dense_shape[i] if size is None else size
      for i, size in enumerate(shape)
  ]
  dense = tf.compat.v1.sparse_to_dense(x.indices, new_dense_shape, x.values,
                                       default_value)
  dense.set_shape(shape)
  return dense


def scale_by_min_max(x,
                     output_min=0.0,
                     output_max=1.0,
                     elementwise=False,
                     name=None):
  """Scale a numerical column into the range [output_min, output_max].

  Args:
    x: A numeric `Tensor`.
    output_min: The minimum of the range of output values.
    output_max: The maximum of the range of output values.
    elementwise: If true, scale each element of the tensor independently.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the input column scaled to [output_min, output_max].

  Raises:
    ValueError: If output_min, output_max have the wrong order.
  """
  with tf.compat.v1.name_scope(name, 'scale_by_min_max'):
    return scale_by_min_max_per_key(x,
                                    key=None,
                                    output_min=output_min,
                                    output_max=output_max,
                                    elementwise=elementwise,
                                    name=name)


def scale_by_min_max_per_key(x,
                             key=None,
                             output_min=0.0,
                             output_max=1.0,
                             elementwise=False,
                             name=None):
  """Scale a numerical column into a predefined range on a per-key basis.

  Args:
    x: A numeric `Tensor` or `SparseTensor`.
    key: A `Tensor` or `SparseTensor` of dtype tf.string.
        Must meet one of the following conditions:
        0. key is None
        1. Both x and key are dense,
        2. Both x and key are sparse and `key` must exactly match `x` in
           everything except values,
        3. The axis=1 index of each x matches its index of dense key.
    output_min: The minimum of the range of output values.
    output_max: The maximum of the range of output values.
    elementwise: If true, scale each element of the tensor independently.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor`  or `SparseTensor` containing the input column scaled to
    [output_min, output_max] on a per-key basis if a key is provided.

  Raises:
    ValueError: If output_min, output_max have the wrong order.
    NotImplementedError: If elementwise is True and key is not None.
    InvalidArgumentError: If indices of sparse x and key do not match.
  """
  with tf.compat.v1.name_scope(name, 'scale_by_min_max_per_key'):
    if output_min >= output_max:
      raise ValueError('output_min must be less than output_max')

    x = tf.cast(x, tf.float32)
    if key is None:
      min_x_value, max_x_value = analyzers._min_and_max(  # pylint: disable=protected-access
          x, reduce_instance_dims=not elementwise)
    else:
      if elementwise:
        raise NotImplementedError('Per-key elementwise reduction not supported')
      key_vocab, min_x_value, max_x_value = analyzers._min_and_max_per_key(  # pylint: disable=protected-access
          x, key, reduce_instance_dims=not elementwise)

      min_x_value, max_x_value = tf_utils.map_per_key_reductions(
          (min_x_value, max_x_value), key, key_vocab, x)

    compose_result_fn = _make_sparse_tensor_wrapper_if_sparse(x)
    x_values = x
    if isinstance(x, tf.SparseTensor):
      if elementwise:
        # Only supports SparseTensors with rank 2.
        x.get_shape().assert_has_rank(2)
        min_x_value = tf.gather(min_x_value, x.indices[:, 1])
        max_x_value = tf.gather(max_x_value, x.indices[:, 1])
      x_values = x.values

    x_shape = tf.shape(input=x_values)

    # If min==max, the result will be the mean of the requested range.
    # Note that both the options of tf.where are computed, which means that this
    # will compute unused NaNs.
    numerator = tf.cast(x_values, min_x_value.dtype) - min_x_value
    where_cond = min_x_value < max_x_value
    where_cond = tf.cast(
        tf.zeros_like(numerator) + tf.cast(where_cond, numerator.dtype),
        dtype=tf.bool)
    scaled_result = tf.where(where_cond,
                             numerator / (max_x_value - min_x_value),
                             tf.fill(x_shape, 0.5))

    return compose_result_fn(
        (scaled_result * (output_max - output_min)) + output_min)


def scale_to_0_1(x, elementwise=False, name=None):
  """Returns a column which is the input column scaled to have range [0,1].

  Args:
    x: A numeric `Tensor`.
    elementwise: If true, scale each element of the tensor independently.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the input column scaled to [0, 1].
  """
  return scale_by_min_max(x, 0, 1, elementwise=elementwise, name=name)


def scale_to_0_1_per_key(x, key, elementwise=False, name=None):
  """Returns a column which is the input column scaled to have range [0,1].

  Args:
    x: A numeric `Tensor`.
    key: A `Tensor` of type string.
    elementwise: If true, scale each element of the tensor independently.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the input column scaled to [0, 1], per key.
  """
  return scale_by_min_max_per_key(x, key, 0, 1, elementwise, name=name)


def scale_to_z_score(x, elementwise=False, name=None, output_dtype=None):
  """Returns a standardized column with mean 0 and variance 1.

  Scaling to z-score subtracts out the mean and divides by standard deviation.
  Note that the standard deviation computed here is based on the biased variance
  (0 delta degrees of freedom), as computed by analyzers.var.

  Args:
    x: A numeric `Tensor` or `SparseTensor`.
    elementwise: If true, scales each element of the tensor independently;
        otherwise uses the mean and variance of the whole tensor.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` or `SparseTensor` containing the input column scaled to mean 0
    and variance 1 (standard deviation 1), given by: (x - mean(x)) / std_dev(x).
    If `x` is floating point, the mean will have the same type as `x`. If `x` is
    integral, the output is cast to tf.float32.

    Note that TFLearn generally permits only tf.int64 and tf.float32, so casting
    this scaler's output may be necessary.
  """
  with tf.compat.v1.name_scope(name, 'scale_to_z_score'):
    return scale_to_z_score_per_key(x=x,
                                    key=None,
                                    elementwise=elementwise,
                                    name=name,
                                    output_dtype=output_dtype)


def scale_to_z_score_per_key(x, key=None, elementwise=False, name=None,
                             output_dtype=None):
  """Returns a standardized column with mean 0 and variance 1, grouped per key.

  Scaling to z-score subtracts out the mean and divides by standard deviation.
  Note that the standard deviation computed here is based on the biased variance
  (0 delta degrees of freedom), as computed by analyzers.var.

  Args:
    x: A numeric `Tensor` or `SparseTensor`.
    key: A Tensor or `SparseTensor` of dtype tf.string.
        Must meet one of the following conditions:
        0. key is None
        1. Both x and key are dense,
        2. Both x and key are sparse and `key` must exactly match `x` in
        everything except values,
        3. The axis=1 index of each x matches its index of dense key.
    elementwise: If true, scales each element of the tensor independently;
        otherwise uses the mean and variance of the whole tensor.
        Currently, not supported for per-key operations.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` or `SparseTensor` containing the input column scaled to mean 0
    and variance 1 (standard deviation 1), grouped per key if a key is provided.

    That is, for all keys k: (x - mean(x)) / std_dev(x) for all x with key k.
    If `x` is floating point, the mean will have the same type as `x`. If `x` is
    integral, the output is cast to tf.float32.

    Note that TFLearn generally permits only tf.int64 and tf.float32, so casting
    this scaler's output may be necessary.
  """
  with tf.compat.v1.name_scope(name, 'scale_to_z_score_per_key'):
    # x_mean will be float16, float32, or float64, depending on type of x

    if key is None:
      x_mean, x_var = analyzers._mean_and_var(  # pylint: disable=protected-access
          x, reduce_instance_dims=not elementwise, output_dtype=output_dtype)
    else:
      if elementwise:
        raise NotImplementedError('Per-key elementwise reduction not supported')

      key_vocab, key_means, key_vars = analyzers._mean_and_var_per_key(  # pylint: disable=protected-access
          x, key, output_dtype=output_dtype)

      x_mean, x_var = tf_utils.map_per_key_reductions(
          (key_means, key_vars), key, key_vocab, x)

    compose_result_fn = _make_sparse_tensor_wrapper_if_sparse(x)
    x_values = x

    if isinstance(x, tf.SparseTensor):
      x_values = x.values
      if elementwise:
        # Only supports SparseTensors with rank 2.
        x.get_shape().assert_has_rank(2)

        x_mean = tf.gather(x_mean, x.indices[:, 1])
        x_var = tf.gather(x_var, x.indices[:, 1])

    numerator = tf.cast(x_values, x_mean.dtype) - x_mean
    denominator = tf.sqrt(x_var)
    cond = tf.not_equal(denominator, 0)

    if cond.shape.as_list() != x_values.shape.as_list():
      # Repeats cond when necessary across the batch dimension for it to be
      # compatible with the shape of numerator.
      cond = tf.cast(
          tf.zeros_like(numerator) + tf.cast(cond, numerator.dtype),
          dtype=tf.bool)

    deviation_values = tf.where(cond, tf.divide(numerator, denominator),
                                numerator)
    return compose_result_fn(deviation_values)


def tfidf(x, vocab_size, smooth=True, name=None):
  """Maps the terms in x to their term frequency * inverse document frequency.

  The term frequency of a term in a document is calculated as
  (count of term in document) / (document size)

  The inverse document frequency of a term is, by default, calculated as
  1 + log((corpus size + 1) / (count of documents containing term + 1)).

  Example usage:
    example strings [["I", "like", "pie", "pie", "pie"], ["yum", "yum", "pie]]
    in: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                              [1, 0], [1, 1], [1, 2]],
                     values=[1, 2, 0, 0, 0, 3, 3, 0])
    out: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                      values=[1, 2, 0, 3, 0])
         SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                      values=[(1/5)*(log(3/2)+1), (1/5)*(log(3/2)+1), (3/5),
                              (2/3)*(log(3/2)+1), (1/3)]
    NOTE that the first doc's duplicate "pie" strings have been combined to
    one output, as have the second doc's duplicate "yum" strings.

  Args:
    x: A `SparseTensor` representing int64 values (most likely that are the
        result of calling string_to_int on a tokenized string).
    vocab_size: An int - the count of vocab used to turn the string into int64s
        including any OOV buckets.
    smooth: A bool indicating if the inverse document frequency should be
        smoothed. If True, which is the default, then the idf is calculated as
        1 + log((corpus size + 1) / (document frequency of term + 1)).
        Otherwise, the idf is
        1 +log((corpus size) / (document frequency of term)), which could
        result in a division by zero error.
    name: (Optional) A name for this operation.

  Returns:
    Two `SparseTensor`s with indices [index_in_batch, index_in_bag_of_words].
    The first has values vocab_index, which is taken from input `x`.
    The second has values tfidf_weight.
  """

  def _to_vocab_range(x):
    """Enforces that the vocab_ids in x are positive."""
    return tf.SparseTensor(
        indices=x.indices,
        values=tf.mod(x.values, vocab_size),
        dense_shape=x.dense_shape)

  with tf.compat.v1.name_scope(name, 'tfidf'):
    cleaned_input = _to_vocab_range(x)

    term_frequencies = _to_term_frequency(cleaned_input, vocab_size)

    count_docs_with_term_column = _count_docs_with_term(term_frequencies)
    # Expand dims to get around the min_tensor_rank checks
    sizes = tf.expand_dims(tf.shape(input=cleaned_input)[0], 0)
    # [batch, vocab] - tfidf
    tfidfs = _to_tfidf(term_frequencies,
                       analyzers.sum(count_docs_with_term_column,
                                     reduce_instance_dims=False),
                       analyzers.sum(sizes),
                       smooth)
    return _split_tfidfs_to_outputs(tfidfs)


def _split_tfidfs_to_outputs(tfidfs):
  """Splits [batch, vocab]-weight into [batch, bow]-vocab & [batch, bow]-tfidf.

  Args:
    tfidfs: the `SparseTensor` output of _to_tfidf
  Returns:
    Two `SparseTensor`s with indices [index_in_batch, index_in_bag_of_words].
    The first has values vocab_index, which is taken from input `x`.
    The second has values tfidf_weight.
  """
  # Split tfidfs tensor into [batch, dummy] -> vocab & [batch, dummy] -> tfidf
  # The "dummy" index counts from 0 to the number of unique tokens in the doc.
  # So example doc ["I", "like", "pie", "pie", "pie"], with 3 unique tokens,
  # will have "dummy" indices [0, 1, 2]. The particular dummy index that any
  # token receives is not important, only that the tfidf value and vocab index
  # have the *same* dummy index, so that feature_column can apply the weight to
  # the correct vocab item.
  dummy_index = segment_indices(tfidfs.indices[:, 0])
  out_index = tf.concat(
      [tf.expand_dims(tfidfs.indices[:, 0], 1),
       tf.expand_dims(dummy_index, 1)], 1)

  out_shape_second_dim = tf.maximum(
      tf.reduce_max(input_tensor=dummy_index), -1) + 1
  out_shape = tf.stack([tfidfs.dense_shape[0], out_shape_second_dim])
  out_shape.set_shape([2])

  de_duped_indicies_out = tf.SparseTensor(  # NOTYPO ('indices')
      indices=out_index,
      values=tfidfs.indices[:, 1],
      dense_shape=out_shape)
  de_duped_tfidf_out = tf.SparseTensor(
      indices=out_index,
      values=tfidfs.values,
      dense_shape=out_shape)
  return de_duped_indicies_out, de_duped_tfidf_out  # NOTYPO ('indices')


def _to_term_frequency(x, vocab_size):
  """Creates a SparseTensor of term frequency for every doc/term pair.

  Args:
    x : a SparseTensor of int64 representing string indices in vocab.
    vocab_size: A scalar int64 Tensor - the count of vocab used to turn the
        string into int64s including any OOV buckets.

  Returns:
    a SparseTensor with the count of times a term appears in a document at
        indices <doc_index_in_batch>, <term_index_in_vocab>,
        with size (num_docs_in_batch, vocab_size).
  """
  # Construct intermediary sparse tensor with indices
  # [<doc>, <term_index_in_doc>, <vocab_id>] and tf.ones values.
  vocab_size = tf.convert_to_tensor(value=vocab_size, dtype=tf.int64)
  split_indices = tf.cast(
      tf.split(x.indices, axis=1, num_or_size_splits=2), dtype=tf.int64)
  expanded_values = tf.cast(tf.expand_dims(x.values, 1), dtype=tf.int64)
  next_index = tf.concat(
      [split_indices[0], split_indices[1], expanded_values], axis=1)

  next_values = tf.ones_like(x.values)
  expanded_vocab_size = tf.expand_dims(vocab_size, 0)
  next_shape = tf.concat(
      [x.dense_shape, expanded_vocab_size], 0)

  next_tensor = tf.SparseTensor(
      indices=tf.cast(next_index, dtype=tf.int64),
      values=next_values,
      dense_shape=next_shape)

  # Take the intermediary tensor and reduce over the term_index_in_doc
  # dimension. This produces a tensor with indices [<doc_id>, <term_id>]
  # and values [count_of_term_in_doc] and shape batch x vocab_size
  term_count_per_doc = tf.compat.v1.sparse_reduce_sum_sparse(next_tensor, 1)

  dense_doc_sizes = tf.cast(
      tf.sparse.reduce_sum(
          tf.SparseTensor(
              indices=x.indices,
              values=tf.ones_like(x.values),
              dense_shape=x.dense_shape), 1),
      dtype=tf.float64)

  gather_indices = term_count_per_doc.indices[:, 0]
  gathered_doc_sizes = tf.gather(dense_doc_sizes, gather_indices)

  term_frequency = (
      tf.cast(term_count_per_doc.values, dtype=tf.float64) /
      tf.cast(gathered_doc_sizes, dtype=tf.float64))
  return tf.SparseTensor(
      indices=term_count_per_doc.indices,
      values=term_frequency,
      dense_shape=term_count_per_doc.dense_shape)


def _to_tfidf(term_frequency, reduced_term_freq, corpus_size, smooth):
  """Calculates the inverse document frequency of terms in the corpus.

  Args:
    term_frequency: The `SparseTensor` output of _to_term_frequency.
    reduced_term_freq: A `Tensor` of shape (vocabSize,) that represents the
        count of the number of documents with each term.
    corpus_size: A scalar count of the number of documents in the corpus.
    smooth: A bool indicating if the idf value should be smoothed. See
        tfidf_weights documentation for details.

  Returns:
    A `SparseTensor` with indices=<doc_index_in_batch>, <term_index_in_vocab>,
    values=term frequency * inverse document frequency,
    and shape=(batch, vocab_size)
  """
  # The idf tensor has shape (vocab_size,)
  if smooth:
    idf = tf.math.log((tf.cast(corpus_size, dtype=tf.float64) + 1.0) /
                      (1.0 + tf.cast(reduced_term_freq, dtype=tf.float64))) + 1
  else:
    idf = tf.math.log(
        tf.cast(corpus_size, dtype=tf.float64) /
        (tf.cast(reduced_term_freq, dtype=tf.float64))) + 1

  gathered_idfs = tf.gather(tf.squeeze(idf), term_frequency.indices[:, 1])
  tfidf_values = (tf.cast(term_frequency.values, tf.float32)
                  * tf.cast(gathered_idfs, tf.float32))

  return tf.SparseTensor(
      indices=term_frequency.indices,
      values=tfidf_values,
      dense_shape=term_frequency.dense_shape)


def _count_docs_with_term(term_frequency):
  """Computes the number of documents in a batch that contain each term.

  Args:
    term_frequency: The `SparseTensor` output of _to_term_frequency.
  Returns:
    A `Tensor` of shape (vocab_size,) that contains the number of documents in
    the batch that contain each term.
  """
  count_of_doc_inter = tf.SparseTensor(
      indices=term_frequency.indices,
      values=tf.ones_like(term_frequency.values),
      dense_shape=term_frequency.dense_shape)
  out = tf.sparse.reduce_sum(count_of_doc_inter, axis=0)
  return tf.expand_dims(out, 0)


def compute_and_apply_vocabulary(
    x,
    default_value=-1,
    top_k=None,
    frequency_threshold=None,
    num_oov_buckets=0,
    vocab_filename=None,
    weights=None,
    labels=None,
    use_adjusted_mutual_info=False,
    min_diff_from_avg=0.0,
    coverage_top_k=None,
    coverage_frequency_threshold=None,
    key_fn=None,
    fingerprint_shuffle=False,
    name=None):
  r"""Generates a vocabulary for `x` and maps it to an integer with this vocab.

  In case one of the tokens contains the '\n' or '\r' characters or is empty it
  will be discarded since we are currently writing the vocabularies as text
  files. This behavior will likely be fixed/improved in the future.

  Note that this function will cause a vocabulary to be computed.  For large
  datasets it is highly recommended to either set frequency_threshold or top_k
  to control the size of the vocabulary, and also the run time of this
  operation.

  Args:
    x: A `Tensor` or `SparseTensor` of type tf.string or tf.int[8|16|32|64].
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      absolute frequency is >= to the supplied threshold. If set to None, the
      full vocabulary is generated.  Absolute frequency means the number of
      occurences of the element in the dataset, as opposed to the proportion of
      instances that contain that element.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.
    vocab_filename: The file name for the vocabulary file. If None, a name based
      on the scope name in the context of this graph will be used as the
      file name. If not None, should be unique within a given preprocessing
      function.
      NOTE in order to make your pipelines resilient to implementation details
      please set `vocab_filename` when you are using the vocab_filename on a
      downstream component.
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
    fingerprint_shuffle: (Optional), (Experimental) Whether to sort the
      vocabularies by fingerprint instead of counts. This is useful for load
      balancing on the training parameter servers. Shuffle only happens while
      writing the files, so all the filters above will still take effect.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `SparseTensor` where each string value is mapped to an
    integer. Each unique string value that appears in the vocabulary
    is mapped to a different integer and integers are consecutive starting from
    zero. String value not in the vocabulary is assigned default_value.

  Raises:
    ValueError: If `top_k` or `frequency_threshold` is negative.
      If `coverage_top_k` or `coverage_frequency_threshold` is negative.
  """
  with tf.compat.v1.name_scope(name, 'compute_and_apply_vocabulary'):
    deferred_vocab_and_filename = analyzers.vocabulary(
        x=x,
        top_k=top_k,
        frequency_threshold=frequency_threshold,
        vocab_filename=vocab_filename,
        weights=weights,
        labels=labels,
        use_adjusted_mutual_info=use_adjusted_mutual_info,
        min_diff_from_avg=min_diff_from_avg,
        coverage_top_k=coverage_top_k,
        coverage_frequency_threshold=coverage_frequency_threshold,
        key_fn=key_fn,
        fingerprint_shuffle=fingerprint_shuffle,
        name=name)
    return apply_vocabulary(
        x, deferred_vocab_and_filename, default_value, num_oov_buckets)


@deprecation.deprecated(None,
                        'Use `tft.compute_and_apply_vocabulary()` instead.')
def string_to_int(x,
                  default_value=-1,
                  top_k=None,
                  frequency_threshold=None,
                  num_oov_buckets=0,
                  vocab_filename=None,
                  weights=None,
                  labels=None,
                  name=None):
  r"""See `tft.compute_and_apply_vocabulary`."""
  return compute_and_apply_vocabulary(
      x=x,
      default_value=default_value,
      top_k=top_k,
      frequency_threshold=frequency_threshold,
      num_oov_buckets=num_oov_buckets,
      vocab_filename=vocab_filename,
      weights=weights,
      labels=labels,
      name=name)


def apply_vocabulary(x,
                     deferred_vocab_filename_tensor,
                     default_value=-1,
                     num_oov_buckets=0,
                     lookup_fn=None,
                     name=None):
  r"""Maps `x` to a vocabulary specified by the deferred tensor.

  This function also writes domain statistics about the vocabulary min and max
  values. Note that the min and max are inclusive, and depend on the vocab size,
  num_oov_buckets and default_value.

  In case one of the tokens contains the '\n' or '\r' characters or is empty it
  will be discarded since we are currently writing the vocabularies as text
  files. This behavior will likely be fixed/improved in the future.

  Args:
    x: A categorical `Tensor` or `SparseTensor` of type tf.string or
      tf.int[8|16|32|64] to which the vocabulary transformation should be
      applied. The column names are those intended for the transformed tensors.
    deferred_vocab_filename_tensor: The deferred vocab filename tensor as
      returned by `tft.vocabulary`, as long as the frequencies were not stored.
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.
    lookup_fn: Optional lookup function, if specified it should take a tensor
      and a deferred vocab filename as an input and return a lookup `op` along
      with the table size, by default `apply_vocab` constructs a StaticHashTable
      for the table lookup.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `SparseTensor` where each string value is mapped to an
    integer. Each unique string value that appears in the vocabulary
    is mapped to a different integer and integers are consecutive
    starting from zero, and string value not in the vocabulary is
    assigned default_value.
  """
  with tf.compat.v1.name_scope(name, 'apply_vocab'):
    if x.dtype != tf.string and not x.dtype.is_integer:
      raise tf.errors.InvalidArgumentError(
          'expected tf.string or tf.int[8|16|32|64] but got %r' % x.dtype)

    if lookup_fn:
      result, table_size = lookup_fn(x, deferred_vocab_filename_tensor)
    else:
      if (deferred_vocab_filename_tensor is None
          or (isinstance(deferred_vocab_filename_tensor,
                         (six.binary_type, six.text_type))
              and not deferred_vocab_filename_tensor)):
        raise ValueError('`deferred_vocab_filename_tensor` must not be empty.')
      initializer = tf.lookup.TextFileInitializer(
          deferred_vocab_filename_tensor,
          key_dtype=x.dtype,
          key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
          value_dtype=tf.int64,
          value_index=tf.lookup.TextFileIndex.LINE_NUMBER)

      if num_oov_buckets > 0:
        table = tf.lookup.StaticVocabularyTable(initializer,
                                                num_oov_buckets=num_oov_buckets,
                                                lookup_key_dtype=x.dtype)
      else:
        table = tf.lookup.StaticHashTable(initializer,
                                          default_value=default_value)
      table_size = table.size()
      result = table.lookup(x)

    # Specify schema overrides which will override the values in the schema
    # with the min and max values, which are deferred as they are only known
    # once the analyzer has run.
    #
    # `table_size` includes the num oov buckets.  The default value is only used
    # if num_oov_buckets <= 0.
    min_value = tf.constant(0, tf.int64)
    max_value = table_size - 1
    if num_oov_buckets <= 0:
      min_value = tf.minimum(min_value, default_value)
      max_value = tf.maximum(max_value, default_value)
    schema_inference.set_tensor_schema_override(
        result.values if isinstance(result, tf.SparseTensor) else result,
        min_value, max_value)

    return result


@deprecation.deprecated(None, 'Use `tft.apply_vocabulary()` instead.')
def apply_vocab(x,
                deferred_vocab_filename_tensor,
                default_value=-1,
                num_oov_buckets=0,
                lookup_fn=None,
                name=None):
  r"""See `tft.apply_vocabulary`."""
  return apply_vocabulary(
      x=x,
      deferred_vocab_filename_tensor=deferred_vocab_filename_tensor,
      default_value=default_value,
      num_oov_buckets=num_oov_buckets,
      lookup_fn=lookup_fn,
      name=name)


def segment_indices(segment_ids, name=None):
  """Returns a `Tensor` of indices within each segment.

  segment_ids should be a sequence of non-decreasing non-negative integers that
  define a set of segments, e.g. [0, 0, 1, 2, 2, 2] defines 3 segments of length
  2, 1 and 3.  The return value is a `Tensor` containing the indices within each
  segment.

  Example input: [0, 0, 1, 2, 2, 2]
  Example output: [0, 1, 0, 0, 1, 2]

  Args:
    segment_ids: A 1-d `Tensor` containing an non-decreasing sequence of
        non-negative integers with type `tf.int32` or `tf.int64`.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the indices within each segment.
  """
  with tf.compat.v1.name_scope(name, 'segment_indices'):
    # TODO(KesterTong): This is a fundamental operation for segments, write a C++
    # op to do this.
    # TODO(KesterTong): Add a check that segment_ids are increasing.
    segment_lengths = tf.math.segment_sum(
        tf.ones_like(segment_ids), segment_ids)
    segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                               segment_ids)
    return (tf.range(tf.size(input=segment_ids, out_type=segment_ids.dtype)) -
            segment_starts)


def deduplicate_tensor_per_row(input_tensor, name=None):
  """Deduplicates each row (0-th dimension) of the provided tensor.

  Args:
    input_tensor: A two-dimensional `Tensor` or `SparseTensor`. The first
      dimension is assumed to be the batch or "row" dimension, and deduplication
      is done on the 2nd dimension. If the Tensor is 1D it is returned as the
      equivalent `SparseTensor` since the "row" is a scalar can't be further
      deduplicated.
    name: Optional name for the operation.

  Returns:
    A  `SparseTensor` containing the unique set of values from each
      row of the input. Note: the original order of the input may not be
      preserved.
  """
  with tf.compat.v1.name_scope(name, 'deduplicate_per_row'):

    if isinstance(input_tensor, tf.SparseTensor):
      batch_dim = tf.cast(input_tensor.dense_shape[0], tf.int32)
      rank = input_tensor.dense_shape.shape[0]
    else:
      batch_dim = tf.cast(tf.shape(input_tensor)[0], tf.int32)
      rank = input_tensor.shape.rank

    def _univalent_dense_to_sparse(batch_dim, input_tensor):
      """Helper to convert a 1D dense `Tensor` to a `SparseTensor`."""
      indices = tf.cast(
          tf.stack([
              tf.range(batch_dim, dtype=tf.int32),
              tf.zeros(batch_dim, dtype=tf.int32)
          ],
                   axis=1),
          dtype=tf.int64)

      return tf.SparseTensor(
          indices=indices, values=input_tensor, dense_shape=(batch_dim, 1))

    if rank is not None:
      # If the rank is known at graph construction time, and it's rank 1, there
      # is no deduplication to be done so we can return early.
      if rank <= 1:
        if isinstance(input_tensor, tf.SparseTensor):
          return input_tensor
        # Even though we are just returning as is, we convert to a SparseTensor
        # to ensure consistent output type.
        return _univalent_dense_to_sparse(batch_dim, input_tensor)
      if rank > 2:
        raise ValueError(
            'Deduplication assumes a rank 2 tensor, got {}.'.format(rank))
      return _deduplicate_tensor_per_row(input_tensor, batch_dim)

    if isinstance(input_tensor, tf.SparseTensor):
      return _deduplicate_tensor_per_row(input_tensor, batch_dim)
    else:
      # Again check for rank 1 tensor (that doesn't need deduplication), this
      # time handling inputs where rank isn't known until execution time.
      dynamic_rank = tf.rank(input_tensor)
      return tf.cond(
          tf.equal(dynamic_rank, 1),
          lambda: _univalent_dense_to_sparse(batch_dim, input_tensor),
          lambda: _deduplicate_tensor_per_row(input_tensor, batch_dim),
      )


_DedupRowLoopArgs = collections.namedtuple(
    'DedupRowLoopArgs',
    [
        'index',  # Index representing the row of input_tensor to be processed.
        'input_tensor',  # `Tensor` or `SparseTensor` to be deuplicated per row.
        'indices',  # `TensorArray` containing indices of each deduplicated row.
        'values',  # `TensorArray` containing values of each deduplicated row.
        'max_unique',  # Tracks the maximum size of any row.
    ])


class _DedupRowLoopVars(_DedupRowLoopArgs):
  """Loop variables for _deduplicate_per_row."""
  pass


def _deduplicate_tensor_per_row(input_tensor, batch_dim):
  """Helper function for deduplicating each row of the provided tensor.

  For each input row, computes the unique values and set them in positions 0
  through num_unique - 1 within the row.

  Args:
    input_tensor: A `Tensor` or `SparseTensor` to be deuplicated per row.
    batch_dim: The batch dimension or number of "rows" in the batch.

  Returns:
    A  `SparseTensor` containing the unique set of values from each
      row of the input. Note: the original order of the input may not be
      preserved.
  """
  max_unique = tf.constant(0, dtype=tf.int64)
  values = tf.TensorArray(
      size=batch_dim,
      dtype=input_tensor.dtype,
      element_shape=[None],
      infer_shape=False)
  indices = tf.TensorArray(
      size=batch_dim,
      dtype=tf.int64,
      element_shape=[None, 2],
      infer_shape=False)

  def _deduplicate_row(dedup_row_loop_vars):
    """Deduplicates the values in the i-th row of the input.

    Args:
      dedup_row_loop_vars: A _DedupRowLoopVars NamedTuple.

    Returns:
      Updated version of the _DedupRowLoopVars for the loop iteration.
    """
    index, input_tensor, indices, values, max_unique = dedup_row_loop_vars
    if isinstance(input_tensor, tf.SparseTensor):

      row = tf.sparse.slice(input_tensor, [index, 0],
                            [1, input_tensor.dense_shape[1]])
      row_values, _ = tf.unique(row.values)
    else:
      row = input_tensor[index]
      row_values, _ = tf.unique(row)

    # Keep track of the maximum number of unique elements in a row, as this
    # will determine the resulting dense shape.
    max_unique = tf.cast(
        tf.maximum(tf.cast(tf.shape(row_values)[0], tf.int64), max_unique),
        tf.int64)
    column_indices = tf.cast(
        tf.expand_dims(tf.range(tf.shape(row_values)[0]), axis=1), tf.int64)
    row_indices = tf.fill(tf.shape(column_indices), tf.cast(index, tf.int64))
    values = values.write(index, row_values)
    indices = indices.write(index, tf.concat([row_indices, column_indices], 1))
    return [
        _DedupRowLoopVars(index + 1, input_tensor, indices, values, max_unique)
    ]

  index = tf.constant(0, tf.int32)
  (loop_output,) = tf.while_loop(
      lambda loop_args: loop_args.index < batch_dim,
      _deduplicate_row,
      [_DedupRowLoopVars(index, input_tensor, indices, values, max_unique)],
      back_prop=False)

  dense_shape = tf.convert_to_tensor(
      [tf.cast(batch_dim, tf.int64),
       tf.cast(loop_output.max_unique, tf.int64)],
      dtype=tf.int64)
  return tf.SparseTensor(
      indices=tf.cast(loop_output.indices.concat(), tf.int64),
      values=loop_output.values.concat(),
      dense_shape=dense_shape)


def bag_of_words(tokens, ngram_range, separator, name=None):
  """Computes a bag of "words" based on the specified ngram configuration.

  A light wrapper around tft.ngrams. First computes ngrams, then transforms the
  ngram representation (list semantics) into a Bag of Words (set semantics) per
  row. Each row reflects the set of *unique* ngrams present in an input record.

  See tft.ngrams for more information.

  Args:
    tokens: a two-dimensional `SparseTensor` of dtype `tf.string` containing
      tokens that will be used to construct a bag of words.
    ngram_range: A pair with the range (inclusive) of ngram sizes to compute.
    separator: a string that will be inserted between tokens when ngrams are
      constructed.
    name: (Optional) A name for this operation.

  Returns:
    A `SparseTensor` containing the unique set of ngrams from each row of the
      input. Note: the original order of the ngrams may not be preserved.
  """
  with tf.compat.v1.name_scope(name, 'bag_of_words'):
    # First compute the ngram representation, which will contain ordered and
    # possibly duplicated ngrams per row.
    all_ngrams = ngrams(tokens, ngram_range, separator)
    # Then deduplicate the ngrams in each row.
    return deduplicate_tensor_per_row(all_ngrams)


def ngrams(tokens, ngram_range, separator, name=None):
  """Create a `SparseTensor` of n-grams.

  Given a `SparseTensor` of tokens, returns a `SparseTensor` containing the
  ngrams that can be constructed from each row.

  `separator` is inserted between each pair of tokens, so " " would be an
  appropriate choice if the tokens are words, while "" would be an appropriate
  choice if they are characters.

  Example:

  `tokens` is a `SparseTensor` with

  indices = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]]
  values = ['One', 'was', 'Johnny', 'Two', 'was', 'a', 'rat']
  dense_shape = [2, 4]

  If we set
  ngrams_range = (1,3)
  separator = ' '

  output is a `SparseTensor` with

  indices = [[0, 0], [0, 1], [0, 2], ..., [1, 6], [1, 7], [1, 8]]
  values = ['One', 'One was', 'One was Johnny', 'was', 'was Johnny', 'Johnny',
            'Two', 'Two was', 'Two was a', 'was', 'was a', 'was a rat', 'a',
            'a rat', 'rat']
  dense_shape = [2, 9]

  Args:
    tokens: a two-dimensional`SparseTensor` of dtype `tf.string` containing
      tokens that will be used to construct ngrams.
    ngram_range: A pair with the range (inclusive) of ngram sizes to return.
    separator: a string that will be inserted between tokens when ngrams are
      constructed.
    name: (Optional) A name for this operation.

  Returns:
    A `SparseTensor` containing all ngrams from each row of the input. Note:
    if an ngram appears multiple times in the input row, it will be present the
    same number of times in the output. For unique ngrams, see tft.bag_of_words.

  Raises:
    ValueError: if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]
  """
  # This function is implemented as follows.  Assume we start with the following
  # `SparseTensor`:
  #
  # indices=[[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [2, 1], [2, 2]]
  # values=['a', 'b', 'c', 'd', 'q', 'x', 'y', 'z']
  # dense_shape=[3, 4]
  #
  # First we then create shifts of the values and first column of indices,
  # buffering to avoid overrunning the end of the array, so the shifted values
  # (if we are ngrams up to size 3) are
  #
  # shifted_batch_indices[0]=[0, 0, 0, 0, 1, 2, 2, 2]
  # shifted_tokens[0]=['a', 'b', 'c', 'd', 'q', 'x', 'y', 'z']
  #
  # shifted_batch_indices[1]=[0, 0, 0, 1, 2, 2, 2, -1]
  # shifted_tokens[1]=['b', 'c', 'd', 'q', 'x', 'y', 'z', '']
  #
  # shifted_batch_indices[2]=[0, 0, 1, 2, 2, 2, -1, -1]
  # shifted_tokens[2]=['c', 'd', 'q', 'x', 'y', 'z', '', '']
  #
  # These shifted ngrams are used to create the ngrams as follows.  We use
  # tf.string_join to join shifted_tokens[:k] to create k-grams. The `separator`
  # string is inserted between each pair of tokens in the k-gram.
  # The batch that the first of these belonged to is given by
  # shifted_batch_indices[0]. However some of these will cross the boundaries
  # between 'batches' and so we we create a boolean mask which is True when
  # shifted_indices[:k] are all equal.
  #
  # This results in tensors of ngrams, their batch indices and a boolean mask,
  # which we then use to construct the output SparseTensor.
  with tf.compat.v1.name_scope(name, 'ngrams'):
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
      raise ValueError('Invalid ngram_range: %r' % (ngram_range,))

    def _sliding_windows(values, num_shifts, fill_value):
      buffered_values = tf.concat(
          [values, tf.fill([num_shifts - 1], fill_value)], 0)
      return [
          tf.slice(buffered_values, [i], tf.shape(input=values))
          for i in range(num_shifts)
      ]

    shifted_batch_indices = _sliding_windows(
        tokens.indices[:, 0], ngram_range[1] + 1,
        tf.constant(-1, dtype=tf.int64))
    shifted_tokens = _sliding_windows(tokens.values, ngram_range[1] + 1, '')

    # Construct a tensor of the form
    # [['a', 'ab, 'abc'], ['b', 'bcd', cde'], ...]
    def _string_join(tensors):
      if tensors:
        return tf.strings.join(tensors, separator=separator)
      else:
        return

    ngrams_array = [_string_join(shifted_tokens[:k])
                    for k in range(ngram_range[0], ngram_range[1] + 1)]
    ngrams_tensor = tf.stack(ngrams_array, 1)

    # Construct a boolean mask for whether each ngram in ngram_tensor is valid,
    # in that each character came from the same batch.
    valid_ngram = tf.equal(
        tf.math.cumprod(
            tf.cast(
                tf.equal(
                    tf.stack(shifted_batch_indices, 1),
                    tf.expand_dims(shifted_batch_indices[0], 1)),
                dtype=tf.int32),
            axis=1), 1)
    valid_ngram = valid_ngram[:, (ngram_range[0] - 1):ngram_range[1]]

    # Construct a tensor with the batch that each ngram in ngram_tensor belongs
    # to.
    batch_indices = tf.tile(tf.expand_dims(tokens.indices[:, 0], 1),
                            [1, ngram_range[1] + 1 - ngram_range[0]])

    # Apply the boolean mask and construct a SparseTensor with the given indices
    # and values, where another index is added to give the position within a
    # batch.
    batch_indices = tf.boolean_mask(tensor=batch_indices, mask=valid_ngram)
    ngrams_tensor = tf.boolean_mask(tensor=ngrams_tensor, mask=valid_ngram)
    instance_indices = segment_indices(batch_indices)
    dense_shape_second_dim = tf.maximum(
        tf.reduce_max(input_tensor=instance_indices), -1) + 1
    return tf.SparseTensor(
        indices=tf.stack([batch_indices, instance_indices], 1),
        values=ngrams_tensor,
        dense_shape=tf.stack(
            [tokens.dense_shape[0], dense_shape_second_dim]))


def word_count(tokens, name=None):
  """Find the token count of each document/row.

  `tokens` is either a `RaggedTensor` or `SparseTensor`, representing tokenized
  strings. This function simply returns size of each row, so the dtype is not
  constrained to string.

  Args:
    tokens: either
      (1) a two-dimensional `SparseTensor`, or
      (2) a `RaggedTensor` with ragged rank of 1, non-ragged rank of 1
      of dtype `tf.string` containing tokens to be counted
    name: (Optional) A name for this operation.

  Returns:
    A one-dimensional `Tensor` the token counts of each row.

  Raises:
    ValueError: if tokens is neither sparse nor ragged
  """
  with tf.compat.v1.name_scope(name, 'word_count'):
    if isinstance(tokens, tf.RaggedTensor):
      return tokens.row_lengths()
    elif isinstance(tokens, tf.SparseTensor):
      result = tf.sparse.reduce_sum(
          tf.SparseTensor(indices=tokens.indices,
                          values=tf.ones_like(tokens.values, dtype=tf.int64),
                          dense_shape=tokens.dense_shape),
          axis=1)
      result.set_shape([tokens.shape[0]])
      return result
    else:
      raise ValueError('Invalid token tensor')


def hash_strings(strings, hash_buckets, key=None, name=None):
  """Hash strings into buckets.

  Args:
    strings: a `Tensor` or `SparseTensor` of dtype `tf.string`.
    hash_buckets: the number of hash buckets.
    key: optional. An array of two Python `uint64`. If passed, output will be
      a deterministic function of `strings` and `key`. Note that hashing will be
      slower if this value is specified.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `SparseTensor` of dtype `tf.int64` with the same shape as the
    input `strings`.

  Raises:
    TypeError: if `strings` is not a `Tensor` or `SparseTensor` of dtype
    `tf.string`.
  """
  if (not isinstance(strings, (tf.Tensor,
                               tf.SparseTensor))) or strings.dtype != tf.string:
    raise TypeError(
        'Input to hash_strings must be a Tensor or SparseTensor of dtype '
        'string; got {}'.
        format(strings.dtype))
  if isinstance(strings, tf.SparseTensor):
    return tf.SparseTensor(indices=strings.indices,
                           values=hash_strings(
                               strings.values, hash_buckets, key),
                           dense_shape=strings.dense_shape)
  if name is None:
    name = 'hash_strings'
  if key is None:
    return tf.strings.to_hash_bucket_fast(strings, hash_buckets, name=name)
  return tf.strings.to_hash_bucket_strong(strings, hash_buckets, key, name=name)


def bucketize(x, num_buckets, epsilon=None, weights=None, elementwise=False,
              always_return_num_quantiles=False, name=None):
  """Returns a bucketized column, with a bucket index assigned to each input.

  Args:
    x: A numeric input `Tensor` or `SparseTensor` whose values should be mapped
      to buckets.  For a `SparseTensor` only non-missing values will be included
      in the quantiles computation, and the result of `bucketize` will be a
      `SparseTensor` with non-missing values mapped to buckets.
    num_buckets: Values in the input `x` are divided into approximately
      equal-sized buckets, where the number of buckets is num_buckets.
      This is a hint. The actual number of buckets computed can be
      less or more than the requested number. Use the generated metadata to
      find the computed number of buckets.
    epsilon: (Optional) Error tolerance, typically a small fraction close to
      zero. If a value is not specified by the caller, a suitable value is
      computed based on experimental results.  For `num_buckets` less
      than 100, the value of 0.01 is chosen to handle a dataset of up to
      ~1 trillion input data values.  If `num_buckets` is larger,
      then epsilon is set to (1/`num_buckets`) to enforce a stricter
      error tolerance, because more buckets will result in smaller range for
      each bucket, and so we want the boundaries to be less fuzzy.
      See analyzers.quantiles() for details.
    weights: (Optional) Weights tensor for the quantiles. Tensor must have the
      same shape as x.
    elementwise: (Optional) If true, bucketize each element of the tensor
      independently.
    always_return_num_quantiles: (Optional) A bool that determines whether the
      exact num_buckets should be returned (defaults to False for now, but will
      be changed to True in an imminent update).
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` of the same shape as `x`, with each element in the
    returned tensor representing the bucketized value. Bucketized value is
    in the range [0, actual_num_buckets). Sometimes the actual number of buckets
    can be different than num_buckets hint, for example in case the number of
    distinct values is smaller than num_buckets, or in cases where the
    input values are not uniformly distributed.

  Raises:
    ValueError: If value of num_buckets is not > 1.
  """
  # TODO(b/137963802): Make always_return_num_quantiles default to True
  with tf.compat.v1.name_scope(name, 'bucketize'):
    if not isinstance(num_buckets, int):
      raise TypeError('num_buckets must be an int, got %s' % type(num_buckets))

    if num_buckets < 1:
      raise ValueError('Invalid num_buckets %d' % num_buckets)

    if epsilon is None:
      # See explanation in args documentation for epsilon.
      epsilon = min(1.0 / num_buckets, 0.01)

    x_values = x.values if isinstance(x, tf.SparseTensor) else x
    bucket_boundaries = analyzers.quantiles(
        x_values, num_buckets, epsilon, weights,
        reduce_instance_dims=not elementwise,
        always_return_num_quantiles=always_return_num_quantiles)

    if not elementwise:
      return apply_buckets(x, bucket_boundaries)

    num_features = tf.math.reduce_prod(x.get_shape()[1:])
    bucket_boundaries = tf.reshape(bucket_boundaries, [num_features, -1])
    x_reshaped = tf.reshape(x, [-1, num_features])
    bucketized = []
    for idx, boundaries in enumerate(tf.unstack(bucket_boundaries, axis=0)):
      bucketized.append(apply_buckets(x_reshaped[:, idx],
                                      tf.expand_dims(boundaries, axis=0)))
    return tf.reshape(tf.stack(bucketized, axis=1),
                      [-1] + x.get_shape().as_list()[1:])


def bucketize_per_key(x, key, num_buckets, epsilon=None, name=None):
  """Returns a bucketized column, with a bucket index assigned to each input.

  Args:
    x: A numeric input `Tensor` or `SparseTensor` with rank 1, whose values
      should be mapped to buckets.  `SparseTensor`s will have their non-missing
      values mapped and missing values left as missing.
    key: A Tensor or `SparseTensor` with the same shape as `x` and dtype
      tf.string.  If `x` is a `SparseTensor`, `key` must exactly match `x` in
      everything except values, i.e. indices and dense_shape must be identical.
    num_buckets: Values in the input `x` are divided into approximately
      equal-sized buckets, where the number of buckets is num_buckets.
    epsilon: (Optional) see `bucketize`
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` of the same shape as `x`, with each element in the
    returned tensor representing the bucketized value. Bucketized value is
    in the range [0, actual_num_buckets).

  Raises:
    ValueError: If value of num_buckets is not > 1.
  """
  with tf.compat.v1.name_scope(name, 'bucketize_per_key'):
    if not isinstance(num_buckets, int):
      raise TypeError(
          'num_buckets must be an int, got {}'.format(type(num_buckets)))

    if num_buckets < 1:
      raise ValueError('Invalid num_buckets {}'.format(num_buckets))

    if epsilon is None:
      # See explanation in args documentation for epsilon.
      epsilon = min(1.0 / num_buckets, 0.01)

    (key_vocab, bucket_boundaries, scale_factor_per_key, shift_per_key,
     actual_num_buckets) = (
         analyzers._quantiles_per_key(  # pylint: disable=protected-access
             x.values if isinstance(x, tf.SparseTensor) else x,
             key.values if isinstance(key, tf.SparseTensor) else key,
             num_buckets, epsilon))
    return _apply_buckets_with_keys(x, key, key_vocab, bucket_boundaries,
                                    scale_factor_per_key, shift_per_key,
                                    actual_num_buckets)


def _make_sparse_tensor_wrapper_if_sparse(x):
  if not isinstance(x, tf.SparseTensor):
    return lambda values: values
  return (lambda values: tf.SparseTensor(  # pylint: disable=g-long-lambda
      indices=x.indices, values=values, dense_shape=x.dense_shape))


def _apply_buckets_with_keys(x,
                             key,
                             key_vocab,
                             bucket_boundaries,
                             scale_factor_per_key,
                             shift_per_key,
                             num_buckets,
                             name=None):
  """Bucketize a Tensor or SparseTensor where boundaries depend on the index.

  Args:
    x: A 1-d Tensor or SparseTensor.
    key: A 1-d Tensor or SparseTensor with the same size as x.
    key_vocab: A vocab containing all keys.  Must be exhaustive, an
        out-of-vocab entry in `key` will cause a crash.
    bucket_boundaries: A rank-1 Tensor.
    scale_factor_per_key: A rank-1 Tensor of shape (key_size,).
    shift_per_key: A rank-1 Tensor of shape (key_size,).
    num_buckets: A scalar.
    name: (Optional) A name for this operation.

  Returns:
    A tensor with the same shape as `x` and dtype tf.int64
  """
  with tf.compat.v1.name_scope(name, 'apply_buckets_with_keys'):
    x_values = x.values if isinstance(x, tf.SparseTensor) else x
    key_values = key.values if isinstance(key, tf.SparseTensor) else key

    x_values = tf.cast(x_values, tf.float32)
    # Convert `key_values` to indices in key_vocab.  We must use apply_function
    # since this uses a Table.
    key_indices = tf_utils.lookup_key(key_values, key_vocab)

    # Apply the per-key offsets to x, which produces offset buckets (where the
    # bucket offset is an integer offset).  Then remove this offset to get the
    # actual per-key buckets for x.
    scale_factors = tf.gather(scale_factor_per_key, key_indices)
    shifts = tf.gather(shift_per_key, key_indices)

    transformed_x = x_values * scale_factors + shifts
    offset_buckets = tf.cast(
        quantile_ops.bucketize_with_input_boundaries(
            transformed_x, tf.cast(bucket_boundaries, tf.float32)),
        dtype=tf.int64)

    max_bucket = num_buckets - 1

    # Shift the bucket numbers back to the correct range [0, num_buckets].
    # We use max_bucket-1 due to different keys sharing 1 boundary.
    corrected_buckets = offset_buckets - ((max_bucket - 1) * key_indices)
    bucketized_values = tf.clip_by_value(corrected_buckets, 0, max_bucket)

    # Attach the relevant metadata to result, so that the corresponding
    # output feature will have this metadata set.
    min_value = tf.constant(0, tf.int64)
    schema_inference.set_tensor_schema_override(
        bucketized_values, min_value, max_bucket)

    if isinstance(x, tf.SparseTensor):
      result = tf.SparseTensor(x.indices, bucketized_values, x.dense_shape)
    else:
      result = bucketized_values

    return result


def apply_buckets_with_interpolation(x, bucket_boundaries, name=None):
  """Interpolates within the provided buckets and then normalizes to 0 to 1.

  A method for normalizing continuous numeric data to the range [0, 1].
  Numeric values are first bucketized according to the provided boundaries, then
  linearly interpolated within their respective bucket ranges. Finally, the
  interpolated values are normalized to the range [0, 1]. Values that are
  less than or equal to the lowest boundary, or greater than or equal to the
  highest boundary, will be mapped to 0 and 1 respectively.

  This is a non-linear approach to normalization that is less sensitive to
  outliers than min-max or z-score scaling. When outliers are present, standard
  forms of normalization can leave the majority of the data compressed into a
  very small segment of the output range, whereas this approach tends to spread
  out the more frequent values (if quantile buckets are used). Note that
  distance relationships in the raw data are not necessarily preserved (data
  points that close to each other in the raw feature space may not be equally
  close in the transformed feature space). This means that unlike linear
  normalization methods, correlations between features may be distorted by the
  transformation.

  Args:
    x: A numeric input `Tensor`/`SparseTensor` (tf.float[32|64], tf.int[32|64])
    bucket_boundaries: Sorted bucket boundaries as a rank-2 `Tensor`.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `SparseTensor` of the same shape as `x`, normalized to the
      range [0, 1]. If the input x is tf.float64, the returned values will be
      tf.float64. Otherwise, returned values are tf.float32.

  """
  with tf.compat.v1.name_scope(name, 'buckets_with_interpolation'):
    tf.assert_rank(bucket_boundaries, 2)
    x_values = x
    compose_result_fn = _make_sparse_tensor_wrapper_if_sparse(x)
    if isinstance(x, tf.SparseTensor):
      x_values = x.values
    if not check_ops.is_numeric_tensor(x_values):
      raise ValueError(
          'Input tensor to be normalized must be numeric, got {}.'.format(
              x_values.dtype))
    return_type = tf.float64 if x.dtype == tf.float64 else tf.float32
    num_boundaries = tf.to_int64(tf.shape(bucket_boundaries)[1])

    # The TF BucketizeWithInputBoundaries Op expects boundaries as tf.float32.
    bucket_boundaries = tf.cast(bucket_boundaries, tf.float32)
    bucket_indices = tf.cast(
        quantile_ops.bucketize_with_input_boundaries(
            x_values, boundaries=bucket_boundaries, name='assign_buckets'),
        tf.int64)

    # Get max, min, and width of the corresponding bucket for each element.
    bucket_max = tf.cast(
        tf.gather(
            tf.concat([bucket_boundaries[0], bucket_boundaries[:, -1]], axis=0),
            bucket_indices), return_type)
    bucket_min = tf.cast(
        tf.gather(
            tf.concat([bucket_boundaries[:, 0], bucket_boundaries[0]], axis=0),
            bucket_indices), return_type)
    bucket_width = bucket_max - bucket_min
    zeros = tf.zeros_like(x_values, dtype=return_type)
    ones = tf.ones_like(x_values, dtype=return_type)

    # Linearly interpolate each value within its respective bucket range.
    interpolation_value = (
        (tf.cast(x_values, return_type) - bucket_min) / bucket_width)
    bucket_interpolation = tf.verify_tensor_all_finite(
        tf.where(
            # If bucket index is first or last, which represents "less than
            # min" and "greater than max" respectively, the bucket logically
            # has an infinite width and we can't meaningfully interpolate.
            tf.logical_or(
                tf.equal(bucket_indices, 0),
                tf.equal(bucket_indices, num_boundaries)),
            zeros,
            tf.where(
                # If the bucket width is zero due to numerical imprecision,
                # there is no point in interpolating
                tf.equal(bucket_width, 0.0),
                ones / 2.0,
                # Finally, for a bucket with a valid width, we can interpolate.
                interpolation_value)),
        'bucket_interpolation')
    bucket_indices_with_interpolation = tf.cast(
        tf.maximum(bucket_indices - 1, 0), return_type) + bucket_interpolation

    # Normalize the interpolated values to the range [0, 1].
    denominator = tf.cast(tf.maximum(num_boundaries - 1, 1), return_type)
    normalized_values = tf.div(bucket_indices_with_interpolation, denominator)
    # If there is only one boundary, all values < the boundary are 0, all values
    # >= the boundary are 1.
    single_boundary_values = lambda: tf.where(  # pylint: disable=g-long-lambda
        tf.equal(bucket_indices, 0), zeros, ones)
    normalized_result = tf.cond(
        tf.equal(num_boundaries, 1),
        single_boundary_values, lambda: normalized_values)
    return compose_result_fn(normalized_result)


def apply_buckets(x, bucket_boundaries, name=None):
  """Returns a bucketized column, with a bucket index assigned to each input.

  Args:
    x: A numeric input `Tensor` or `SparseTensor` whose values should be mapped
        to buckets.  For `SparseTensor`s, the non-missing values will be mapped
        to buckets and missing value left missing.
    bucket_boundaries: The bucket boundaries represented as a rank 2 `Tensor`.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` of the same shape as `x`, with each element in the
    returned tensor representing the bucketized value. Bucketized value is
    in the range [0, len(bucket_boundaries)].
  """
  tf.compat.v1.assert_rank(bucket_boundaries, 2)
  with tf.compat.v1.name_scope(name, 'apply_buckets'):
    x_values = x.values if isinstance(x, tf.SparseTensor) else x
    buckets = quantile_ops.bucketize_with_input_boundaries(
        x_values, boundaries=bucket_boundaries, name='assign_buckets')
    # Convert to int64 because int32 is not compatible with tf.Example parser.
    # See _TF_EXAMPLE_ALLOWED_TYPES in FixedColumnRepresentation()
    # in tf_metadata/dataset_schema.py
    bucketized_values = tf.cast(buckets, dtype=tf.int64)

    # Attach the relevant metadata to result, so that the corresponding
    # output feature will have this metadata set.
    min_value = tf.constant(0, tf.int64)
    max_value = tf.shape(input=bucket_boundaries)[1]
    schema_inference.set_tensor_schema_override(
        bucketized_values, min_value, max_value)
    _annotate_buckets(bucketized_values, bucket_boundaries)

    if isinstance(x, tf.SparseTensor):
      result = tf.SparseTensor(x.indices, bucketized_values, x.dense_shape)
    else:
      result = bucketized_values
    return result


def _annotate_buckets(x, bucket_boundaries):
  """Annotates a bucketized tensor with the boundaries that were applied.

  Creates a deferred annotation for the specified tensor.

  Args:
    x: The tensor to annotate.
    bucket_boundaries: A tensor of boundaries that were used to bucketize x.
  """
  # The annotations proto currently isn't available in OSS builds, so schema
  # annotations are not supported.
  try:
    message_type = annotations_pb2.BucketBoundaries.DESCRIPTOR.full_name
  except NameError:
    return
  # The BucketBoundaries annotation expects a float field.
  bucket_boundaries = tf.cast(bucket_boundaries, tf.float32)
  # Some callers provide rank 2 boundaries like [[.25], [.5], [.75], [1.]],
  # whereas we expect rank 2 boundaries like [[.25, .5, .75, 1.]]
  bucket_boundaries = tf.reshape(bucket_boundaries, [-1])
  bucket_boundaries = tf.expand_dims(bucket_boundaries, 0)
  size = (tf.shape(bucket_boundaries)[1],)
  message_proto = tf.raw_ops.EncodeProto(sizes=[size],
                                         values=[bucket_boundaries],
                                         field_names=['boundaries'],
                                         message_type=message_type)
  assert message_proto.shape == [1]
  message_proto = message_proto[0]

  type_url = os.path.join(schema_inference.ANNOTATION_PREFIX_URL, message_type)
  schema_inference.annotate(type_url, message_proto, tensor=x)
