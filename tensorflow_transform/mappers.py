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
"""Helper functions built on top of TF.Transform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import api

from tensorflow.contrib import lookup


def scale_by_min_max(x, output_min=0.0, output_max=1.0):
  """Scale a numerical column into the range [output_min, output_max].

  Args:
    x: A `Column` representing a numeric value.
    output_min: The minimum of the range of output values.
    output_max: The maximum of the range of output values.

  Returns:
    A `Column` representing the input column scaled to [output_min, output_max].

  Raises:
    ValueError: If output_min, output_max have the wrong order.
  """
  if output_min >= output_max:
    raise ValueError('output_min must be less than output_max')

  # A TITO function that scales x.
  def _scale(x, min_x_value, max_x_value):
    return ((((x - min_x_value) * (output_max - output_min)) /
             (max_x_value - min_x_value)) + output_min)

  return api.map(_scale, x, analyzers.min(x), analyzers.max(x))


def scale_to_0_1(x):
  """Returns a column which is the input column scaled to have range [0,1].

  Args:
    x: A `Column` representing a numeric value.

  Returns:
    A `Column` representing the input column scaled to [0, 1].
  """
  return scale_by_min_max(x, 0, 1)


def tfidf_weights(x, vocab_size):
  """Maps the terms in x to their (1/doc_length) * inverse document frequency.

  Args:
    x: A `Column` representing int64 values (most likely that are the result
        of calling string_to_int on a tokenized string).
    vocab_size: An int - the count of vocab used to turn the string into int64s
        including any OOV buckets

  Returns:
    A `Column` where each int value is mapped to a double equal to
    (1 if that term appears in that row, 0 otherwise / the number of terms in
    that row) * the log of (the number of rows in `x` / (1 + the number of
    rows in `x` where the term appears at least once))

  NOTE:
    This is intented to be used with the feature_column 'sum' combiner to arrive
    at the true term frequncies.
  """

  def _map_to_vocab_range(x):
    """Enforces that the vocab_ids in x are positive."""
    return tf.SparseTensor(
        indices=x.indices,
        values=tf.mod(x.values, vocab_size),
        dense_shape=x.dense_shape)

  def _map_to_doc_contains_term(x):
    """Creates a SparseTensor with 1s at every doc/term pair index.

    Args:
      x : a SparseTensor of int64 representing string indices in vocab.

    Returns:
      a SparseTensor with 1s at indices <doc_index_in_batch>,
          <term_index_in_vocab> for every term/doc pair.
    """
    # Construct intermediary sparse tensor with indices
    # [<doc>, <term_index_in_doc>, <vocab_id>] and tf.ones values.
    split_indices = tf.to_int64(
        tf.split(x.indices, axis=1, num_or_size_splits=2))
    expanded_values = tf.to_int64(tf.expand_dims(x.values, 1))
    next_index = tf.concat(
        [split_indices[0], split_indices[1], expanded_values], axis=1)

    next_values = tf.ones_like(x.values)
    vocab_size_as_tensor = tf.constant([vocab_size], dtype=tf.int64)
    next_shape = tf.concat(
        [x.dense_shape, vocab_size_as_tensor], 0)

    next_tensor = tf.SparseTensor(
        indices=tf.to_int64(next_index),
        values=next_values,
        dense_shape=next_shape)

    # Take the intermediar tensor and reduce over the term_index_in_doc
    # dimension. This produces a tensor with indices [<doc_id>, <term_id>]
    # and values [count_of_term_in_doc] and shape batch x vocab_size
    term_count_per_doc = tf.sparse_reduce_sum_sparse(next_tensor, 1)

    one_if_doc_contains_term = tf.SparseTensor(
        indices=term_count_per_doc.indices,
        values=tf.to_double(tf.greater(term_count_per_doc.values, 0)),
        dense_shape=term_count_per_doc.dense_shape)

    return one_if_doc_contains_term

  def _map_to_tfidf(x, reduced_term_freq, corpus_size):
    """Calculates the inverse document frequency of terms in the corpus.

    Args:
      x : a SparseTensor of int64 representing string indices in vocab.
      reduced_term_freq: A dense tensor of shape (vocabSize,) that represents
          the count of the number of documents with each term.
      corpus_size: A scalar count of the number of documents in the corpus

    Returns:
      The tf*idf values
    """
    # Add one to the reduced term freqnencies to avoid dividing by zero.
    idf = tf.log(tf.to_double(corpus_size) / (
        1.0 + tf.to_double(reduced_term_freq)))

    dense_doc_sizes = tf.to_double(tf.sparse_reduce_sum(tf.SparseTensor(
        indices=x.indices,
        values=tf.ones_like(x.values),
        dense_shape=x.dense_shape), 1))

    # For every term in x, divide the idf by the doc size.
    # The two gathers both result in shape <sum_doc_sizes>
    idf_over_doc_size = (tf.gather(idf, x.values) /
                         tf.gather(dense_doc_sizes, x.indices[:, 0]))

    return tf.SparseTensor(
        indices=x.indices,
        values=idf_over_doc_size,
        dense_shape=x.dense_shape)

  cleaned_input = api.map(_map_to_vocab_range, x)

  docs_with_terms = api.map(_map_to_doc_contains_term, cleaned_input)

  def count_docs_with_term(term_frequency):
    # Sum w/in batch.
    count_of_doc_inter = tf.SparseTensor(
        indices=term_frequency.indices,
        values=tf.ones_like(term_frequency.values),
        dense_shape=term_frequency.dense_shape)
    out = tf.sparse_reduce_sum(count_of_doc_inter, axis=0)
    return tf.expand_dims(out, 0)
  count_docs_with_term_column = api.map(count_docs_with_term, docs_with_terms)

  # Expand dims to get around the min_tensor_rank checks
  sizes = api.map(lambda y: tf.expand_dims(tf.shape(y)[0], 0), cleaned_input)

  return api.map(_map_to_tfidf, cleaned_input,
                 analyzers.sum(count_docs_with_term_column,
                               reduce_instance_dims=False),
                 analyzers.sum(sizes))


def string_to_int(x, default_value=-1, top_k=None, frequency_threshold=None,
                  num_oov_buckets=0):
  """Generates a vocabulary for `x` and maps it to an integer with this vocab.

  Args:
    x: A `Column` representing a string value or values.
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      frequency is >= to the supplied threshold. If set to None, the full
      vocabulary is generated.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.

  Returns:
    A `Column` where each string value is mapped to an integer where each unique
    string value is mapped to a different integer and integers are consecutive
    and starting from 0.

  Raises:
    ValueError: If `top_k` or `count_threshold` is negative.
  """
  if top_k is not None:
    top_k = int(top_k)
    if top_k < 0:
      raise ValueError('top_k must be non-negative, but got: %r' % top_k)

  if frequency_threshold is not None:
    frequency_threshold = int(frequency_threshold)
    if frequency_threshold < 0:
      raise ValueError('frequency_threshold must be non-negative, but got: %r' %
                       frequency_threshold)

  def _map_to_int(x, vocab):
    """Maps string tensor into indexes using vocab.

    It uses a dummy vocab when the input vocab is empty.

    Args:
      x : a Tensor/SparseTensor of string.
      vocab : a Tensor/SparseTensor containing unique string values within x.

    Returns:
      a Tensor/SparseTensor of indexes (int) of the same shape as x.
    """

    def _fix_vocab_if_needed(vocab):
      num_to_add = 1 - tf.minimum(tf.size(vocab), 1)
      return tf.concat([
          vocab, tf.fill(
              tf.reshape(num_to_add, (1,)), '__dummy_value__index_zero__')
      ], 0)

    table = lookup.string_to_index_table_from_tensor(
        _fix_vocab_if_needed(vocab), num_oov_buckets=num_oov_buckets,
        default_value=default_value)
    return table.lookup(x)

  return api.map(_map_to_int, x,
                 analyzers.uniques(
                     x, top_k=top_k, frequency_threshold=frequency_threshold))


def segment_indices(segment_ids):
  """Returns a tensor of indices within each segment.

  segment_ids should be a sequence of non-decreasing non-negative integers that
  define a set of segments, e.g. [0, 0, 1, 2, 2, 2] defines 3 segments of length
  2, 1 and 3.  The return value is a tensor containing the indices within each
  segment.

  Example input: [0, 0, 1, 2, 2, 2]
  Example output: [0, 1, 0, 0, 1, 2]

  Args:
    segment_ids: A 1-d tensor containing an non-decreasing sequence of
        non-negative integers with type `tf.int32` or `tf.int64`.

  Returns:
    A tensor containing the indices within each segment.
  """
  segment_lengths = tf.segment_sum(tf.ones_like(segment_ids), segment_ids)
  segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                             segment_ids)
  return (tf.range(tf.size(segment_ids, out_type=segment_ids.dtype)) -
          segment_starts)


def ngrams(strings, ngram_range):
  """Create a tensor of n-grams.

  Given a vector of strings, return a sparse matrix containing the ngrams from
  each string.  Each row in the output sparse tensor contains the set of
  ngrams from the corresponding element in the input tensor.

  The output ngrams including all whitespace and punctuation from the original
  strings.

  Example:

  strings = ['ab: c', 'wxy.']
  ngrams_range = (1,3)

  output is a sparse tensor with

  indices = [[0, 0], [0, 1], ..., [0, 11], [1, 0], [1, 1], ..., [1, 8]]
  values = ['a', 'ab', 'ab:', 'b', 'b:', 'b: ', ':', ': ', ': c', ' ', ' c',
            'c', 'w', 'wx', 'wxy', 'x', 'xy', 'xy.', 'y', 'y.', '.']
  dense_shape = [2, 12]

  Args:
    strings: A tensor of strings with size [batch_size,].
    ngram_range: A pair with the range (inclusive) of ngram sizes to return.

  Returns:
    A SparseTensor containing all ngrams from each element of the input.

  Raises:
    ValueError: if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]
  """
  # This function is implemented as follows.  First we split the input.  If the
  # input is ['abcd', 'q', 'xyz'] then the split opreation returns a
  # SparseTensor with
  #
  # indices=[[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [2, 1], [2, 2]]
  # values=['a', 'b', 'c', 'd', 'q', 'x', 'y', 'z']
  # dense_shape=[3, 4]
  #
  # We then create shifts of the values and first column of indices, buffering
  # to avoid overruning the end of the array, so the shifted values (if we are
  # creating ngrams up to size 3) are
  #
  # shifted_batch_indices[0]=[0, 0, 0, 0, 1, 2, 2, 2]
  # shifted_chars[0]=['a', 'b', 'c', 'd', 'q', 'x', 'y', 'z']
  #
  # shifted_batch_indices[1]=[0, 0, 0, 1, 2, 2, 2, -1]
  # shifted_chars[1]=['b', 'c', 'd', 'q', 'x', 'y', 'z', '']
  #
  # shifted_batch_indices[2]=[0, 0, 1, 2, 2, 2, -1, -1]
  # shifted_chars[2]=['c', 'd', 'q', 'x', 'y', 'z', '', '']
  #
  # These shifted ngrams are used to create the ngrams as follows.  We use
  # tf.string_join to join shifted_chars[:k] to create k-grams.  The batch that
  # the first of these belonged to is given by shifted_batch_indices[0].
  # However some of these will cross the boundaries between 'batches' and so
  # we we create a boolean mask which is True when shifted_indices[:k] are all
  # equal.
  #
  # This results in tensors of ngrams, their batch indices and a boolean mask,
  # which we then use to construct the output SparseTensor.
  chars = tf.string_split(strings, delimiter='')

  if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
    raise ValueError('Invalid ngram_range: %r' % (ngram_range,))

  def _sliding_windows(values, num_shifts, fill_value):
    buffered_values = tf.concat(
        [values, tf.fill([num_shifts - 1], fill_value)], 0)
    return [tf.slice(buffered_values, [i], tf.shape(values))
            for i in range(num_shifts)]

  shifted_batch_indices = _sliding_windows(
      chars.indices[:, 0], ngram_range[1] + 1, tf.constant(-1, dtype=tf.int64))
  shifted_chars = _sliding_windows(chars.values, ngram_range[1] + 1, '')

  # Construct a tensor of the form
  # [['a', 'ab, 'abc'], ['b', 'bcd', cde'], ...]
  def _string_join(tensors):
    if tensors:
      return tf.string_join(tensors)
    else:
      return

  ngrams_array = [_string_join(shifted_chars[:k])
                  for k in range(ngram_range[0], ngram_range[1] + 1)]
  ngrams_tensor = tf.stack(ngrams_array, 1)

  # Construct a boolean mask for whether each ngram in ngram_tensor is valid,
  # in that each character cam from the same batch.
  valid_ngram = tf.equal(tf.cumprod(
      tf.to_int32(tf.equal(tf.stack(shifted_batch_indices, 1),
                           tf.expand_dims(shifted_batch_indices[0], 1))),
      axis=1), 1)
  valid_ngram = valid_ngram[:, (ngram_range[0] - 1):ngram_range[1]]

  # Construct a tensor with the batch that each ngram in ngram_tensor belongs
  # to.
  batch_indices = tf.tile(tf.expand_dims(chars.indices[:, 0], 1),
                          [1, ngram_range[1] + 1 - ngram_range[0]])

  # Apply the boolean mask and construct a SparseTensor with the given indices
  # and values, where another index is added to give the position within a
  # batch.
  batch_indices = tf.boolean_mask(batch_indices, valid_ngram)
  ngrams_tensor = tf.boolean_mask(ngrams_tensor, valid_ngram)
  instance_indices = segment_indices(batch_indices)
  return tf.SparseTensor(
      tf.stack([batch_indices, instance_indices], 1),
      ngrams_tensor,
      tf.stack([tf.size(strings, out_type=tf.int64),
                tf.reduce_max(instance_indices) + 1], 0))
