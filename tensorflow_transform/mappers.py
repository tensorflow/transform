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
from tensorflow.python.util.deprecation import deprecated


def scale_by_min_max(x, output_min=0.0, output_max=1.0):
  """Scale a numerical column into the range [output_min, output_max].

  Args:
    x: A numeric `Tensor`.
    output_min: The minimum of the range of output values.
    output_max: The maximum of the range of output values.

  Returns:
    A `Tensor` containing the input column scaled to [output_min, output_max].

  Raises:
    ValueError: If output_min, output_max have the wrong order.
  """
  if output_min >= output_max:
    raise ValueError('output_min must be less than output_max')

  min_x_value = analyzers.min(x)
  max_x_value = analyzers.max(x)
  return ((((x - min_x_value) * (output_max - output_min)) /
           (max_x_value - min_x_value)) + output_min)


def scale_to_0_1(x):
  """Returns a column which is the input column scaled to have range [0,1].

  Args:
    x: A numeric te`Tensor`sor.

  Returns:
    A `Tensor` containing the input column scaled to [0, 1].
  """
  return scale_by_min_max(x, 0, 1)


def tfidf(x, vocab_size, smooth=True):
  """Maps the terms in x to their term frequency * inverse document frequency.

  The inverse document frequency of a term is calculated as
  log((corpus size + 1) / (document frequency of term + 1)) by default.

  Example usage:
    example strings [["I", "like", "pie", "pie", "pie"], ["yum", "yum", "pie]]
    in: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                              [1, 0], [1, 1], [1, 2]],
                     values=[1, 2, 0, 0, 0, 3, 3, 0])
    out: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                      values=[1, 2, 0, 3, 0])
         SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                      values=[(1/5)*log(3/2), (1/5)*log(3/2), 0,
                              0, (2/3)*log(3/2)])
    NOTE that the first doc's duplicate "pie" strings have been combined to
    one output, as have the second doc's duplicate "yum" strings.

  Args:
    x: A `SparseTensor` representing int64 values (most likely that are the
        result of calling string_to_int on a tokenized string).
    vocab_size: An int - the count of vocab used to turn the string into int64s
        including any OOV buckets.
    smooth: A bool indicating if the inverse document frequency should be
        smoothed. If True, which is the default, then the idf is calculated as
        log((corpus size + 1) / (document frequency of term + 1)).
        Otherwise, the idf is
        log((corpus size) / (document frequency of term)), which could
        result in a divizion by zero error.

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

  cleaned_input = _to_vocab_range(x)

  term_frequencies = _to_term_frequency(cleaned_input, vocab_size)

  count_docs_with_term_column = _count_docs_with_term(term_frequencies)
  # Expand dims to get around the min_tensor_rank checks
  sizes = tf.expand_dims(tf.shape(cleaned_input)[0], 0)
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
  # token recieves is not important, only that the tfidf value and vocab index
  # have the *same* dummy index, so that feature_column can apply the weight to
  # the correct vocab item.
  dummy_index = segment_indices(tfidfs.indices[:, 0])
  out_index = tf.concat(
      [tf.expand_dims(tfidfs.indices[:, 0], 1),
       tf.expand_dims(dummy_index, 1)], 1)
  out_shape = [tfidfs.dense_shape[0], tf.reduce_max(dummy_index)+1]

  de_duped_indicies_out = tf.SparseTensor(
      indices=out_index,
      values=tfidfs.indices[:, 1],
      dense_shape=out_shape)
  de_duped_tfidf_out = tf.SparseTensor(
      indices=out_index,
      values=tfidfs.values,
      dense_shape=out_shape)
  return de_duped_indicies_out, de_duped_tfidf_out


def _to_term_frequency(x, vocab_size):
  """Creates a SparseTensor of term frequency for every doc/term pair.

  Args:
    x : a SparseTensor of int64 representing string indices in vocab.
    vocab_size: An int - the count of vocab used to turn the string into int64s
        including any OOV buckets.

  Returns:
    a SparseTensor with the count of times a term appears in a document at
        indices <doc_index_in_batch>, <term_index_in_vocab>,
        with size (num_docs_in_batch, vocab_size).
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

  dense_doc_sizes = tf.to_double(tf.sparse_reduce_sum(tf.SparseTensor(
      indices=x.indices,
      values=tf.ones_like(x.values),
      dense_shape=x.dense_shape), 1))

  gather_indices = term_count_per_doc.indices[:, 0]
  gathered_doc_sizes = tf.gather(dense_doc_sizes, gather_indices)

  term_frequency = (tf.to_double(term_count_per_doc.values) /
                    tf.to_double(gathered_doc_sizes))
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
    idf = tf.log((tf.to_double(corpus_size) + 1.0) / (
        1.0 + tf.to_double(reduced_term_freq)))
  else:
    idf = tf.log(tf.to_double(corpus_size) / (
        tf.to_double(reduced_term_freq)))

  gathered_idfs = tf.gather(tf.squeeze(idf), term_frequency.indices[:, 1])
  tfidf_values = tf.to_float(term_frequency.values) * tf.to_float(gathered_idfs)

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
  out = tf.sparse_reduce_sum(count_of_doc_inter, axis=0)
  return tf.expand_dims(out, 0)


def string_to_int(x, default_value=-1, top_k=None, frequency_threshold=None,
                  num_oov_buckets=0):
  """Generates a vocabulary for `x` and maps it to an integer with this vocab.

  Args:
    x: A `Tensor` or `SparseTensor` of type tf.string.
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
    A `Tensor` or `SparseTensor` where each string value is mapped to an integer
    where each unique string value is mapped to a different integer and integers
    are consecutive and starting from 0.

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

  def _fix_vocab_if_needed(vocab):
    num_to_add = 1 - tf.minimum(tf.size(vocab), 1)
    return tf.concat([
        vocab, tf.fill(
            tf.reshape(num_to_add, (1,)), '__dummy_value__index_zero__')
    ], 0)

  def _apply_vocab(x, vocab):
    table = lookup.string_to_index_table_from_tensor(
        vocab, num_oov_buckets=num_oov_buckets, default_value=default_value)
    return table.lookup(x)

  vocab = analyzers.uniques(
      x, top_k=top_k, frequency_threshold=frequency_threshold)
  vocab = _fix_vocab_if_needed(vocab)
  return api.apply_function(_apply_vocab, x, vocab)


def segment_indices(segment_ids):
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

  Returns:
    A `Tensor` containing the indices within each segment.
  """
  segment_lengths = tf.segment_sum(tf.ones_like(segment_ids), segment_ids)
  segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                             segment_ids)
  return (tf.range(tf.size(segment_ids, out_type=segment_ids.dtype)) -
          segment_starts)


def ngrams(tokens, ngram_range, separator):
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

  Returns:
    A `SparseTensor` containing all ngrams from each row of the input.

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
  # buffering to avoid overruning the end of the array, so the shifted values
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

  if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
    raise ValueError('Invalid ngram_range: %r' % (ngram_range,))

  def _sliding_windows(values, num_shifts, fill_value):
    buffered_values = tf.concat(
        [values, tf.fill([num_shifts - 1], fill_value)], 0)
    return [tf.slice(buffered_values, [i], tf.shape(values))
            for i in range(num_shifts)]

  shifted_batch_indices = _sliding_windows(
      tokens.indices[:, 0], ngram_range[1] + 1, tf.constant(-1, dtype=tf.int64))
  shifted_tokens = _sliding_windows(tokens.values, ngram_range[1] + 1, '')

  # Construct a tensor of the form
  # [['a', 'ab, 'abc'], ['b', 'bcd', cde'], ...]
  def _string_join(tensors):
    if tensors:
      return tf.string_join(tensors, separator=separator)
    else:
      return

  ngrams_array = [_string_join(shifted_tokens[:k])
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
  batch_indices = tf.tile(tf.expand_dims(tokens.indices[:, 0], 1),
                          [1, ngram_range[1] + 1 - ngram_range[0]])

  # Apply the boolean mask and construct a SparseTensor with the given indices
  # and values, where another index is added to give the position within a
  # batch.
  batch_indices = tf.boolean_mask(batch_indices, valid_ngram)
  ngrams_tensor = tf.boolean_mask(ngrams_tensor, valid_ngram)
  instance_indices = segment_indices(batch_indices)
  dense_shape_second_dim = tf.maximum(tf.reduce_max(instance_indices), -1) + 1
  return tf.SparseTensor(
      indices=tf.stack([batch_indices, instance_indices], 1),
      values=ngrams_tensor,
      dense_shape=tf.stack(
          [tokens.dense_shape[0], dense_shape_second_dim]))


def hash_strings(strings, hash_buckets, key=None):
  """Hash strings into buckets.

  Args:
    strings: a `Tensor` or `SparseTensor` of dtype `tf.string`.
    hash_buckets: the number of hash buckets.
    key: optional. An array of two Python `uint64`. If passed, output will be
      a deterministic function of `strings` and `key`. Note that hashing will be
      slower if this value is specified.

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
  if key is None:
    return tf.string_to_hash_bucket_fast(
        strings, hash_buckets, name='hash_strings')
  return tf.string_to_hash_bucket_strong(
      strings, hash_buckets, key, name='hash_strings')


##############################################################################
###                                                                        ###
###                               DEPRECATED                               ###
###                                                                        ###
##############################################################################


@deprecated('2017-08-25',
            'Use tfidf() instead.')
def tfidf_weights(x, vocab_size):
  """Maps the terms in x to their (1/doc_length) * inverse document frequency.

  Args:
    x: A `SparseTensor` representing int64 values (most likely that are the
        result of calling string_to_int on a tokenized string).
    vocab_size: An int - the count of vocab used to turn the string into int64s
        including any OOV buckets.

  Returns:
    A `SparseTensor` where each int value is mapped to a double equal to
    (1 if that term appears in that row, 0 otherwise / the number of terms in
    that row) * the log of (the number of rows in `x` / (1 + the number of
    rows in `x` where the term appears at least once))

  NOTE:
    This is intented to be used with the feature_column 'sum' combiner to arrive
    at the true term frequncies.
  """

  def _to_vocab_range(x):
    """Enforces that the vocab_ids in x are positive."""
    return tf.SparseTensor(
        indices=x.indices,
        values=tf.mod(x.values, vocab_size),
        dense_shape=x.dense_shape)

  def _to_doc_contains_term(x):
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

  def _to_idf_over_doc_size(x, reduced_term_freq, corpus_size):
    """Calculates the inverse document frequency of terms in the corpus.

    Args:
      x : a `SparseTensor` of int64 representing string indices in vocab.
      reduced_term_freq: A `Tensor` of shape (vocabSize,) that represents the
          count of the number of documents with each term.
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
        values=tf.to_float(idf_over_doc_size),
        dense_shape=x.dense_shape)

  cleaned_input = _to_vocab_range(x)

  docs_with_terms = _to_doc_contains_term(cleaned_input)

  def count_docs_with_term(term_frequency):
    # Sum w/in batch.
    count_of_doc_inter = tf.SparseTensor(
        indices=term_frequency.indices,
        values=tf.ones_like(term_frequency.values),
        dense_shape=term_frequency.dense_shape)
    out = tf.sparse_reduce_sum(count_of_doc_inter, axis=0)
    return tf.expand_dims(out, 0)

  count_docs_with_term_column = count_docs_with_term(docs_with_terms)
  # Expand dims to get around the min_tensor_rank checks
  sizes = tf.expand_dims(tf.shape(cleaned_input)[0], 0)
  return _to_idf_over_doc_size(cleaned_input,
                               analyzers.sum(count_docs_with_term_column,
                                             reduce_instance_dims=False),
                               analyzers.sum(sizes))
