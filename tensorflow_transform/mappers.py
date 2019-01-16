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


import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import schema_inference

from tensorflow.contrib.boosted_trees.python.ops import quantile_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.util import deprecation


def sparse_tensor_to_dense_with_shape(x, shape):
  """Converts a `SparseTensor` into a dense tensor and sets its shape.

  Args:
    x: A `SparseTensor`.
    shape: The desired shape of the densified `Tensor`.

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
  dense = tf.sparse_to_dense(x.indices, new_dense_shape, x.values)
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
  with tf.name_scope(name, 'scale_by_min_max'):
    if output_min >= output_max:
      raise ValueError('output_min must be less than output_max')

    x = tf.to_float(x)
    min_x_value, max_x_value = analyzers._min_and_max(  # pylint: disable=protected-access
        x, reduce_instance_dims=not elementwise)

    x_shape = tf.shape(x)

    # If min==max, the result will be the mean of the requested range.
    # Note that both the options of tf.where are computed, which means that this
    # will compute unused NaNs.
    if elementwise:
      where_cond = tf.tile(
          tf.expand_dims(min_x_value < max_x_value, 0),
          tf.concat([[x_shape[0]], tf.ones_like(x_shape[1:])], axis=0))
    else:
      where_cond = tf.fill(x_shape, min_x_value < max_x_value)
    scaled_result = tf.where(where_cond,
                             (x - min_x_value) / (max_x_value - min_x_value),
                             tf.fill(x_shape, 0.5))

    return (scaled_result * (output_max - output_min)) + output_min


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
  with tf.name_scope(name, 'scale_to_z_score'):
    # x_mean will be float16, float32, or float64, depending on type of x.
    x_mean, x_var = analyzers._mean_and_var(  # pylint: disable=protected-access
        x, reduce_instance_dims=not elementwise, output_dtype=output_dtype)
    compose_result_fn = lambda values: values
    x_values = x

    if isinstance(x, tf.SparseTensor):

      x_values = x.values
      compose_result_fn = (lambda values: tf.SparseTensor(  # pylint: disable=g-long-lambda
          indices=x.indices, values=values, dense_shape=x.dense_shape))
      if elementwise:
        # Only supports SparseTensors with rank 2.
        x.get_shape().assert_has_rank(2)

        x_mean = tf.gather(x_mean, x.indices[:, 1])
        x_var = tf.gather(x_var, x.indices[:, 1])

    numerator = tf.cast(x_values, x_mean.dtype) - x_mean
    denominator = tf.sqrt(x_var)
    cond = tf.not_equal(denominator, 0)

    if elementwise and isinstance(x, tf.Tensor):
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

  with tf.name_scope(name, 'tfidf'):
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
  # token receives is not important, only that the tfidf value and vocab index
  # have the *same* dummy index, so that feature_column can apply the weight to
  # the correct vocab item.
  dummy_index = segment_indices(tfidfs.indices[:, 0])
  out_index = tf.concat(
      [tf.expand_dims(tfidfs.indices[:, 0], 1),
       tf.expand_dims(dummy_index, 1)], 1)

  out_shape_second_dim = tf.maximum(tf.reduce_max(dummy_index), -1) + 1
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
  vocab_size = tf.convert_to_tensor(vocab_size, dtype=tf.int64)
  split_indices = tf.to_int64(
      tf.split(x.indices, axis=1, num_or_size_splits=2))
  expanded_values = tf.to_int64(tf.expand_dims(x.values, 1))
  next_index = tf.concat(
      [split_indices[0], split_indices[1], expanded_values], axis=1)

  next_values = tf.ones_like(x.values)
  expanded_vocab_size = tf.expand_dims(vocab_size, 0)
  next_shape = tf.concat(
      [x.dense_shape, expanded_vocab_size], 0)

  next_tensor = tf.SparseTensor(
      indices=tf.to_int64(next_index),
      values=next_values,
      dense_shape=next_shape)

  # Take the intermediary tensor and reduce over the term_index_in_doc
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
        1.0 + tf.to_double(reduced_term_freq))) + 1
  else:
    idf = tf.log(tf.to_double(corpus_size) / (
        tf.to_double(reduced_term_freq))) + 1

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
    x: A `Tensor` or `SparseTensor` of type tf.string.
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
  with tf.name_scope(name, 'compute_and_apply_vocabulary'):
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
    x: A `Tensor` or `SparseTensor` of type tf.string to which the vocabulary
      transformation should be applied.
      The column names are those intended for the transformed tensors.
    deferred_vocab_filename_tensor: The deferred vocab filename tensor as
      returned by `tft.vocabulary`.
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.
    lookup_fn: Optional lookup function, if specified it should take a tensor
      and a deferred vocab filename as an input and return a lookup `op` along
      with the table size, by default `apply_vocab` performs a
      lookup_ops.index_table_from_file for the table lookup.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `SparseTensor` where each string value is mapped to an
    integer. Each unique string value that appears in the vocabulary
    is mapped to a different integer and integers are consecutive
    starting from zero, and string value not in the vocabulary is
    assigned default_value.
  """
  with tf.name_scope(name, 'apply_vocab'):
    if lookup_fn:
      result, table_size = lookup_fn(x, deferred_vocab_filename_tensor)
    else:
      table = lookup_ops.index_table_from_file(
          deferred_vocab_filename_tensor,
          num_oov_buckets=num_oov_buckets,
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
  with tf.name_scope(name, 'segment_indices'):
    segment_lengths = tf.segment_sum(tf.ones_like(segment_ids), segment_ids)
    segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                               segment_ids)
    return (tf.range(tf.size(segment_ids, out_type=segment_ids.dtype)) -
            segment_starts)


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
  with tf.name_scope(name, 'ngrams'):
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
      raise ValueError('Invalid ngram_range: %r' % (ngram_range,))

    def _sliding_windows(values, num_shifts, fill_value):
      buffered_values = tf.concat(
          [values, tf.fill([num_shifts - 1], fill_value)], 0)
      return [tf.slice(buffered_values, [i], tf.shape(values))
              for i in range(num_shifts)]

    shifted_batch_indices = _sliding_windows(
        tokens.indices[:, 0], ngram_range[1] + 1,
        tf.constant(-1, dtype=tf.int64))
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
    return tf.string_to_hash_bucket_fast(strings, hash_buckets, name=name)
  return tf.string_to_hash_bucket_strong(strings, hash_buckets, key, name=name)


def bucketize(x, num_buckets, epsilon=None, weights=None, name=None):
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
  with tf.name_scope(name, 'bucketize'):
    if not isinstance(num_buckets, int):
      raise TypeError('num_buckets must be an int, got %s' % type(num_buckets))

    if num_buckets < 1:
      raise ValueError('Invalid num_buckets %d' % num_buckets)

    if epsilon is None:
      # See explanation in args documentation for epsilon.
      epsilon = min(1.0 / num_buckets, 0.01)

    x_values = x.values if isinstance(x, tf.SparseTensor) else x
    bucket_boundaries = analyzers.quantiles(x_values, num_buckets, epsilon,
                                            weights)
    return apply_buckets(x, bucket_boundaries)


def bucketize_per_key(x, key, num_buckets, epsilon=None, name=None):
  """Returns a bucketized column, with a bucket index assigned to each input.

  Args:
    x: A numeric input `Tensor` or `SparseTensor` with rank 1, whose values
      should be mapped to buckets.  `SparseTensor`s will have their non-missing
      values mapped and missing values left as missing.
    key: A Tensor with the same shape as `x` and dtype tf.string.  If `x` is
      a `SparseTensor`, `key` must exactly match `x` in everything except
      values, i.e. indices and dense_shape must be identical.
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
  with tf.name_scope(name, 'bucketize_per_key'):
    if not isinstance(num_buckets, int):
      raise TypeError(
          'num_buckets must be an int, got {}'.format(type(num_buckets)))

    if num_buckets < 1:
      raise ValueError('Invalid num_buckets {}'.format(num_buckets))

    if epsilon is None:
      # See explanation in args documentation for epsilon.
      epsilon = min(1.0 / num_buckets, 0.01)

    key_vocab, bucket_boundaries = analyzers._quantiles_per_key(  # pylint: disable=protected-access
        x.values if isinstance(x, tf.SparseTensor) else x,
        key.values if isinstance(key, tf.SparseTensor) else key,
        num_buckets, epsilon)
    return _apply_buckets_with_keys(x, key, key_vocab, bucket_boundaries)


def _lookup_key(key, key_vocab):
  table = lookup_ops.index_table_from_tensor(key_vocab, default_value=-1)
  key_indices = table.lookup(key)
  with tf.control_dependencies([tf.assert_non_negative(key_indices)]):
    return tf.identity(key_indices)


def _combine_bucket_boundaries(bucket_boundaries, epsilon=0.1):
  """Combine all boundaries into a single vector with offsets.

  We offset boundaries so that this vector is increasing, and store the offsets.
  In order to make the vector strictly increasing, we use an arbitrary epsilon
  value.  E.g. if

  bucket_boundaries = [[0, 5, 7], [2, 3, 4]]

  then the second row will be offset to move the first bucket boundary from
  2 to 7 + epsilon = 7.1.  Thus we will have:

  combined_boundaries = [0, 5, 7, 7.1, 8.1, 9.1]
  offsets = [0, 5.1]

  Args:
    bucket_boundaries: A Tensor with shape (num_keys, num_buckets) where each
        row is increasing.
    epsilon: The distance between values to use when stacking rows of
        `bucket_boundaries` into a single vector.

  Returns:
    A pair (combined_boundaries, offsets) where combined_boundaries has shape
        (num_keys * num_buckets,) and offsets has shape (num_keys,)
  """
  # For each row of bucket_boundaries, compute where that row should start in
  # combined_boundaries.  This is given by taking the cumulative sum of the
  # size of the segment in the number-line taken up by each row (including the
  # extra padding of epsilon).
  row_starts = tf.cumsum(
      epsilon + bucket_boundaries[:, -1] - bucket_boundaries[:, 0],
      exclusive=True)
  offsets = row_starts - bucket_boundaries[:, 0]
  combined_boundaries = tf.reshape(
      bucket_boundaries + tf.expand_dims(offsets, axis=1), [-1])
  return combined_boundaries, offsets


def _apply_buckets_with_keys(x, key, key_vocab, bucket_boundaries, name=None):
  """Bucketize a Tensor or SparseTensor where boundaries depend on the index.

  Args:
    x: A 1-d Tensor or SparseTensor.
    key: A 1-d Tensor or SparseTensor with the same size as x.
    key_vocab: A vocab containing all keys.  Must be exhaustive, an
        out-of-vocab entry in `key` will cause a crash.
    bucket_boundaries: A rank-2 Tensor of shape (key_size, num_buckets)
    name: (Optional) A name for this operation.

  Returns:
    A tensor with the same shape as `x` and dtype tf.int64
  """
  with tf.name_scope(name, 'apply_buckets_with_keys'):
    x_values = x.values if isinstance(x, tf.SparseTensor) else x
    key_values = key.values if isinstance(key, tf.SparseTensor) else key

    x_values = tf.to_float(x_values)
    # Convert `key_values` to indices in key_vocab.  We must use apply_function
    # since this uses a Table.
    key_indices = _lookup_key(key_values, key_vocab)

    combined_boundaries, offsets = _combine_bucket_boundaries(bucket_boundaries)

    # Apply the per-key offsets to x, which produces offset buckets (where the
    # bucket offset is an integer offset).  Then remove this offset to get the
    # actual per-key buckets for x.
    offset_x = x_values + tf.gather(offsets, key_indices)
    offset_buckets = tf.to_int64(quantile_ops.bucketize_with_input_boundaries(
        offset_x, combined_boundaries))
    num_buckets = tf.to_int64(tf.shape(bucket_boundaries)[1])
    bucketized_values = tf.clip_by_value(
        offset_buckets - key_indices * num_buckets, 0, num_buckets)

    # Attach the relevant metadata to result, so that the corresponding
    # output feature will have this metadata set.
    min_value = tf.constant(0, tf.int64)
    max_value = num_buckets
    schema_inference.set_tensor_schema_override(
        bucketized_values, min_value, max_value)

    if isinstance(x, tf.SparseTensor):
      result = tf.SparseTensor(x.indices, bucketized_values, x.dense_shape)
    else:
      result = bucketized_values

    return result


def apply_buckets(x, bucket_boundaries, name=None):
  """Returns a bucketized column, with a bucket index assigned to each input.

  Args:
    x: A numeric input `Tensor` or `SparseTensor` whose values should be mapped
        to buckets.  For `SparseTensor`s, the non-missing values will be mapped
        to buckets and missing value left missing.
    bucket_boundaries: The bucket boundaries represented as a rank 1 `Tensor`.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` of the same shape as `x`, with each element in the
    returned tensor representing the bucketized value. Bucketized value is
    in the range [0, len(bucket_boundaries)].
  """
  with tf.name_scope(name, 'apply_buckets'):
    x_values = x.values if isinstance(x, tf.SparseTensor) else x
    buckets = quantile_ops.bucketize_with_input_boundaries(
        x_values, boundaries=bucket_boundaries, name='assign_buckets')
    # Convert to int64 because int32 is not compatible with tf.Example parser.
    # See _TF_EXAMPLE_ALLOWED_TYPES in FixedColumnRepresentation()
    # in tf_metadata/dataset_schema.py
    bucketized_values = tf.to_int64(buckets)

    # Attach the relevant metadata to result, so that the corresponding
    # output feature will have this metadata set.
    min_value = tf.constant(0, tf.int64)
    max_value = tf.shape(bucket_boundaries)[1]
    schema_inference.set_tensor_schema_override(
        bucketized_values, min_value, max_value)

    if isinstance(x, tf.SparseTensor):
      result = tf.SparseTensor(x.indices, bucketized_values, x.dense_shape)
    else:
      result = bucketized_values

    return result
