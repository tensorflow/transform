# Copyright 2022 Google Inc. All Rights Reserved.
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
"""Experimental functions that transform features based on full-pass analysis.

The core tf.Transform API requires a user to construct a
"preprocessing function" that accepts and returns `Tensor`s.  This function is
built by composing regular functions built from TensorFlow ops, as well as
special functions we refer to as `Analyzer`s.  `Analyzer`s behave similarly to
TensorFlow ops but require a full pass over the whole dataset to compute their
output value.  The analyzers are defined in analyzers.py, while this module
provides helper functions that call analyzers and then use the results of the
anaylzers to transform the original data.

The user-defined preprocessing function should accept and return `Tensor`s that
are batches from the dataset, whose batch size may vary.
"""

from typing import Any, Optional, Union, Sequence

import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import mappers
from tensorflow_transform import tf_utils
from tensorflow_transform.experimental import analyzers as experimental_analyzers


@common.log_api_use(common.MAPPER_COLLECTION)
def compute_and_apply_approximate_vocabulary(
    x: common_types.ConsistentTensorType,
    *,  # Force passing optional parameters by keys.
    default_value: Any = -1,
    top_k: Optional[int] = None,
    num_oov_buckets: int = 0,
    vocab_filename: Optional[str] = None,
    weights: Optional[tf.Tensor] = None,
    file_format: common_types.VocabularyFileFormatType = analyzers.DEFAULT_VOCABULARY_FILE_FORMAT,
    store_frequency: Optional[bool] = False,
    reserved_tokens: Optional[Union[Sequence[str], tf.Tensor]] = None,
    name: Optional[str] = None,
) -> common_types.ConsistentTensorType:
  """Generates an approximate vocabulary for `x` and maps it to an integer.

  Args:
    x: A `Tensor`, `SparseTensor`, or `RaggedTensor` of type tf.string or
      tf.int[8|16|32|64].
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.
    vocab_filename: The file name for the vocabulary file. If None, a name based
      on the scope name in the context of this graph will be used as the file
      name. If not None, should be unique within a given preprocessing function.
      NOTE in order to make your pipelines resilient to implementation details
      please set `vocab_filename` when you are using the vocab_filename on a
      downstream component.
    weights: (Optional) Weights `Tensor` for the vocabulary. It must have the
      same shape as x.
    file_format: (Optional) A str. The format of the resulting vocabulary file.
      Accepted formats are: 'tfrecord_gzip', 'text'. 'tfrecord_gzip' requires
      tensorflow>=2.4. The default value is 'text'.
    store_frequency: If True, frequency of the words is stored in the vocabulary
      file. In the case labels are provided, the mutual information is stored in
      the file instead. Each line in the file will be of the form 'frequency
      word'. NOTE: if True and text_format is 'text' then spaces will be
      replaced to avoid information loss.
    reserved_tokens: (Optional) A list of tokens that should appear in the
      vocabulary regardless of their appearance in the input. These tokens would
      maintain their order, and have a reserved spot at the beginning of the
      vocabulary. Note: this field has no affect on cache.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor`, `SparseTensor`, or `RaggedTensor` where each string value is
    mapped to an integer. Each unique string value that appears in the
    vocabulary is mapped to a different integer and integers are consecutive
    starting from zero. String value not in the vocabulary is assigned
    `default_value`. Alternatively, if `num_oov_buckets` is specified, out of
    vocabulary strings are hashed to values in
    [vocab_size, vocab_size + num_oov_buckets) for an overall range of
    [0, vocab_size + num_oov_buckets).

  Raises:
    ValueError: If `top_k` is negative.
      If `file_format` is not in the list of allowed formats.
      If x.dtype is not string or integral.
  """
  with tf.compat.v1.name_scope(name,
                               'compute_and_apply_approximate_vocabulary'):
    if store_frequency and file_format == 'text':
      x = tf_utils.maybe_format_vocabulary_input(x)
    deferred_vocab_and_filename = experimental_analyzers.approximate_vocabulary(
        x=x,
        top_k=top_k,
        vocab_filename=vocab_filename,
        weights=weights,
        file_format=file_format,
        store_frequency=store_frequency,
        reserved_tokens=reserved_tokens,
        name=name,
    )
    return mappers._apply_vocabulary_internal(  # pylint: disable=protected-access
        x,
        deferred_vocab_and_filename,
        default_value,
        num_oov_buckets,
        lookup_fn=None,
        file_format=file_format,
        store_frequency=store_frequency,
        name=None,
    )


@common.log_api_use(common.MAPPER_COLLECTION)
def document_frequency(x: tf.SparseTensor,
                       vocab_size: int,
                       name: Optional[str] = None) -> tf.SparseTensor:
  """Maps the terms in x to their document frequency in the same order.

  The document frequency of a term is the number of documents that contain the
  term in the entire dataset. Each unique vocab term has a unique document
  frequency.

  Example usage:

  >>> def preprocessing_fn(inputs):
  ...   integerized = tft.compute_and_apply_vocabulary(inputs['x'])
  ...   vocab_size = tft.get_num_buckets_for_transformed_feature(integerized)
  ...   return {
  ...      'df': tft.experimental.document_frequency(integerized, vocab_size),
  ...      'integerized': integerized,
  ...   }
  >>> raw_data = [dict(x=["I", "like", "pie", "pie", "pie"]),
  ...             dict(x=["yum", "yum", "pie"])]
  >>> feature_spec = dict(x=tf.io.VarLenFeature(tf.string))
  >>> raw_data_metadata = tft.DatasetMetadata.from_feature_spec(feature_spec)
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'df': array([1, 1, 2, 2, 2]), 'integerized': array([3, 2, 0, 0, 0])},
   {'df': array([1, 1, 2]), 'integerized': array([1, 1, 0])}]

    ```
    example strings: [["I", "like", "pie", "pie", "pie"], ["yum", "yum", "pie]]
    in: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                              [1, 0], [1, 1], [1, 2]],
                     values=[1, 2, 0, 0, 0, 3, 3, 0])
    out: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                              [1, 0], [1, 1], [1, 2]],
                     values=[1, 1, 2, 2, 2, 1, 1, 2])
    ```

  Args:
    x: A 2D `SparseTensor` representing int64 values (most likely that are the
      result of calling `compute_and_apply_vocabulary` on a tokenized string).
    vocab_size: An int - the count of vocab used to turn the string into int64s
      including any OOV buckets.
    name: (Optional) A name for this operation.

  Returns:
    `SparseTensor`s with indices [index_in_batch, index_in_local_sequence] and
    values document_frequency. Same shape as the input `x`.

  Raises:
    ValueError if `x` does not have 2 dimensions.
  """
  if x.get_shape().ndims != 2:
    raise ValueError('tft.tfidf requires a 2D SparseTensor input. '
                     'Input had {} dimensions.'.format(x.get_shape().ndims))

  with tf.compat.v1.name_scope(name, 'df'):
    cleaned_input = tf_utils.to_vocab_range(x, vocab_size)

    # all_df is a (1, vocab_size)-shaped sparse tensor storing number of docs
    # containing each term in the entire dataset.
    all_df = _to_global_document_frequency(cleaned_input, vocab_size)

    # df_values is a batch_size * sequence_size sparse tensor storing the
    # document frequency of each term, following the same order as the terms
    # within each document.
    df_values = tf.gather(tf.squeeze(all_df), cleaned_input.values)

    return tf.SparseTensor(
        indices=cleaned_input.indices,
        values=df_values,
        dense_shape=cleaned_input.dense_shape)


@common.log_api_use(common.MAPPER_COLLECTION)
def idf(x: tf.SparseTensor,
        vocab_size: int,
        smooth: bool = True,
        add_baseline: bool = True,
        name: Optional[str] = None) -> tf.SparseTensor:
  """Maps the terms in x to their inverse document frequency in the same order.

  The inverse document frequency of a term, by default, is calculated as
  1 + log ((corpus size + 1) / (count of documents containing term + 1)).

  Example usage:

  >>> def preprocessing_fn(inputs):
  ...   integerized = tft.compute_and_apply_vocabulary(inputs['x'])
  ...   vocab_size = tft.get_num_buckets_for_transformed_feature(integerized)
  ...   idf_weights = tft.experimental.idf(integerized, vocab_size)
  ...   return {
  ...      'idf': idf_weights,
  ...      'integerized': integerized,
  ...   }
  >>> raw_data = [dict(x=["I", "like", "pie", "pie", "pie"]),
  ...             dict(x=["yum", "yum", "pie"])]
  >>> feature_spec = dict(x=tf.io.VarLenFeature(tf.string))
  >>> raw_data_metadata = tft.DatasetMetadata.from_feature_spec(feature_spec)
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> # 1 + log(3/2) = 1.4054651
  >>> transformed_data
  [{'idf': array([1.4054651, 1.4054651, 1., 1., 1.], dtype=float32),
    'integerized': array([3, 2, 0, 0, 0])},
   {'idf': array([1.4054651, 1.4054651, 1.], dtype=float32),
    'integerized': array([1, 1, 0])}]

    ```
    example strings: [["I", "like", "pie", "pie", "pie"], ["yum", "yum", "pie]]
    in: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                              [1, 0], [1, 1], [1, 2]],
                     values=[1, 2, 0, 0, 0, 3, 3, 0])
    out: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                              [1, 0], [1, 1], [1, 2]],
                     values=[1 + log(3/2), 1 + log(3/2), 1, 1, 1,
                             1 + log(3/2), 1 + log(3/2), 1])
    ```

  Args:
    x: A 2D `SparseTensor` representing int64 values (most likely that are the
      result of calling `compute_and_apply_vocabulary` on a tokenized string).
    vocab_size: An int - the count of vocab used to turn the string into int64s
      including any OOV buckets.
    smooth: A bool indicating if the inverse document frequency should be
      smoothed. If True, which is the default, then the idf is calculated as 1 +
      log((corpus size + 1) / (document frequency of term + 1)). Otherwise, the
      idf is 1 + log((corpus size) / (document frequency of term)), which could
      result in a division by zero error.
    add_baseline: A bool indicating if the inverse document frequency should be
      added with a constant baseline 1.0. If True, which is the default, then
      the idf is calculated as 1 + log(*). Otherwise, the idf is log(*) without
      the constant 1 baseline. Keeping the baseline reduces the discrepancy in
      idf between commonly seen terms and rare terms.
    name: (Optional) A name for this operation.

  Returns:
    `SparseTensor`s with indices [index_in_batch, index_in_local_sequence] and
    values inverse document frequency. Same shape as the input `x`.

  Raises:
    ValueError if `x` does not have 2 dimensions.
  """
  if x.get_shape().ndims != 2:
    raise ValueError('tft.tfidf requires a 2D SparseTensor input. '
                     'Input had {} dimensions.'.format(x.get_shape().ndims))

  with tf.compat.v1.name_scope(name, 'idf'):
    cleaned_input = tf_utils.to_vocab_range(x, vocab_size)

    batch_sizes = tf.expand_dims(tf.shape(input=cleaned_input)[0], 0)

    # all_df is a (1, vocab_size)-shaped tensor storing number of documents
    # containing each term in the entire dataset.
    all_df = _to_global_document_frequency(cleaned_input, vocab_size)

    # all_idf is a (1, vocab_size)-shaped tensor storing the inverse document
    # frequency of each term in the entire dataset.
    all_idf = tf_utils.document_frequency_to_idf(
        all_df,
        analyzers.sum(batch_sizes),
        smooth=smooth,
        add_baseline=add_baseline)

    # idf_values is a batch_size * sequence_size sparse tensor storing the
    # inverse document frequency of each term, following the same order as the
    # terms within each document.
    idf_values = tf.gather(
        tf.reshape(all_idf, [-1]), tf.cast(cleaned_input.values, dtype=tf.int64)
    )

    return tf.SparseTensor(
        indices=cleaned_input.indices,
        values=idf_values,
        dense_shape=cleaned_input.dense_shape)


def _to_term_document_one_hot(
    x: tf.SparseTensor, vocab_size: Union[int, tf.Tensor]) -> tf.SparseTensor:
  """Creates a one-hot SparseTensor of term existence for every doc/term pair.

  Converts a <batch_indices, local_sequence_index>-indexed, <vocab_index>-valued
  sparse tensor to <batch_indices, vocab_index> one-hot tensor to represent the
  existence of each vocab term in each document of a batch. For example, when x
  has the dense form:
    [[3, 2, 3],  # first example of the batch has vocab term 2 and 3
     [1, 1]],    # second example of the batch has vocab term 1
  with vocab_size=4, the dense form of the out one-hot tensor is
    [[0, 0, 1, 1],
     [0, 1, 0, 0]]

  Args:
    x: a SparseTensor of int64 representing string indices in vocab. The indices
      are <batch_index, local_index> and the values are <vocab_index>.
      Typically, x is the output of tft.compute_and_apply_vocabulary.
    vocab_size: A scalar int64 Tensor - the count of vocab used to turn the
      string into int64s including any OOV buckets.

  Returns:
    a SparseTensor with size (batch_size, vocab_size), indices being
      <doc_index_in_batch>, <term_index_in_vocab> and int32 values being 1 for
      all mentioned terms or 0 if not shown in each document.
  """
  vocab_size = tf.convert_to_tensor(value=vocab_size, dtype=tf.int64)

  # Combine batch indices (first column of x's indices) and vocab indices (
  # x's values) as new indices (<batch_index>, <vocab_index>).
  batch_indices = x.indices[:, 0]  # sparse tensor indices are int64
  vocab_indices = tf.cast(x.values, dtype=tf.int64)

  # Dedup (<batch_index>, <vocab_index>) pairs. This is because document
  # frequency only cares the existence of a term in a document, not the
  # occurrence frequency within that document.
  # Hashing (<batch_index>, <vocab_index>) pairs for dedup.
  multiplier = vocab_size + 1
  unique_flatten_indices, _ = tf.raw_ops.UniqueV2(
      x=batch_indices * multiplier + vocab_indices, axis=[0])
  unique_batch_indices = tf.cast(
      tf.math.divide(unique_flatten_indices, multiplier), dtype=tf.int64)
  unique_vocab_indices = tf.math.mod(unique_flatten_indices, multiplier)
  unique_batch_vocab_indices = tf.transpose(
      tf.stack([unique_batch_indices, unique_vocab_indices]))

  # If term i shows at least once in document j, then doc_freq<i, j> = 1
  one_hot_values = tf.ones_like(unique_flatten_indices, dtype=tf.int32)

  # New shape of the one hot tensor is batch_size * vocab_size
  new_shape = tf.stack([x.dense_shape[0], vocab_size])
  new_shape.set_shape([2])

  return tf.SparseTensor(
      indices=unique_batch_vocab_indices,
      values=one_hot_values,
      dense_shape=new_shape)


def _to_global_document_frequency(
    x: tf.SparseTensor, vocab_size: Union[int, tf.Tensor]) -> tf.Tensor:
  """Summerizes term/doc one-hot tensor to get document frequency of each term.

  Args:
    x: a SparseTensor of size (batch_size, vocab_size) and values 0/1 to
      indicate to existence of each term in each document. x is expected to be
      the output of _to_term_document_one_hot.
    vocab_size: A scalar int64 Tensor - the count of vocab used to turn the
      string into int64s including any OOV buckets.

  Returns:
    a tensor with indices as (1, <vocab_index>) and values as the count of
      documents in the entire dataset that contain each vocab term.
  """
  # term_doc_freq is the one-hot encoding of term existence in each document.
  # It is a (batch_size, vocab_size)-shaped, 0/1 valued sparse tensor.
  term_doc_one_hot = _to_term_document_one_hot(x, vocab_size)

  # Reduce sum the one-hot tensor within each mini batch to get one
  # (1, vocab_size)-shaped sparse tensor for each mini batch, with the value
  # being the count of documents containing each term in that batch.
  count_docs_with_term = tf.sparse.reduce_sum(
      term_doc_one_hot, axis=0, keepdims=True)

  # Sum up all batches to get a (1, vocab_size)-shaped sparse tensor storing
  # count of documents containing each term across the entire dataset.
  return analyzers.sum(count_docs_with_term, reduce_instance_dims=False)
