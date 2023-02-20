# Copyright 2021 Google Inc. All Rights Reserved.
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
"""Experimental APIs to get annotations."""

from typing import Sequence, Union

import tensorflow as tf
from tensorflow_transform import annotators
from tensorflow_transform import schema_inference

from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'get_vocabulary_size_by_name',
    'annotate_sparse_output_shape',
    'annotate_true_sparse_output',
]


def get_vocabulary_size_by_name(vocab_filename: str) -> tf.Tensor:
  # pyformat: disable
  """Gets the size of a vocabulary created using `tft.vocabulary`.

  This is the number of keys in the output `vocab_filename` and does not include
  number of OOV buckets.

  Args:
    vocab_filename: The name of the vocabulary file whose size is to be
      retrieved.

  Example:

  >>> def preprocessing_fn(inputs):
  ...   num_oov_buckets = 1
  ...   x_int = tft.compute_and_apply_vocabulary(
  ...     inputs['x'], vocab_filename='my_vocab',
  ...     num_oov_buckets=num_oov_buckets)
  ...   depth = (
  ...     tft.experimental.get_vocabulary_size_by_name('my_vocab') +
  ...     num_oov_buckets)
  ...   x_encoded = tf.one_hot(
  ...     x_int, depth=tf.cast(depth, tf.int32), dtype=tf.int64)
  ...   return {'x_encoded': x_encoded}
  >>> raw_data = [dict(x='foo'), dict(x='foo'), dict(x='bar')]
  >>> feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string))
  >>> raw_data_metadata = tft.DatasetMetadata.from_feature_spec(feature_spec)
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'x_encoded': array([1, 0, 0])}, {'x_encoded': array([1, 0, 0])},
  {'x_encoded': array([0, 1, 0])}]

  Returns:
    An integer tensor containing the size of the requested vocabulary.

  Raises:
    ValueError: if no vocabulary size found for the given `vocab_filename`.

  """
  # pyformat: enable
  vocabulary_sizes_coll = ops.get_default_graph().get_collection(
      annotators.VOCABULARY_SIZE_BY_NAME_COLLECTION)

  result = dict(vocabulary_sizes_coll).get(vocab_filename, None)

  if result is None:
    raise ValueError(
        f'Vocabulary size not found for {vocab_filename}. If this vocabulary '
        'was created using `tft.vocabulary`, this should be the same as the '
        '`vocab_filename` argument passed to it.')

  return result


def annotate_sparse_output_shape(
    tensor: tf.SparseTensor, shape: Union[Sequence[int], tf.Tensor]):
  """Annotates a sparse output to have a given dense_shape.

  Args:
    tensor: An `SparseTensor` to be annotated.
    shape: A dense_shape to annotate `tensor` with. Note that this shape does
      not include batch_size.
  """
  if not isinstance(shape, tf.Tensor):
    if (tensor.shape.rank > 1 and tensor.shape.rank - 1 != len(shape)) or (
        tensor.shape.rank == 1 and len(shape) != 1):
      raise ValueError(
          f'Annotated shape {shape} was expected to have rank'
          f' {tensor.shape.rank - 1}')
    if not all(a is None or a <= b for a, b in zip(tensor.shape[1:], shape)):
      raise ValueError(
          f'Shape {shape} cannot contain annotated tensor {tensor}')
    shape = tf.convert_to_tensor(shape, dtype=tf.int64)
  elif shape.shape.rank > 1 or (
      shape.shape.rank == 1 and shape.shape[0] != tensor.shape.rank - 1):
    raise ValueError(
        f'Annotation shape has rank {shape.shape.rank} but expected to have'
        f' rank {tensor.shape.rank - 1}')
  if shape.shape.rank < 1:
    shape = tf.expand_dims(shape, -1)
  # There's currently no way to override SparseTensor.dense_shape directly,
  # unless composing and returning a new SparseTensor.
  tensor._dense_shape = tf.concat(  # pylint: disable=protected-access
      [tf.expand_dims(tensor.dense_shape[0], -1), tf.cast(shape, tf.int64)],
      axis=0)
  schema_inference.annotate_sparse_output_shape(tensor, shape)


def annotate_true_sparse_output(tensor: tf.SparseTensor):
  """Annotates a sparse output to be truely sparse and not varlen."""
  schema_inference.annotate_true_sparse_output(tensor)
