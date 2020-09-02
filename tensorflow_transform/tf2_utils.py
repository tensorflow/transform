# Copyright 2020 Google Inc. All Rights Reserved.
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
"""TF2 utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION

import tensorflow as tf


def supply_missing_tensor(batch_size, tensor_shape, tensor_dtype):
  """Supplies a `tf.Tensor` compatible with `tensor`.

  Supports only string and numeric dtypes.
  Args:
    batch_size: an integer representing the size of the batch returned.
    tensor_shape: a `tf.TensorShape`. The returned tensor will have shape
      compatible with this.
    tensor_dtype: The dtype of the returned tensors.

  Returns:
    A batch of `tf.Tensor` tensors.
  """
  # If tensor rank is 0 or unknown, return a scalar.
  if tensor_shape.ndims is None or tensor_shape.ndims == 0:
    return tf.zeros([], dtype=tensor_dtype)

  input_shape = tensor_shape.as_list()
  result_shape = [input_shape[0] or batch_size]

  for shape in input_shape[1:]:
    if shape is None:
      result_shape = result_shape + [1]
    else:
      result_shape = result_shape + [shape]
  return tf.zeros(result_shape, dtype=tensor_dtype)


def supply_missing_inputs(structured_inputs, batch_size, missing_keys=None):
  """Supply inputs for unfed features.

  Supports only tf.Tensor.
  Args:
    structured_inputs: a dict from keys to batches of placeholder graph tensors.
    batch_size: an integer representing the size of the batch returned.
    missing_keys: (Optional) a subset of the keys of `structured_inputs` for
      which concrete tensors need to be supplied. If `None`, tensors are
      supplied for all keys.

  Returns:
    A batch of tensors with the same structure as in `structured_inputs`
    for the keys in `missing_keys`.
  """
  missing_keys = missing_keys or list(structured_inputs)
  result = {}
  for key in missing_keys:
    tensor = structured_inputs[key]
    if isinstance(tensor, tf.Tensor):
      result[key] = supply_missing_tensor(batch_size, tensor.shape,
                                          tensor.dtype)
    elif isinstance(tensor, tf.SparseTensor):
      values = supply_missing_tensor(batch_size, tensor.values.shape,
                                     tensor.values.dtype)
      # TODO(b/149997088): assert this complies with tensor.dense_shape.
      indices = tf.cast(
          tf.stack([tf.range(1, dtype=tf.int32),
                    tf.zeros(1, dtype=tf.int32)],
                   axis=1),
          dtype=tf.int64)

      result[key] = tf.SparseTensor(
          indices=indices, values=values, dense_shape=(1, 1))
    else:
      # TODO(b/153663890): Add support for CompositeTensors.
      raise ValueError('Received unsupported input tensor type. Only dense '
                       'tensors are currently supported.')
  return result
