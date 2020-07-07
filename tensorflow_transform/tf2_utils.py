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

_STRING_MISSING_VALUE = 'foo'
_NUMERIC_MISSING_VALUE = 1


def _get_default_value_for_dtype(dtype):
  if dtype == tf.string:
    return _STRING_MISSING_VALUE
  elif dtype.is_floating or dtype.is_integer:
    return _NUMERIC_MISSING_VALUE
  else:
    raise ValueError('Received unsupported input dtype: {}'.format(dtype))


def _supply_missing_tensor(batch_size, tensor):
  """Supplies a `tf.Tensor` compatible with `tensor`.

  Supports only string and numeric dtypes.
  Args:
    batch_size: an integer representing the size of the batch returned.
    tensor: a `tf.Tensor`. The tensor returned by this method will have shape
      and dtype compatible with this tensor.

  Returns:
    A batch of tensors with same shape and dtype as `tensor`.
  """
  assert isinstance(tensor, tf.Tensor)
  # Since `tensor` represents input features to Transform, it's rank should
  # always be known. Hence, it is safe to call as_list() on it's shape.
  input_shape = tensor.shape.as_list()
  result_shape = [input_shape[0] or batch_size]

  for s in input_shape[1:]:
    if s is None:
      result_shape = result_shape + [1]
    else:
      result_shape = result_shape + [s]
  return tf.cast(
      tf.fill(result_shape, _get_default_value_for_dtype(tensor.dtype)),
      tensor.dtype)


def supply_missing_inputs(structured_inputs, batch_size, missing_keys):
  """Supply inputs for unfed features.

  Supports only tf.Tensor.
  Args:
    structured_inputs: a dict from keys to batches of placeholder graph tensors.
    batch_size: an integer representing the size of the batch returned.
    missing_keys: a subset of the keys of `structured_inputs` for which concrete
      tensors need to be supplied.

  Returns:
    A batch of tensors with the same structure as in `structured_inputs`
    for the keys in `missing_keys`.
  """
  result = {}
  for key in missing_keys:
    tensor = structured_inputs[key]
    if isinstance(tensor, tf.Tensor):
      result[key] = _supply_missing_tensor(batch_size, tensor)
    else:
      # TODO(b/153663890): Add support for CompositeTensors.
      raise ValueError('Received unsupported input tensor type. Only dense '
                       'tensors are currently supported.')
  return result
