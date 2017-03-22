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


def scale_to_0_1(x):
  """Returns a column which is the input column scaled to have range [0,1].

  Args:
    x: A `Column` representing a numeric value.

  Returns:
    A `Column` representing the input column scaled to [0, 1].
  """

  # A TITO function that scales x.
  def scale(x, min_value, max_value):
    return (x - min_value) / (max_value - min_value)

  return api.map(scale, x, analyzers.min(x), analyzers.max(x))


def string_to_int(x, default_value=-1, top_k=None, frequency_threshold=None):
  """Generates a vocabulary for `x` and maps it to an integer with this vocab.

  Args:
    x: A `Column` representing a string value or values.
    default_value: The value to use for out-of-vocabulary values.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      frequency is >= to the supplied threshold. If set to None, the full
      vocabulary is generated.

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

  def map_to_int(x, vocab):
    """Maps string tensor into indexes using vocab.

    It uses a dummy vocab when the input vocab is empty.

    Args:
      x : a Tensor/SparseTensor of string.
      vocab : a Tensor/SparseTensor containing unique string values within x.

    Returns:
      a Tensor/SparseTensor of indexes (int) of the same shape as x.
    """

    def fix_vocab_if_needed(vocab):
      num_to_add = 1 - tf.minimum(tf.size(vocab), 1)
      return tf.concat([
          vocab, tf.fill(
              tf.reshape(num_to_add, (1,)), '__dummy_value__index_zero__')
      ], 0)

    table = lookup.string_to_index_table_from_tensor(
        fix_vocab_if_needed(vocab), default_value=default_value)
    return table.lookup(x)

  return api.map(map_to_int, x,
                 analyzers.uniques(
                     x, top_k=top_k, frequency_threshold=frequency_threshold))
