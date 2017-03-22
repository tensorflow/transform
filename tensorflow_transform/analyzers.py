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
"""TF.Transform analyzers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_transform import api


def min(x):  # pylint: disable=redefined-builtin
  """Computes the minimum of a `Column`.

  Args:
    x: An input `Column' wrapping a `Tensor`.

  Returns:
    A `Statistic`.
  """
  if not isinstance(x.tensor, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x.tensor)


  # pylint: disable=protected-access
  return api._AnalyzerOutput(tf.placeholder(x.tensor.dtype, ()),
                             api.CanonicalAnalyzers.MIN, [x], {})


def max(x):  # pylint: disable=redefined-builtin
  """Computes the maximum of a `Column`.

  Args:
    x: An input `Column' wrapping a `Tensor`.

  Returns:
    A `Statistic`.
  """
  if not isinstance(x.tensor, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x.tensor)

  # pylint: disable=protected-access
  return api._AnalyzerOutput(tf.placeholder(x.tensor.dtype, ()),
                             api.CanonicalAnalyzers.MAX, [x], {})


def sum(x):  # pylint: disable=redefined-builtin
  """Computes the sum of a `Column`.

  Args:
    x: An input `Column' wrapping a `Tensor`.

  Returns:
    A `Statistic`.
  """
  if not isinstance(x.tensor, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x.tensor)

  # pylint: disable=protected-access
  return api._AnalyzerOutput(tf.placeholder(x.tensor.dtype, ()),
                             api.CanonicalAnalyzers.SUM, [x], {})


def size(x):
  """Computes the total size of instances in a `Column`.

  Args:
    x: An input `Column' wrapping a `Tensor`.

  Returns:
    A `Statistic`.
  """
  if not isinstance(x.tensor, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x.tensor)

  # Note: Calling `sum` defined in this module, not the builtin.
  return sum(api.map(tf.ones_like, x))


def mean(x):
  """Computes the mean of the values in a `Column`.

  Args:
    x: An input `Column' wrapping a `Tensor`.

  Returns:
    A `Column` with an underlying `Tensor` of shape [1], containing the mean.
  """
  if not isinstance(x.tensor, tf.Tensor):
    raise TypeError('Expected a Tensor, but got %r' % x.tensor)

  # Note: Calling `sum` defined in this module, not the builtin.
  return api.map_statistics(tf.divide, sum(x), size(x))


def uniques(x, top_k=None, frequency_threshold=None):
  """Returns the unique values of the input tensor.

  Computes The unique values taken by the input column `x`, which can be backed
  by a `Tensor` or `SparseTensor` of any size.  The unique values will be
  aggregated over all dimensions of `x` and all instances.

  The unique values are sorted by decreasing frequency and then decreasing
  value.

  Args:
    x: An input `Column` wrapping a `Tensor` or `SparseTensor`.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      frequency is >= to the supplied threshold. If set to None, the full
      vocabulary is generated.

  Returns:
    The unique values of `x`.

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
      raise ValueError('frequency_threshold must be non-negative, but got: %r' %
                       frequency_threshold)

  if isinstance(x.tensor, tf.SparseTensor):
    values = x.tensor.values
  else:
    values = x.tensor
  arg_dict = {'top_k': top_k, 'frequency_threshold': frequency_threshold}
  # Create output placeholder whose shape is a 1-d tensor of unkown size.
  # pylint: disable=protected-access
  return api._AnalyzerOutput(tf.placeholder(values.dtype, (None,)),
                             api.CanonicalAnalyzers.UNIQUES, [x], arg_dict)
