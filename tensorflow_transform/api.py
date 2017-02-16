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
"""The core public API of TFTransform.  Provide functions to transform columns.

The core TFTransform API provides a way for the user to construct an abstract
representation of a transformation from an input data set to an output data set.
This function is constructed using the provided functions that operate on the
`Column` and `Statistic` classes.  In particular, the user combines these
functions to build a function that accepts and returns a dictionary whose
keys are strings and whose values are `Column`s or `Statistics`, e.g.

import tensorflow_transform as tft

def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']

  # Apply the `mean` analyzer to obtain the mean x.
  x_mean = tft.mean(x)

  # Apply `map` together with a function that accepts and returns tensors, to
  # subtract the mean.
  x_normalized = tft.map(lambda x, mean: x - mean, x, x_mean)

  # Return a new dictionary containing x normalized, and y unchanged
  return {
    'x_normalized': x_normalized,
    'y': y
  }

This user-defined function then must be run using an implementation which takes
the user defined preprocessing function and runs it using some distributed
computation framework.  The canonical beam implementation uses Apache Beam as
the underlying framework.  See beam/impl.py for how to use the Beam
implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf


class Statistic(object):
  """Statistic represents a statistic of a column in a preprocessing function.

  The result of a summary statistic (e.g. mean, sum or a vocabulary) computed
  on one or more (possibly transformed) columns.

  Args:
    tensor: The `Tensor` or `SparseTensor` that will represent the statistic.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, tensor):
    self._tensor = tensor

  @property
  def tensor(self):
    return self._tensor


class Column(object):
  """A Column represents a column in a preprocessing function.

  Columns are either the columns of the input dataset or a column constructed
  by applying some row-wise transformation to the input dataset.

  Args:
    tensor: A `Tensor` or `SparseTensor` that will represent the column.
    schema: A `ColumnSchema` describing the column. Defaults to None.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, tensor, schema=None):
    self._tensor = tensor
    self._schema = schema

  @property
  def tensor(self):
    return self._tensor

  @property
  def schema(self):
    return self._schema

  @schema.setter
  def schema(self, schema):
    self._schema = schema


class _AnalyzerOutput(Statistic):
  """A Column containing the output of a transformation.

  A `_AnalyzerOutput` is defined by zero or more inputs (which may be `Column`s
  or `Statistic`s) and an analyzer applied to them.

  Args:
    tensor: The `Tensor` or `SparseTensor` that will represent the statistic.
    analyzer_name: The name of the analyzer to be applied.
    inputs: A list of `Column` or `Statistic`s to which the analyzer should
        be applied.
    args_dict: Extra arguments for the analyzer.
  """

  def __init__(self, tensor, analyzer_name, inputs, args_dict):
    super(_AnalyzerOutput, self).__init__(tensor)
    self._analyzer_name = analyzer_name
    self._inputs = inputs
    self._args_dict = args_dict

  @property
  def analyzer_name(self):
    return self._analyzer_name

  @property
  def inputs(self):
    return self._inputs

  @property
  def args_dict(self):
    return self._args_dict


class _InputColumn(Column):
  """A Column representing a column in the input dataset.

  Args:
    placeholder: The `Tensor` or `SparseTensor` that will represent the column.
    schema: A `ColumnSchema` describing the column.
  """

  def __init__(self, placeholder, schema):
    # In order to avoid a bug where import_graph_def fails when the input_map
    # and return_elements of an imported graph are the same (b/34288791), we
    # avoid using the placeholder of an input column as an output of a graph.
    # We do this by applying tf.identity to the placeholder and using the output
    # of tf.identity as the tensor representing the output of this column, thus
    # preventing the placeholder from being used as both an input and an output.
    if isinstance(placeholder, tf.SparseTensor):
      tensor = tf.SparseTensor(indices=tf.identity(placeholder.indices),
                               values=tf.identity(placeholder.values),
                               dense_shape=tf.identity(placeholder.dense_shape))
    else:
      tensor = tf.identity(placeholder)
    super(_InputColumn, self).__init__(tensor, schema)
    self._placeholder = placeholder

  @property
  def placeholder(self):
    return self._placeholder


class _TransformedColumn(Column):
  """A Column containing the output of a transformation.

  A `_TransformedColumn` is defined by zero or more inputs (which may be
  `Column`s or `Statistic`s) and a function that accepts `Tensor`s or
  `SparseTensor`s as arguments and returns a `Tensor` or `SparseTensor`.

  Args:
    tensor: The `Tensor` or `SparseTensor` that will represent the column.
    fn: A function that accepts one or more `Tensor`s or `SparseTensor`s and
      returns a `Tensor` or `SparseTensor`.
    inputs: A list of `Column` or `Statistic`s to which the transform should
        be applied.
  """

  def __init__(self, tensor, fn, inputs):
    # Transforms are required to produce an output with a batch dimension. The
    # assertions below attempt to verify this. In the case of dense tensors the
    # check occurs statically if possible but falls back on a runtime check. In
    # the case of sparse tensors, the check happens at runtime.
    min_tensor_rank = 1
    if isinstance(tensor, tf.SparseTensor):
      with tf.control_dependencies(
          [tf.assert_greater_equal(tf.size(tensor.dense_shape),
                                   min_tensor_rank)]):
        tensor = tf.SparseTensor(indices=tf.identity(tensor.indices),
                                 values=tensor.values,
                                 dense_shape=tensor.dense_shape)
    else:
      with tf.control_dependencies(
          [tf.assert_rank_at_least(tensor, min_tensor_rank)]):
        tensor = tf.identity(tensor)
    super(_TransformedColumn, self).__init__(tensor)
    self._fn = fn
    self._inputs = inputs

  @property
  def fn(self):
    return self._fn

  @property
  def inputs(self):
    return self._inputs


class _TransformedStatistic(Statistic):
  """A Statistic containing the output of a transformation.

  A `_TransformedStatistic` is defined by zero or more input `Statistic`s and a
  function that accepts `Tensor`s or `SparseTensor`s as arguments and returns a
  `Tensor` or `SparseTensor`.

  Args:
    tensor: The `Tensor` or `SparseTensor` that will represent the statistic.
    fn: A function that accepts one or more `Tensor`s or `SparseTensor`s and
      returns a `Tensor` or `SparseTensor`.
    inputs: A list of `Statistic`s to which the transform should be applied.
  """

  def __init__(self, tensor, fn, inputs):
    super(_TransformedStatistic, self).__init__(tensor)
    self._fn = fn
    self._inputs = inputs

  @property
  def fn(self):
    return self._fn

  @property
  def inputs(self):
    return self._inputs


def map(fn, *args):  # pylint: disable=redefined-builtin
  """Applies a function to some columns.

  Applies a function to some columns given by the argument list. The number
  of arguments should match the number of inputs to the function. The args can
  also contain `Statistic`s in which case the values are the same for each
  batch.

  Args:
    fn: A function that accepts one or more `Tensor`s or `SparseTensor`s and
      returns a `Tensor` or `SparseTensor`.
    *args: The list of `Column`s or `Statistic`s to apply the arguments to.

  Returns:
    A `Column` representing the application of the function.
  """
  input_tensors = [arg.tensor for arg in args]
  output_tensor = fn(*input_tensors)
  return _TransformedColumn(output_tensor, fn, args)


def map_statistics(fn, *args):
  """Applies a function to some statistics.

  Applies a function to some `Statistics`s given by the argument list. The
  number of arguments should match the number of inputs to the function.

  Args:
    fn: A function that accepts one or more `Tensor`s or `SparseTensor`s and
      returns a `Tensor` or `SparseTensor`.
    *args: The list of `Statistic`s to apply the arguments to.

  Returns:
    A `Statistic` representing the application of the function.
  """
  input_tensors = [arg.tensor for arg in args]
  output_tensor = fn(*input_tensors)
  return _TransformedStatistic(output_tensor, fn, args)


class CanonicalAnalyzers(object):
  """Enum-like class containing names of canonical analyzers."""

  MIN = 'min'
  MAX = 'max'
  SUM = 'sum'
  UNIQUES = 'uniques'
