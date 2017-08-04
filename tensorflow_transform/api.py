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

The core tf.Transform API provides a way for the user to construct a function
that accepts and returns `Tensor`s.  This function is built by composing regular
functions built from TensorFlow ops, as well as special functions we refer to as
`Analyzer`s.  `Analyzer`s behave similarly to TensorFlow ops but require a full
pass over the whole dataset to compute their output value.

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

import tensorflow as tf
from tensorflow_transform import analyzers

FUNCTION_APPLICATION_COLLECTION = 'tft_function_applications'


class FunctionApplication(object):
  """Contains data to help tf.Transform keep track of function applications."""

  def __init__(self, fn, args):
    def _decompose_tensors(tensor_list):
      result = []
      for tensor in tensor_list:
        if isinstance(tensor, tf.SparseTensor):
          result.append(tensor.indices)
          result.append(tensor.values)
          result.append(tensor.dense_shape)
        else:
          result.append(tensor)
      return result

    def _copy_tensor(tensor):
      if isinstance(tensor, tf.SparseTensor):
        return tf.SparseTensor(
            tf.identity(tensor.indices),
            tf.identity(tensor.values),
            tf.identity(tensor.dense_shape))
      else:
        return tf.identity(tensor)

    # Apply fn to its args, keeping track of any table initializers that are
    # added while fn is running, and also checking that no analyzers are added
    # while fn is running.
    all_table_initializers = tf.get_collection_ref(
        tf.GraphKeys.TABLE_INITIALIZERS)
    all_analyzers = tf.get_collection_ref(analyzers.ANALYZER_COLLECTION)
    original_num_table_initializers = len(all_table_initializers)
    original_num_analyzers = len(all_analyzers)
    output = fn(*args)
    if len(all_analyzers) != original_num_analyzers:
      raise ValueError(
          'One or more `Analyzer`s were created while inside '
          'FunctionApplication.__init__')

    # Set inputs and outputs of this op, flattening inputs and outputs into a
    # list of tensors, but storing outputs in the original format for the return
    # value of `apply_function`.
    self._table_initializers = all_table_initializers[
        original_num_table_initializers:]
    self._inputs = _decompose_tensors(args)
    # When traversing the graph, there isn't a clean way to handle `Map`s whose
    # inputs and outputs overlap.  Therefore we apply tf.identity to all outputs
    # to ensure the outputs and inputs don't overlap.
    if isinstance(output, tuple):
      self._user_output = [_copy_tensor(tensor) for tensor in output]
      self._outputs = _decompose_tensors(self._user_output)
    else:
      self._user_output = _copy_tensor(output)
      self._outputs = _decompose_tensors([self._user_output])

    tf.add_to_collection(FUNCTION_APPLICATION_COLLECTION, self)

  @property
  def user_output(self):
    """Outputs in the same format as the original return value of fn."""
    return self._user_output

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def table_initializers(self):
    return self._table_initializers


def apply_function(fn, *args):
  """Apply a function to its args in a way that tf.Transform can track.

  Functions that involve tables or control flow ops must be wrapped in
  apply_function.  E.g.

  def preprocessing_fn(inputs):
    ...
    label = inputs['label']
    ...
    def _convert_label(x):
      table = lookup.index_table_from_tensor(['bad', 'good'])
      return table.lookup(x)

    label = api.apply_function(_convert_label, x) # Works
    label = _convert_label(x)  # Doesn't work.

  The reason this function is needed is so that tf.Transform knows to treat the
  wrapped function as a single unit, and not try to trace the control flow in
  TensorFlow by analyzing the ops that make up the function.  This function
  does not need to be used when calling helper functions in mappers.py as those
  functions already do the wrapping themselves.

  Args:
    fn: The function to apply
    *args: The arguments to apply `fn` to.

  Returns:
    The results of applying fn.
  """
  return FunctionApplication(fn, args).user_output


_TF_METADATA_TENSORS_COLLECTION = 'tft_metadata_tensors'
_TF_METADATA_COLUMN_SCHEMAS_COLLECTION = 'tft_metadata_schemas'


def set_column_schema(tensor, column_schema):
  """Sets the schema of a `Tensor` or `SparseTensor`."""
  graph = tensor.graph
  graph.add_to_collection(_TF_METADATA_TENSORS_COLLECTION, tensor)
  graph.add_to_collection(_TF_METADATA_COLUMN_SCHEMAS_COLLECTION, column_schema)


def get_column_schemas(graph):
  """Gets a dict from `Tensor` or `SparseTensor`s to `ColumnSchema`s."""
  return dict(zip(
      graph.get_collection(_TF_METADATA_TENSORS_COLLECTION),
      graph.get_collection(_TF_METADATA_COLUMN_SCHEMAS_COLLECTION)))
