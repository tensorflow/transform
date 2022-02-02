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

import copy
import itertools
from typing import Collection, Iterable, Mapping, Optional, Tuple

import tensorflow as tf
from tensorflow_transform import common_types
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
# pylint: enable=g-direct-tensorflow-import


def use_tf_compat_v1(force_tf_compat_v1: bool) -> bool:
  """Evaluate from environment variables if TF should be used in compat.v1 mode."""
  major, _, _ = tf.version.VERSION.split('.')
  # TODO(b/160294509): Use tf.compat.v1 when we stop supporting TF 1.15.
  # If tf.enable_v2_behavior has been called, but eager execution has been
  # disabled, force compat v1 behavior. Hence, check
  # `executing_eagerly_outside_functions` as well.
  return (force_tf_compat_v1 or int(major) < 2 or not tf2.enabled() or
          not ops.executing_eagerly_outside_functions())


def strip_and_get_tensors_and_control_dependencies(
    flat_tensor_list: Iterable[tf.Tensor]
) -> Tuple[Iterable[tf.Tensor], Iterable[tf.Operation]]:
  """Strips automatic control dependencies from `flat_tensor_list`.

  Args:
    flat_tensor_list: A flattened list of output tensors from a tf.function.

  Returns:
    A tuple of:
      Tensors from `flat_tensor_list` with control dependencies removed.
      The set of control dependency ops that `flat_tensor_list` depended on.
  """
  # If an automatic control dependency node was added, all tensors in
  # `flat_tensor_list` will be the result of Identity ops with the original
  # tensor as an input and the automatic control dependencies as control inputs.
  if all(tensor.op.type == 'Identity' and len(tensor.op.inputs) == 1
         for tensor in flat_tensor_list):
    control_dependency_ops = [t.op.control_inputs for t in flat_tensor_list]
    return ([t.op.inputs[0] for t in flat_tensor_list],
            set(itertools.chain(*control_dependency_ops)))
  else:
    return flat_tensor_list, set()


def supply_missing_tensor(batch_size: int, tensor_shape: tf.TensorShape,
                          tensor_dtype: tf.DType) -> tf.Tensor:
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


def supply_missing_inputs(
    structured_inputs: Mapping[str, common_types.TensorType],
    batch_size: int,
    missing_keys: Optional[Collection[str]] = None
) -> Mapping[str, common_types.TensorType]:
  """Supply inputs for unfed features.

  Supports only tf.Tensor, tf.SparseTensor and tf.RaggedTensor.

  Note: Since this returns placeholders, it should be called from within a graph
  context.

  Args:
    structured_inputs: a dict from keys to batches of placeholder graph tensors.
    batch_size: an integer representing the size of the batch returned.
    missing_keys: (Optional) a subset of the keys of `structured_inputs` for
      which concrete tensors need to be supplied. If `None`, tensors are
      supplied for all keys.

  Returns:
    A batch of placeholders with default values having the same structure as in
    `structured_inputs` for the keys in `missing_keys`.
  """
  missing_keys = missing_keys or list(structured_inputs)
  # Return placeholders to ensure that tensor shape is not constrained to the
  # dummy shape of the missing tensor created here during tracing.
  result = {}
  for key in missing_keys:
    tensor = structured_inputs[key]
    if isinstance(tensor, tf.Tensor) or (isinstance(tensor, tf.RaggedTensor) and
                                         tensor.ragged_rank == 0):
      missing_tensor = supply_missing_tensor(batch_size, tensor.shape,
                                             tensor.dtype)
      result[key] = tf.raw_ops.PlaceholderWithDefault(
          input=missing_tensor, shape=tensor.shape)
    elif isinstance(tensor, tf.SparseTensor):
      values = supply_missing_tensor(batch_size, tensor.values.shape,
                                     tensor.values.dtype)
      dense_rank = tensor.shape.ndims
      # Since values is always a 1-D tensor, set index for every ith value in
      # values to be [i 0 0 ...]. Each index should be compatible with the
      # rank of the SparseTensor. Hence, the number of 0s is dense_rank-1.
      actual_batch_size = tf.shape(values)[0]
      indices = tf.stack(
          [tf.range(actual_batch_size, dtype=tf.int64)] +
          [tf.zeros(actual_batch_size, dtype=tf.int64)] * (dense_rank - 1),
          axis=1)
      dense_shape = tf.cast(
          [actual_batch_size] + [1] * (dense_rank - 1), dtype=tf.int64)

      indices = tf.raw_ops.PlaceholderWithDefault(
          input=indices, shape=tensor.indices.shape)
      values = tf.raw_ops.PlaceholderWithDefault(
          input=values, shape=tensor.values.shape)
      dense_shape = tf.raw_ops.PlaceholderWithDefault(
          input=dense_shape, shape=tensor.dense_shape.shape)
      result[key] = tf.SparseTensor(
          indices=indices, values=values, dense_shape=dense_shape)
    elif isinstance(tensor, tf.RaggedTensor):
      # Builds a ragged tensor similar to tf.ragged.placeholder, except with
      # default values for all components.
      ragged_rank = tensor.ragged_rank
      values = supply_missing_tensor(batch_size, tensor.flat_values.shape,
                                     tensor.flat_values.dtype)
      result[key] = tf.raw_ops.PlaceholderWithDefault(
          input=values, shape=tensor.flat_values.shape)
      for _ in range(ragged_rank):
        if isinstance(values, tf.RaggedTensor):
          values_batch_size = values.bounding_shape(axis=0)
        else:
          values_batch_size = tf.shape(values)[0]
        row_splits = tf.range(values_batch_size + 1, dtype=tf.int64)
        values = tf.RaggedTensor.from_row_splits(
            values, row_splits, validate=False)
        row_splits = tf.raw_ops.PlaceholderWithDefault(
            input=row_splits, shape=[None])
        result[key] = tf.RaggedTensor.from_row_splits(
            result[key], row_splits, validate=False)
    else:
      # TODO(b/169666856): Add support for generic CompositeTensors.
      raise ValueError('Received unsupported input tensor type. Only '
                       'dense/sparse/ragged tensors are currently supported.')
  return result


def get_structured_inputs_from_func_graph(
    func_graph: FuncGraph) -> Mapping[str, common_types.TensorType]:
  """Get structured inputs to a FuncGraph.

  Args:
    func_graph: A `FuncGraph` object.

  Returns:
    Input graph tensors of `func_graph` formatted as possibly-nested python
    objects received by it.
  """
  # structured_input_signature is a tuple of (args, kwargs). [0][0] retrieves
  # the structure of the first arg, which for the preprocessing function is
  # the dictionary of features.
  input_signature = func_graph.structured_input_signature[0][0]
  num_captures = len(func_graph.internal_captures +
                     func_graph.deferred_internal_captures)
  # `func_graph.inputs` contains placeholders that represent regular inputs
  # followed by captured inputs. We are only interested in the regular inputs.
  graph_inputs = copy.copy(func_graph.inputs)
  if num_captures > 0:
    graph_inputs = graph_inputs[:-num_captures]
  return tf.nest.pack_sequence_as(
      input_signature, graph_inputs, expand_composites=True)
