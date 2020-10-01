# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Utilities for inspecting users' preprocessing_fns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import graph_tools
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
from tensorflow_transform import tf2_utils


class _SourcedTensorsVisitor(nodes.Visitor):
  """Visitor used to extract tensors that are inputs to `TensorSource` nodes."""

  def __init__(self):
    self.sourced_tensors = []

  def visit(self, operation_def, input_values):
    if isinstance(operation_def, analyzer_nodes.TensorSource):
      for tensor in operation_def.tensors:
        self.sourced_tensors.append(tensor)
    return nodes.OperationNode(operation_def, input_values).outputs

  def validate_value(self, value):
    assert isinstance(value, nodes.ValueNode)


def get_analyze_input_columns(preprocessing_fn, specs, force_tf_compat_v1=True):
  """Return columns that are required inputs of `AnalyzeDataset`.

  Args:
    preprocessing_fn: A tf.transform preprocessing_fn.
    specs: A dict of feature name to tf.TypeSpecs. If `force_tf_compat_v1` is
      True, this can also be feature specifications.
    force_tf_compat_v1: (Optional) If `True`, use Tensorflow in compat.v1 mode.
      Defaults to `True`.

  Returns:
    A list of columns that are required inputs of analyzers.
  """
  if not force_tf_compat_v1:
    assert all([isinstance(s, tf.TypeSpec) for s in specs.values()]), specs
  graph, structured_inputs, _ = (
      impl_helper.trace_preprocessing_function(
          preprocessing_fn,
          specs,
          use_tf_compat_v1=tf2_utils.use_tf_compat_v1(force_tf_compat_v1)))

  tensor_sinks = graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  visitor = _SourcedTensorsVisitor()
  for tensor_sink in tensor_sinks:
    nodes.Traverser(visitor).visit_value_node(tensor_sink.future)

  analyze_input_tensors = graph_tools.get_dependent_inputs(
      graph, structured_inputs, visitor.sourced_tensors)
  return list(analyze_input_tensors.keys())


def get_transform_input_columns(preprocessing_fn,
                                specs,
                                force_tf_compat_v1=True):
  """Return columns that are required inputs of `TransformDataset`.

  Args:
    preprocessing_fn: A tf.transform preprocessing_fn.
    specs: A dict of feature name to tf.TypeSpecs. If `force_tf_compat_v1` is
      True, this can also be feature specifications.
    force_tf_compat_v1: (Optional) If `True`, use Tensorflow in compat.v1 mode.
      Defaults to `True`.

  Returns:
    A list of columns that are required inputs of the transform `tf.Graph`
    defined by `preprocessing_fn`.
  """
  if not force_tf_compat_v1:
    assert all([isinstance(s, tf.TypeSpec) for s in specs.values()]), specs
  graph, structured_inputs, structured_outputs = (
      impl_helper.trace_preprocessing_function(
          preprocessing_fn,
          specs,
          use_tf_compat_v1=tf2_utils.use_tf_compat_v1(force_tf_compat_v1)))

  transform_input_tensors = graph_tools.get_dependent_inputs(
      graph, structured_inputs, structured_outputs)
  return list(transform_input_tensors.keys())
