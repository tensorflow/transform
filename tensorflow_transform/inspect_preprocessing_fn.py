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


def get_analyze_input_columns(preprocessing_fn, feature_spec):
  """Return columns that are required inputs of `AnalyzeDataset`.

  Args:
    preprocessing_fn: A tf.transform preprocessing_fn.
    feature_spec: A dict of feature name to feature specification.

  Returns:
    A list of columns that are required inputs of analyzers.
  """
  with tf.Graph().as_default() as graph:
    input_signature = impl_helper.feature_spec_as_batched_placeholders(
        feature_spec)
    _ = preprocessing_fn(input_signature.copy())

    tensor_sinks = graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
    visitor = _SourcedTensorsVisitor()
    for tensor_sink in tensor_sinks:
      nodes.Traverser(visitor).visit_value_node(tensor_sink.future)

    analyze_input_tensors = graph_tools.get_dependent_inputs(
        graph, input_signature, visitor.sourced_tensors)
    return analyze_input_tensors.keys()


def get_transform_input_columns(preprocessing_fn, feature_spec):
  """Return columns that are required inputs of `TransformDataset`.

  Args:
    preprocessing_fn: A tf.transform preprocessing_fn.
    feature_spec: A dict of feature name to feature specification.

  Returns:
    A list of columns that are required inputs of the transform `tf.Graph`
    defined by `preprocessing_fn`.
  """
  with tf.Graph().as_default() as graph:
    input_signature = impl_helper.feature_spec_as_batched_placeholders(
        feature_spec)
    output_signature = preprocessing_fn(input_signature.copy())
    transform_input_tensors = graph_tools.get_dependent_inputs(
        graph, input_signature, output_signature)
    return transform_input_tensors.keys()
