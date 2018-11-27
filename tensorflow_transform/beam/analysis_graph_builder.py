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
"""Functions to create the implementation graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import graph_tools
from tensorflow_transform import nodes
from tensorflow_transform.beam import beam_nodes


def _tensor_name(tensor):
  """Get a name of a tensor without trailing ":0" when relevant."""
  # tensor.name is unicode in Python 3 and bytes in Python 2 so convert to
  # bytes here.
  name = str(tensor.name)
  return name[:-2] if name.endswith(':0') else name


class _ReadyVisitor(nodes.Visitor):
  """Visitor to determine if a node is ready to run."""

  def __init__(self, graph_analyzer):
    self._graph_analyzer = graph_analyzer

  def visit(self, operation_def, input_values):
    if isinstance(operation_def, analyzer_nodes.TensorSource):
      is_ready = all(self._graph_analyzer.ready_to_run(tensor)
                     for tensor in operation_def.tensors)
    else:
      is_ready = all(input_values)
    return (is_ready,) * operation_def.num_outputs

  def validate_value(self, value):
    assert isinstance(value, bool)


class _TranslateVisitor(nodes.Visitor):
  """Visitor that translates the operation graph.

  The original graph is defined by the user in the preprocessing_fn.  The
  translated graph represents a Beam pipeline.
  """

  def __init__(self):
    self.phase = None
    self.extracted_values_dict = None
    self.intermediate_output_signature = None

  def visit(self, operation_def, input_values):
    if isinstance(operation_def, analyzer_nodes.TensorSource):
      tensors = operation_def.tensors
      label = operation_def.label
      # Add tensor to signature so it gets produced by the SavedModel.
      for tensor in tensors:
        self.intermediate_output_signature[_tensor_name(tensor)] = tensor
      keys = tuple(map(_tensor_name, tensors))
      output = nodes.apply_operation(
          beam_nodes.ExtractFromDict, self.extracted_values_dict,
          keys=keys, label=label)
      return (output,)
    else:
      return nodes.OperationNode(operation_def, input_values).outputs

  def validate_value(self, value):
    assert isinstance(value, nodes.ValueNode)


def build(graph, input_signature, output_signature):
  """Returns a list of `Phase`s describing how to execute the pipeline.

  The default graph is assumed to contain some `Analyzer`s which must be
  executed by doing a full pass over the dataset, and passing the inputs for
  that analyzer into some implementation, then taking the results and replacing
  the `Analyzer`s outputs with constants in the graph containing these results.

  The execution plan is described by a list of `Phase`s.  Each phase contains
  a list of `Analyzer`s, which are the `Analyzer`s which are ready to run in
  that phase, together with a list of ops, which are the table initializers that
  are ready to run in that phase.

  An `Analyzer` or op is ready to run when all its dependencies in the graph
  have been computed.  Thus if the graph is constructed by

  def preprocessing_fn(input)
    x = inputs['x']
    scaled_0 = x - tft.min(x)
    scaled_0_1 = scaled_0 / tft.max(scaled_0)

  Then the first phase will contain the analyzer corresponding to the call to
  `min`, because `x` is an input and so is ready to compute in the first phase,
  while the second phase will contain the analyzer corresponding to the call to
  `max` since `scaled_1` depends on the result of the call to `tft.min` which
  is computed in the first phase.

  More generally, we define a level for each op and each `Analyzer` by walking
  the graph, assigning to each operation the max level of its inputs, to each
  `Tensor` the level of its operation, unless it's the output of an `Analyzer`
  in which case we assign the level of its `Analyzer` plus one.

  Args:
    graph: A `tf.Graph`.
    input_signature: A dict whose keys are strings and values are `Tensor`s or
        `SparseTensor`s.
    output_signature: A dict whose keys are strings and values are `Tensor`s or
        `SparseTensor`s.

  Returns:
    A list of `Phase`s.

  Raises:
    ValueError: if the graph cannot be analyzed.
  """
  tensor_sinks = graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  graph.clear_collection(analyzer_nodes.TENSOR_REPLACEMENTS)
  phase = 0
  tensor_bindings = []
  sink_tensors_ready = {tensor_sink.tensor: False
                        for tensor_sink in tensor_sinks}
  translate_visitor = _TranslateVisitor()
  translate_traverser = nodes.Traverser(translate_visitor)

  while not all(sink_tensors_ready.values()):
    # Determine which table init ops are ready to run in this phase
    # Determine which keys of pending_tensor_replacements are ready to run
    # in this phase, based in whether their dependencies are ready.
    graph_analyzer = graph_tools.InitializableGraphAnalyzer(
        graph, input_signature.values(), sink_tensors_ready)
    ready_traverser = nodes.Traverser(_ReadyVisitor(graph_analyzer))

    # Now create and apply a SavedModel with all tensors in tensor_bindings
    # bound, which outputs all the tensors in the required tensor tuples.
    intermediate_output_signature = collections.OrderedDict()
    saved_model_future = nodes.apply_operation(
        beam_nodes.CreateSavedModel,
        *tensor_bindings,
        table_initializers=tuple(graph_analyzer.ready_table_initializers),
        output_signature=intermediate_output_signature,
        label='CreateSavedModelForAnalyzerInputs[{}]'.format(phase))
    extracted_values_dict = nodes.apply_operation(
        beam_nodes.ApplySavedModel,
        saved_model_future,
        phase=phase,
        label='ApplySavedModel[{}]'.format(phase))

    translate_visitor.phase = phase
    translate_visitor.intermediate_output_signature = (
        intermediate_output_signature)
    translate_visitor.extracted_values_dict = extracted_values_dict
    for tensor, value_node, is_asset_filepath in tensor_sinks:
      # Don't compute a binding/sink/replacement that's already been computed
      if sink_tensors_ready[tensor]:
        continue

      if not ready_traverser.visit_value_node(value_node):
        continue

      translated_value_node = translate_traverser.visit_value_node(value_node)

      name = _tensor_name(tensor)
      tensor_bindings.append(nodes.apply_operation(
          beam_nodes.CreateTensorBinding, translated_value_node,
          tensor=str(tensor.name), is_asset_filepath=is_asset_filepath,
          label='CreateTensorBinding[{}]'.format(name)))
      sink_tensors_ready[tensor] = True

    phase += 1

  # We need to make sure that the representation of this output_signature is
  # deterministic.
  output_signature = collections.OrderedDict(
      sorted(output_signature.items(), key=lambda t: t[0]))

  return nodes.apply_operation(
      beam_nodes.CreateSavedModel,
      *tensor_bindings,
      table_initializers=tuple(
          graph.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)),
      output_signature=output_signature,
      label='CreateSavedModel')
