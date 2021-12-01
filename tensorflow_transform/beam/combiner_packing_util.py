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
"""Functions to perform combiner packing optimization.

Packing accumulate combines:
a) First, we visit the TFT graph to gather all the combine accumulate nodes that
   can be packed under the same grandparent node (the parent is an
   ExtractFronDict node which will now follow the accumulate node).
b) Second, we visit the graph to replace the individual combine accumulate nodes
   with the packed node.

Packing merge combines:
a) First, we visit the TFT graph to gather all the combine merges that can be
   packed (i.e., all the combine merges within a TFT phase). We currently do
   this packing only when there is a single phase pre-processing function.
b) Since the inputs to the flatten node (which flattens the output of the
   combine accumulates) before the packed merge come from different paths, we
   add redundant flatten and packed merge nodes as and when we visit a new input
   of this flatten node. At the end of this traversal, we would have one final
   packed merge node with a corresponding flatten node having all the needed
   inputs, and in addition to this we would have a set of redundant packed merge
   and flatten nodes which needs to be removed.
c) Finally, we remove the redundant flatten and packed merge nodes.

"""

import collections

from tensorflow_transform import analyzer_nodes
from tensorflow_transform import nodes
from tensorflow_transform.beam import beam_nodes
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple


_CombinerOpWrapper = tfx_namedtuple.namedtuple('_CombinerOpWrapper',
                                               ['combiner', 'keys', 'label'])


class _ValidationVisitor(nodes.Visitor):
  """Visitor to determine if a node is ready to run."""

  def __init__(self):
    self._visited_operation_def_labels = set()

  def validate_operation_def(self, operation_def):
    assert operation_def.label not in self._visited_operation_def_labels
    self._visited_operation_def_labels.add(operation_def.label)

  def validate_value(self, value):
    assert isinstance(value, nodes.ValueNode)


class _InspectAccumulateCombineVisitor(_ValidationVisitor):
  """A visitor that inspects the graph and looks for combine nodes.

  As this visitor visits the TFT Beam Graph, we group together all the
  packable combine nodes. Specifically, we look for the following path:
             ExtractFromDict --> CacheableCombineAccumulate
  The combines under the same grand parent can be packed together.
  In this visitor, we group all the packable combines for each unique
  grand parent node and save their reference in the `packable_combines` class
  attribute.
  """

  def __init__(self):
    super().__init__()
    # Group all packable combines. We pack all the combines that have the same
    # grand parent.
    # {grand_parent_label: List of packable _CombinerOpWrapper's}
    self.packable_combines = collections.defaultdict(list)

  def visit(self, operation_def, input_values):
    self.validate_operation_def(operation_def)
    self._maybe_add_packable_combine(operation_def, input_values)
    return nodes.OperationNode(operation_def, input_values).outputs

  def _maybe_add_packable_combine(self, operation_def, input_values):
    # We cannot pack the per-key combine analyzers as the key may be different
    # for each analyzer.
    if not isinstance(operation_def, analyzer_nodes.CacheableCombineAccumulate):
      return
    assert len(input_values) == 1

    # Get the ExtractFromDict parent node of the current
    # CacheableCombineAccumulate node.
    parent = input_values[0].parent_operation
    if not isinstance(parent.operation_def, beam_nodes.ExtractFromDict):
      return
    assert len(parent.inputs) == 1

    # Get the parent of the current ExtractFromDict node.
    grand_parent = parent.inputs[0].parent_operation
    assert isinstance(grand_parent.operation_def, beam_nodes.ApplySavedModel)

    # This is a packable combine.
    grand_parent_label = grand_parent.operation_def.label
    self.packable_combines[grand_parent_label].append(_CombinerOpWrapper(
        combiner=operation_def.combiner,
        keys=parent.operation_def.keys,
        label=operation_def.label))


class _PackAccumulateCombineVisitor(_ValidationVisitor):
  r"""A visitor that packs combine nodes in the graph.

  This visitor takes the grouped combines and performs the packing of those
  combines.
                 Before packing
             GrandParentNode
              /           \
   ExtractFromDict1     ExtractFromDict2
             /              \
         Combine1         Combine2

                 After packing
             GrandParentNode
                     |
               PackedCombine
                 /      \
   ExtractFromDict1'   ExtractFromDict2'

  The ExtractFromDict nodes after packing extracts the accumulator corresponding
  to the individual combines.
  """

  def __init__(self, packable_combines):
    super().__init__()
    self._packable_combines = packable_combines

    self._combine_to_grand_parent = {}
    for grand_parent_label, group in self._packable_combines.items():
      for combine_op in group:
        self._combine_to_grand_parent[combine_op.label] = grand_parent_label

    # Cache the packed combine node.
    # Grand parent node label -> Packed combine node
    self._packed_combine_cache = {}

  def visit(self, operation_def, input_values):
    self.validate_operation_def(operation_def)
    # If we see a combine node which can be packed, create the packed combine
    # node and cache it as we will use the same packed node for all the combines
    # in the group.
    if operation_def.label in self._combine_to_grand_parent:
      return self._get_packed_combine(operation_def, input_values)
    return nodes.OperationNode(operation_def, input_values).outputs

  def _get_packed_combine(self, operation_def, input_values):
    grand_parent_label = self._combine_to_grand_parent[operation_def.label]
    # If we are seeing a combine from a group for the first time, create the
    # the packed combine node and cache it.
    if grand_parent_label not in self._packed_combine_cache:
      # Get the grand parent node of the CacheableCombineAccumulate node.
      # We will make this node as the parent of the
      # PackedCombineAccumulate node.
      assert len(input_values) == 1
      parent_node = input_values[0]
      assert isinstance(parent_node.parent_operation.operation_def,
                        beam_nodes.ExtractFromDict)
      assert len(parent_node.parent_operation.inputs) == 1
      grand_parent_node = parent_node.parent_operation.inputs[0]
      assert (grand_parent_node.parent_operation.operation_def.label ==
              grand_parent_label)
      self._packed_combine_cache[grand_parent_label] = (
          nodes.apply_operation(
              analyzer_nodes.PackedCombineAccumulate,
              grand_parent_node,
              combiners=self._packable_combines[grand_parent_label],
              label='PackedCombineAccumulate[{}]'.format(grand_parent_label)))
    # For the current combine, create the ExtractFromDict node which
    # extracts the accumulator corresponding to this combine from the
    # packed combine output.
    result = nodes.apply_operation(
        beam_nodes.ExtractFromDict,
        self._packed_combine_cache[grand_parent_label],
        keys=operation_def.label, label=operation_def.label)
    return (result,)

_COMBINE_PARENT_NODE_TYPES = (
    beam_nodes.ExtractFromDict, beam_nodes.Flatten, analyzer_nodes.DecodeCache)


class _InspectMergeCombineVisitor(_ValidationVisitor):
  """A visitor that inspects the graph and looks for merge combine nodes."""

  def __init__(self):
    super().__init__()
    # Gather all the packable merge combines.
    # Dict {ExtractCombineMergeOutputs (child of CacheableCombineMerge) label:
    #       _CombinerOpWrapper}
    self.packable_combine_extract_outputs = collections.OrderedDict()

  def visit(self, operation_def, input_values):
    self.validate_operation_def(operation_def)
    self._maybe_add_packable_combine(operation_def, input_values)
    return nodes.OperationNode(operation_def, input_values).outputs

  def _maybe_add_packable_combine(self, operation_def, input_values):
    if not isinstance(operation_def, analyzer_nodes.ExtractCombineMergeOutputs):
      return
    # Verify we have a CacheableCombineMerge parent.
    parent = input_values[0].parent_operation
    if not isinstance(parent.operation_def,
                      analyzer_nodes.CacheableCombineMerge):
      return
    assert len(parent.inputs) == 1
    grand_parent = parent.inputs[0].parent_operation
    # We look for packable combines. Specifically, CacheableCombineMerge nodes
    # whose parent is one of the type in _COMBINE_PARENT_NODE_TYPES.
    if isinstance(grand_parent.operation_def, _COMBINE_PARENT_NODE_TYPES):
      # This is a packable combine.
      self.packable_combine_extract_outputs[operation_def.label] = (
          _CombinerOpWrapper(
              combiner=parent.operation_def.combiner,
              keys=(parent.operation_def.label,),
              label=parent.operation_def.label))


class _PackMergeCombineVisitor(_ValidationVisitor):
  r"""A visitor that inspects the graph and looks for combine nodes.

  This visitor takes the grouped combines and performs the packing of those
  combines.
                 Before packing
             ...               ...
              /                  \
           Combine1             Combine2
             /                     \
  ExtractCombineMergeOutputs1    ExtractCombineMergeOutputs2

                 After packing
             ...        ...
              /           \
           AddKey1         AddKey2
              \             /
               \           /
                \         /
                  Flatten
                     |
                 PackedCombineMerge
               /                    \
     ExtractFromDict1              ExtractFromDict2
              /                        \
  ExtractPackedCombineMergeOutputs1    ExtractPackedCombineMergeOutputs2

  Since the inputs to the final flatten node before the packed merge come from
  different paths, we add redundant flatten and packed merge nodes each time we
  visit a new input of the final flatten node. At the end of this traversal,
  we would have one final packed merge node with a corresponding flatten node
  having all the needed inputs, and in addition to this we would have a set of
  redundant packed merge and flatten nodes which needs to be removed.
  """

  def __init__(self, packable_combine_extract_outputs):
    super().__init__()
    self._packable_combine_extract_outputs = packable_combine_extract_outputs
    # Gather all the input nodes that we need to flatten to be passed as input
    # to the packed merge node.
    self._flatten_inputs = []
    # Keep track of the label of the final packed merge combine node.
    self.final_packed_merge_combine_label = None

  def visit(self, operation_def, input_values):
    self.validate_operation_def(operation_def)
    # We look for the ExtractOutputs node of packable combines
    if operation_def.label in self._packable_combine_extract_outputs:
      return self._add_flatten_placeholder(operation_def, input_values)
    return nodes.OperationNode(operation_def, input_values).outputs

  def _add_flatten_placeholder(self, operation_def, input_values):
    assert isinstance(operation_def, analyzer_nodes.ExtractCombineMergeOutputs)
    parent = input_values[0].parent_operation
    assert isinstance(parent.operation_def,
                      analyzer_nodes.CacheableCombineMerge)
    packed_combine = self._get_packed_combine(
        parent.operation_def, parent.inputs)
    # For the current combine, create the ExtractFromDict node which
    # extracts the accumulator corresponding to this combine from the
    # packed combine output.
    extract_dict_node = nodes.apply_operation(
        beam_nodes.ExtractFromDict,
        packed_combine,
        keys=parent.operation_def.label,
        label='ExtractFromDict[{}]'.format(parent.operation_def.label))
    # Create the new ExtractPackedCombineMergeOutputs node.
    return nodes.apply_multi_output_operation(
        analyzer_nodes.ExtractPackedCombineMergeOutputs,
        extract_dict_node,
        output_tensor_info_list=operation_def.output_tensor_infos,
        label='ExtractPackedCombineMergeOutputs[{}]'.format(
            parent.operation_def.label)
    )

  def _get_packed_combine(self, operation_def, input_values):
    for value in input_values:
      keyed_value = nodes.apply_operation(
          analyzer_nodes.AddKey,
          value,
          key=operation_def.label,
          label='AddKey[{}]'.format(operation_def.label))
      self._flatten_inputs.append(keyed_value)
    # TODO(b/134414978): When we add support for multi-phase merge packing,
    # add phase number to the flatten and packed combine labels.
    flatten_label = 'FlattenInputForPackedCombineMerge[{}]'.format(
        len(self._flatten_inputs))
    flatten_node = nodes.apply_operation(
        beam_nodes.Flatten, *self._flatten_inputs, label=flatten_label)
    packed_combine_label = 'PackedCombineMerge[{}]'.format(
        len(self._flatten_inputs))
    packed_combine = nodes.apply_operation(
        analyzer_nodes.PackedCombineMerge,
        flatten_node,
        combiners=list(self._packable_combine_extract_outputs.values()),
        label=packed_combine_label)
    self.final_packed_merge_combine_label = packed_combine_label
    return packed_combine


_TensorBindingInfo = tfx_namedtuple.namedtuple(
    '_TensorBindingInfo',
    ['intermediate_post_processing_op_defs', 'output_index'])

# Maximum search depth for packed post-processing nodes.
_MAX_PACKED_POST_PROCESSING_DEPTH = 5


class _RemoveRedundantPackedMergeCombineVisitor(_ValidationVisitor):
  """A visitor that inspects the graph and removes redundant merge nodes.

  This visitor removes the redundant flatten and packed merge nodes added
  by the _PackMergeCombineVisitor and reconstructs the descendants of the
  removed nodes with the final flatten and packed merge node.
  """

  def __init__(self, final_packed_merge_combine_label):
    super().__init__()
    self._final_packed_merge_combine_label = final_packed_merge_combine_label
    self._packed_post_processing_nodes_cache = {}

  def visit(self, operation_def, input_values):
    self.validate_operation_def(operation_def)
    if input_values and isinstance(operation_def, beam_nodes.CreateSavedModel):
      # This will only be called once since this is a single phase analysis
      # graph and in that case only the final CreateSavedModel node has inputs.
      return self._remove_redundant_nodes(operation_def, input_values)
    return nodes.OperationNode(operation_def, input_values).outputs

  def _remove_redundant_nodes(self, operation_def, input_values):
    # Input values to be used as input to CreateSavedModel.
    # Since some of the input values are generated from the redundant nodes,
    # those needs to be reconstructed with the final packed merge node.
    reconstructed_input_values = []

    redundant_values, non_redundant_values = (
        self._get_redundant_and_non_redundant_input_values(input_values))

    # Keep track of the final packed merge combine node. For those input nodes
    # which are descendants of the redundant nodes, we would create a new node
    # generated from the final packed merge combine node.
    (final_packed_merge_combine, final_packed_merge_combine_tensor_bindings) = (
        self._get_final_packed_combine_and_tensor_bindings(redundant_values))
    reconstructed_input_values.extend(
        final_packed_merge_combine_tensor_bindings)

    # Add the non-redundant nodes to the input values.
    reconstructed_input_values.extend(non_redundant_values)

    # Keep track of the info needed to reconstruct the descendents of the
    # redundant nodes.
    to_be_created_tensor_bindings = (
        self._get_to_be_created_tensor_bindings_info(redundant_values))

    reconstructed_input_values.extend(self._create_tensor_bindings(
        to_be_created_tensor_bindings, final_packed_merge_combine))
    assert len(input_values) == len(reconstructed_input_values)
    return nodes.OperationNode(
        operation_def, tuple(reconstructed_input_values)).outputs

  def _is_packed_post_processing_node(self,
                                      value_node: nodes.ValueNode) -> bool:
    # ValueNode is considered a packed post-processing node iff
    # PackedCombineMerge node is its ancestor.
    if value_node in self._packed_post_processing_nodes_cache:
      return self._packed_post_processing_nodes_cache[value_node]

    input_nodes = set()
    search_depth = 0
    result = False
    while (value_node.parent_operation.inputs and
           search_depth < _MAX_PACKED_POST_PROCESSING_DEPTH):
      # Post-processing nodes form a tree. Looking only at the first input.
      input_nodes.add(value_node)
      value_node = value_node.parent_operation.inputs[0]
      if isinstance(value_node.parent_operation.operation_def,
                    analyzer_nodes.PackedCombineMerge):
        result = True
        break
      search_depth += 1
    self._packed_post_processing_nodes_cache.update(
        {node: result for node in input_nodes})
    return result

  def _get_redundant_and_non_redundant_input_values(
      self, input_values):
    redundant_values, non_redundant_values = [], []
    for value in input_values:
      assert isinstance(value.parent_operation.operation_def,
                        beam_nodes.CreateTensorBinding)
      # If it's from a packed combine node, this is a redundant value.
      if self._is_packed_post_processing_node(value):
        redundant_values.append(value)
      else:
        non_redundant_values.append(value)
    return redundant_values, non_redundant_values

  def _get_final_packed_combine_and_tensor_bindings(self, input_values):
    final_packed_merge_combine = None
    final_packed_merge_combine_tensor_bindings = []
    for value in input_values:
      # PackedCombineMerge is the first not post-processing node on backwards
      # traversal. Post-processing nodes form a tree, it is enough to iterate
      # through first inputs.
      packed_combine = value.parent_operation.inputs[0]
      while self._is_packed_post_processing_node(packed_combine):
        packed_combine = packed_combine.parent_operation.inputs[0]
      # If the input is generated from the final packed merge node, add it to
      # the filtered inputs and keep track of the node for reconstruction of
      # the other inputs.
      packed_combine_op_def = packed_combine.parent_operation.operation_def
      if (isinstance(packed_combine_op_def, analyzer_nodes.PackedCombineMerge)
          and (packed_combine_op_def.label
               == self._final_packed_merge_combine_label)):
        final_packed_merge_combine = packed_combine
        final_packed_merge_combine_tensor_bindings.append(value)
    return (final_packed_merge_combine,
            final_packed_merge_combine_tensor_bindings)

  def _get_to_be_created_tensor_bindings_info(self, input_values):
    result = []
    for value in input_values:
      intermidiate_post_processing_op_defs = []
      intermidiate_value = value
      output_index = None
      while self._is_packed_post_processing_node(intermidiate_value):
        intermidiate_op_def = intermidiate_value.parent_operation.operation_def
        intermidiate_post_processing_op_defs.append(intermidiate_op_def)
        if isinstance(intermidiate_op_def,
                      analyzer_nodes.ExtractPackedCombineMergeOutputs):
          assert output_index is None
          output_index = intermidiate_value.value_index
        intermidiate_value = intermidiate_value.parent_operation.inputs[0]

      # If the input is not generated from the final packed merge node, keep
      # track of the node for reconstruction of the other inputs.
      if (intermidiate_value.parent_operation.operation_def.label !=
          self._final_packed_merge_combine_label):
        # Store the info needed to reconstruct the input node, including
        # CreateTensorBinding node's input value index.
        result.append(
            _TensorBindingInfo(intermidiate_post_processing_op_defs,
                               output_index))
    return result

  def _create_tensor_bindings(self, to_be_created_tensor_bindings,
                              final_packed_merge_combine):
    labels_to_new_nodes = {}
    def _maybe_create_node(op_def, inputs):
      if op_def.label in labels_to_new_nodes:
        return labels_to_new_nodes[op_def.label]
      new_node = nodes.OperationNode(op_def, inputs).outputs
      labels_to_new_nodes[op_def.label] = new_node
      return new_node

    result = []
    if to_be_created_tensor_bindings:
      assert final_packed_merge_combine is not None
      # Reconstruct the remaining inputs from the final packed merge node.
      for tensor_binding_info in to_be_created_tensor_bindings:
        intermediate_nodes = (final_packed_merge_combine,)
        for op_def in reversed(
            tensor_binding_info.intermediate_post_processing_op_defs):
          intermediate_nodes = _maybe_create_node(op_def, intermediate_nodes)
          if isinstance(op_def,
                        analyzer_nodes.ExtractPackedCombineMergeOutputs):
            intermediate_nodes = (
                intermediate_nodes[tensor_binding_info.output_index],)
        # The last node must be a single CreateTensorBinding.
        assert len(intermediate_nodes) == 1, intermediate_nodes
        assert isinstance(intermediate_nodes[0].parent_operation.operation_def,
                          beam_nodes.CreateTensorBinding), intermediate_nodes[0]
        result.append(intermediate_nodes[0])
    return result


def _update_cache_value_node_references(cache_value_nodes, traverser):
  """Updates value node references in the cache."""
  if cache_value_nodes:
    cache_value_nodes = {
        key: traverser.visit_value_node(value_node)
        for key, value_node in cache_value_nodes.items()
    }
  return cache_value_nodes


def perform_combiner_packing_optimization(saved_model_future,
                                          cache_value_nodes, num_phases):
  """Optimizes the graph by packing possible combine nodes."""
  # Inspect the graph to identify all the packable combines.
  inspect_acc_combine_visitor = _InspectAccumulateCombineVisitor()
  inspect_acc_combine_traverser = nodes.Traverser(inspect_acc_combine_visitor)
  _ = inspect_acc_combine_traverser.visit_value_node(saved_model_future)

  packable_combines = inspect_acc_combine_visitor.packable_combines
  # Do not pack if we have only a single combine in the group.
  packable_combines = {
      label: group for label, group in packable_combines.items()
      if len(group) > 1
  }

  pack_acc_combine_visitor = _PackAccumulateCombineVisitor(packable_combines)
  pack_acc_combine_traverser = nodes.Traverser(pack_acc_combine_visitor)
  saved_model_future = pack_acc_combine_traverser.visit_value_node(
      saved_model_future)

  # Replace cache nodes to point to the corresponding new nodes.
  cache_value_nodes = _update_cache_value_node_references(
      cache_value_nodes, pack_acc_combine_traverser)

  # TODO(b/134414978): Consider also packing the merges even when we have
  # multiple phases.
  if num_phases > 1:
    return (saved_model_future, cache_value_nodes)

  # Identify the merge combines that can be packed together.
  inspect_merge_combine_visitor = _InspectMergeCombineVisitor()
  inspect_merge_combine_traverser = nodes.Traverser(
      inspect_merge_combine_visitor)
  _ = inspect_merge_combine_traverser.visit_value_node(saved_model_future)

  # Only pack if we have more than one merge combines.
  if len(inspect_merge_combine_visitor.packable_combine_extract_outputs) <= 1:
    return (saved_model_future, cache_value_nodes)

  # Add flatten and packed merge nodes.
  pack_merge_combine_visitor = _PackMergeCombineVisitor(
      packable_combine_extract_outputs=
      inspect_merge_combine_visitor.packable_combine_extract_outputs)
  pack_merge_combine_traverser = nodes.Traverser(pack_merge_combine_visitor)
  saved_model_future = pack_merge_combine_traverser.visit_value_node(
      saved_model_future)
  # Replace cache nodes to point to the corresponding new nodes.
  cache_value_nodes = _update_cache_value_node_references(
      cache_value_nodes, pack_merge_combine_traverser)

  # Remove redundant flatten and packed merge nodes.
  remove_redundant_visitor = _RemoveRedundantPackedMergeCombineVisitor(
      final_packed_merge_combine_label=
      pack_merge_combine_visitor.final_packed_merge_combine_label)
  remove_redundant_traverser = nodes.Traverser(remove_redundant_visitor)
  saved_model_future = remove_redundant_traverser.visit_value_node(
      saved_model_future)
  # Replace cache nodes to point to the corresponding new nodes.
  cache_value_nodes = _update_cache_value_node_references(
      cache_value_nodes, remove_redundant_traverser)

  return (saved_model_future, cache_value_nodes)
