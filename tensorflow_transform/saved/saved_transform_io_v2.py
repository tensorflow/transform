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
"""Utility functions to save and load from SavedModels in TF 2.x."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# GOOGLE-INITIALIZATION

import six
import tensorflow as tf
from tensorflow_transform.saved import constants
from tensorflow_transform.saved import saved_model_loader
from tensorflow_transform.saved import saved_transform_io
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import object_identity
# pylint: enable=g-direct-tensorflow-import


class SavedModelLoader(object):
  """Handles a SavedModel exported using TF 1.x APIs in TF 2.x."""

  def __init__(self, saved_model_dir):
    """Init method for SavedModelLoader.

    Args:
      saved_model_dir: A SavedModel directory providing a transform graph.  The
        MetaGraphDef and signature are selected from the SavedModel using keys
        defined in `../constants.py` ('transform' and 'transform_signature',
        respectively).
    """
    self._imported = tf.compat.v2.saved_model.load(saved_model_dir)
    self._load_v2_in_compat = (
        constants.TRANSFORM_SIGNATURE in self._imported.signatures)
    if self._load_v2_in_compat:
      self._wrapped = self._imported.signatures[constants.TRANSFORM_SIGNATURE]
      self._func_graph = self._wrapped.graph
      self._structured_inputs = self._get_input_signature_from_v1_saved_model(
          saved_model_dir)
      structured_outputs = self._wrapped.structured_outputs
    else:
      self._wrapped = self._imported.transform_fn
      self._func_graph = self._get_func_graph_from_v2_saved_model(
          self._wrapped.get_concrete_function().graph)
      self._structured_inputs = self._get_structured_inputs_from_func_graph(
          self._func_graph)
      structured_outputs = tf.nest.pack_sequence_as(
          self._func_graph.structured_outputs,
          self._func_graph.outputs,
          expand_composites=True)
    self._output_to_inputs_map = (
        self._get_output_to_inputs_map(structured_outputs))
    saved_transform_io._maybe_register_addon_ops()  # pylint: disable=protected-access

  def _get_input_signature_from_v1_saved_model(self, saved_model_dir):
    """Get structured inputs for a TF1 compat SavedModel."""
    saved_model = saved_model_loader.parse_saved_model(saved_model_dir)
    meta_graph_def = saved_model_loader.choose_meta_graph_def(
        saved_model, [constants.TRANSFORM_TAG])
    signature = meta_graph_def.signature_def[constants.TRANSFORM_SIGNATURE]
    return signature.inputs

  def _get_func_graph_from_v2_saved_model(self, outer_graph):
    """Retrieves nested func graph from `outer_graph`."""
    # TODO(b/160550490): Remove local import.
    from tensorflow_transform import graph_tools  # pylint: disable=g-import-not-at-top

    # `outer_graph` represents the func graph of a TF function imported from a
    # SavedModel. `outer_graph`'s graph_def contains a single
    # StatefulPartitionedCall/PartitionedCall representing the TF function
    # before export to a SavedModel. We extract the func graph of this attribute
    # as this is the graph we want to walk.
    functions = []
    for node in outer_graph.as_graph_def().node:
      if node.op not in ('PartitionedCall', 'StatefulPartitionedCall'):
        continue
      for key in node.attr:
        if node.attr[key].func.name:
          functions.append(node.attr[key].func.name)
    # At the outermost level, there should only be one function.
    assert len(functions) == 1
    return graph_tools.get_func_graph_for_name(outer_graph, functions[0])

  def _get_structured_inputs_from_func_graph(self, func_graph):
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
    # `func_graph.inputs` contains placeholders that represent regular inputs
    # followed by captured inputs. We are only interested in the regular inputs.
    num_captures = len(func_graph.internal_captures)
    graph_inputs = copy.copy(func_graph.inputs)
    if num_captures > 0:
      graph_inputs = graph_inputs[:-num_captures]
    return tf.nest.pack_sequence_as(
        input_signature, graph_inputs, expand_composites=True)

  def _get_output_to_inputs_map(self, output_signature):
    # TODO(b/160550490): Remove local import.
    from tensorflow_transform import graph_tools  # pylint: disable=g-import-not-at-top

    result = {}
    for name, output in six.iteritems(output_signature):
      components = self._get_component_tensors(output)
      sinks = [self._as_operation(component) for component in components]
      result[name] = graph_tools.retrieve_sources(sinks)
    return result

  def _as_operation(self, op_or_tensor):
    if isinstance(op_or_tensor, tf.Tensor):
      return op_or_tensor.op
    return op_or_tensor

  def _get_component_tensors(self, tensor):
    """Get all component tensors.

    Args:
      tensor: A `Tensor` or `CompositeTensor`.

    Returns:
      All `Tensor` components of `tensor`.

    Raises:
      ValueError if supplied `tensor` parameter is neither a `Tensor` nor a
      `CompositeTensor`.
    """
    if isinstance(tensor, tf.Tensor):
      return [tensor]
    elif isinstance(tensor, composite_tensor.CompositeTensor):
      return tf.nest.flatten(tensor, expand_composites=True)
    else:
      raise ValueError(
          'Unsupported tensor. Arg `tensor` is neither a `Tensor` nor a '
          '`CompositeTensor`: {}.'.format(tensor))

  def _get_fetches(self, feeds):
    result = {}
    for name, output in six.iteritems(self._func_graph.structured_outputs):
      extra_sources = self._output_to_inputs_map[name].difference(feeds)
      # If output does not depend on an input placeholder that is not being fed,
      # add it to fetches.
      if not extra_sources.difference(self._func_graph.internal_captures):
        result[name] = output
    return result

  def _apply_v1_transform_model_in_v2(self, logical_input_map):
    """Applies a V1 transform graph to `Tensor`s.

    This method applies the transformation graph as a pruned function to the
    `logical_input_map`.
    It prunes the function loaded from the SavedModel to return only outputs
    that can be computed from the keys provided in `logical_input_map`.

    Args:
      logical_input_map: a dict of logical name to Tensor.  The logical names
        must be a subset of those in the input signature of the transform graph,
        and the corresponding Tensors must have the expected types and shapes.

    Returns:
      A dict of logical name to Tensor, as provided by the output signature of
      the transform graph.
    """
    input_map = (
        saved_transform_io._expand_input_map(  # pylint: disable=protected-access
            logical_input_map, self._structured_inputs))

    feeds = []
    pruned_input_args = []
    for name in six.iterkeys(input_map):
      tensor = self._func_graph.get_tensor_by_name(name)
      try:
        tensor.shape.assert_is_compatible_with(input_map[name].shape)
      except ValueError as e:
        raise ValueError('{}: {}'.format(name, e))
      feeds.append(tensor)
      pruned_input_args.append(input_map[name])

    fetches = self._get_fetches(feeds)
    pruned = self._wrapped.prune(feeds, fetches)
    result = pruned(*pruned_input_args)
    # TODO(b/163329414): Remove set_shape when calling pruned no longer produces
    # tensors with unknown shapes.
    for name, output in fetches.items():
      if hasattr(result[name], 'set_shape'):
        result[name].set_shape(output.shape)
    return result

  def _apply_v2_transform_model(self, logical_input_map):
    """Applies a V2 transform graph to `Tensor`s.

    This method applies the transformation graph to the `logical_input_map` to
    return only outputs that can be computed from the keys provided in
    `logical_input_map`.

    Args:
      logical_input_map: a dict of logical name to Tensor.  The logical names
        must be a subset of those in the input signature of the transform graph,
        and the corresponding Tensors must have the expected types and shapes.

    Returns:
      A dict of logical name to Tensor, as provided by the output signature of
      the transform graph.
    """
    # TODO(b/160550490): Remove local import.
    from tensorflow_transform import tf2_utils  # pylint: disable=g-import-not-at-top

    feeds = object_identity.ObjectIdentitySet(self._func_graph.inputs)
    unfed_input_keys = (
        set(six.iterkeys(self._structured_inputs)) -
        set(six.iterkeys(logical_input_map)))
    for input_key in unfed_input_keys:
      unfed_input_components = self._get_component_tensors(
          self._structured_inputs[input_key])
      feeds = feeds.difference(unfed_input_components)

    modified_inputs = copy.copy(logical_input_map)
    if unfed_input_keys:
      batch_size = 1
      if logical_input_map:
        an_input = next(six.itervalues(logical_input_map))
        if tf.shape(an_input)[0] is not None:
          batch_size = tf.shape(an_input)[0]
      missing_inputs = (
          tf2_utils.supply_missing_inputs(self._structured_inputs, batch_size,
                                          unfed_input_keys))
      modified_inputs.update(missing_inputs)

    fetches = self._get_fetches(feeds)
    transformed_features = self._wrapped(modified_inputs)
    return {key: transformed_features[key] for key in fetches.keys()}

  def apply_transform_model(self, logical_input_map):
    """Applies a transform graph to `Tensor`s.

    Args:
      logical_input_map: a dict of logical name to Tensor.  The logical names
        must be a subset of those in the input signature of the transform graph,
        and the corresponding Tensors must have the expected types and shapes.

    Returns:
      A dict of logical name to Tensor, as provided by the output signature of
      the transform graph.
    """
    unexpected_inputs = (
        set(six.iterkeys(logical_input_map)) -
        set(six.iterkeys(self._structured_inputs)))
    if unexpected_inputs:
      raise ValueError(
          'Unexpected inputs to transform: {}'.format(unexpected_inputs))

    if self._load_v2_in_compat:
      return self._apply_v1_transform_model_in_v2(logical_input_map)
    else:
      return self._apply_v2_transform_model(logical_input_map)
