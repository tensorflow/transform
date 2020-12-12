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
from tensorflow.python.saved_model import load
from tensorflow.python.util import object_identity
# pylint: enable=g-direct-tensorflow-import


class _Loader(load.Loader):

  def _recreate_asset(self, *args, **kwargs):
    result = super()._recreate_asset(*args, **kwargs)
    if not tf.executing_eagerly():
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS,
                                     result[0].asset_path)
    return result


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
    if tf.version.VERSION < '2.5':
      self._imported = load.load_internal(saved_model_dir, loader_cls=_Loader)
      if isinstance(self._imported, dict):
        self._imported = self._imported['root']
    else:
      # TODO(b/160294509): Stop using tf.compat.v2 when TF1.15 support is
      # dropped.
      self._imported = tf.compat.v2.saved_model.load(saved_model_dir)
    self.load_v2_in_compat = (
        constants.TRANSFORM_SIGNATURE in self._imported.signatures)
    if self.load_v2_in_compat:
      self._wrapped = self._imported.signatures[constants.TRANSFORM_SIGNATURE]
      self._func_graph = self._wrapped.graph
      self._structured_inputs = self._get_input_signature_from_v1_saved_model(
          saved_model_dir)
      self._structured_outputs = self._wrapped.structured_outputs
    else:
      # TODO(b/160550490): Remove local import.
      from tensorflow_transform import tf2_utils  # pylint: disable=g-import-not-at-top

      # Since `input_signature` was specified when exporting the tf function to
      # transform_fn is now a ConcreteFunction, but was a tf.function. We need
      # to handle both to maintain backward compatiblity. If it's a tf.function,
      # since `input_signature` was specified when exporting the tf function to
      # `SavedModel`, there should be exactly one concrete function present on
      # loading the `SavedModel`.
      if hasattr(self._imported.transform_fn, 'concrete_functions'):
        concrete_functions = self._imported.transform_fn.concrete_functions
        assert len(concrete_functions) == 1, concrete_functions
        self._wrapped = concrete_functions[0]
      else:
        self._wrapped = self._imported.transform_fn
      self._func_graph = self._wrapped.graph
      self._structured_inputs = (
          tf2_utils.get_structured_inputs_from_func_graph(self._func_graph))
      self._structured_outputs = tf.nest.pack_sequence_as(
          self._func_graph.structured_outputs,
          self._func_graph.outputs,
          expand_composites=True)
    self._output_to_inputs_map = (
        self._get_output_to_inputs_map(self._structured_outputs))
    saved_transform_io._maybe_register_addon_ops()  # pylint: disable=protected-access

  def _get_input_signature_from_v1_saved_model(self, saved_model_dir):
    """Get structured inputs for a TF1 compat SavedModel."""
    saved_model = saved_model_loader.parse_saved_model(saved_model_dir)
    meta_graph_def = saved_model_loader.choose_meta_graph_def_and_raise(
        saved_model)
    signature = meta_graph_def.signature_def[constants.TRANSFORM_SIGNATURE]
    return signature.inputs

  def _get_output_to_inputs_map(self, output_signature):
    """Get all graph inputs that the tensors in output_signature depend on."""
    # TODO(b/160550490): Remove local import.
    from tensorflow_transform import graph_tools  # pylint: disable=g-import-not-at-top

    result = {}
    for name, output in six.iteritems(output_signature):
      components = self._get_component_tensors(output)
      sinks = [self._as_operation(component) for component in components]
      # Ignore control dependencies when walking the graph as we only care about
      # which user defined inputs this output depends on.
      result[name] = graph_tools.retrieve_sources(
          sinks, ignore_control_dependencies=True)
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

    if self.load_v2_in_compat:
      return self._apply_v1_transform_model_in_v2(logical_input_map)
    else:
      return self._apply_v2_transform_model(logical_input_map)

  def get_dependent_input_output_keys(self, input_keys, exclude_output_keys):
    """Determine inputs needed to get outputs excluding exclude_output_keys.

    Args:
      input_keys: A collection of all input keys available to supply to the
        SavedModel.
      exclude_output_keys: A collection of output keys returned by the
        SavedModel that should be excluded.

    Returns:
      A pair of:
        required_input_keys: A subset of the input features to this SavedModel
          that are required to compute the set of output features excluding
          `exclude_output_keys`. It is sorted to be deterministic.
        output_keys: The set of output features excluding `exclude_output_keys`.
          It is sorted to be deterministic.

    """
    # Assert inputs being fed and outputs being excluded are part of the
    # SavedModel.
    if set(input_keys).difference(self._structured_inputs.keys()):
      raise ValueError(
          'Input tensor names contained tensors not in graph: {}'.format(
              input_keys))

    if set(exclude_output_keys).difference(self._structured_outputs.keys()):
      raise ValueError(
          'Excluded outputs contained keys not in graph: {}'.format(
              exclude_output_keys))

    output_keys = (
        set(self._structured_outputs.keys()).difference(exclude_output_keys))

    # Get all the input tensors that are required to evaluate output_keys.
    required_inputs = object_identity.ObjectIdentitySet()
    for key in output_keys:
      required_inputs.update(self._output_to_inputs_map[key])

    # Get all the input feature names that have atleast one component tensor in
    # required_inputs.
    required_input_keys = []
    for key, tensor in six.iteritems(self._structured_inputs):
      if any(x in required_inputs for x in self._get_component_tensors(tensor)):
        required_input_keys.append(key)

    return sorted(required_input_keys), sorted(output_keys)
