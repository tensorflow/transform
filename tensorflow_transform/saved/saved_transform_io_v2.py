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

from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import tensorflow as tf
from tensorflow_transform import annotators
from tensorflow_transform import common_types
from tensorflow_transform import graph_tools
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.py_func import pyfunc_helper
from tensorflow_transform.saved import constants
from tensorflow_transform.saved import saved_model_loader
from tensorflow_transform.saved import saved_transform_io
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import object_identity
# pylint: enable=g-direct-tensorflow-import


def _restore_from_v1_saved_model(
    restored_function: function.ConcreteFunction, saved_model_dir: str
) -> Tuple[function.ConcreteFunction, Mapping[str, Any], Mapping[
    str, common_types.TensorType]]:
  """Restores an exported TF1 compat SavedModel."""
  saved_model = saved_model_loader.parse_saved_model(saved_model_dir)
  meta_graph_def = saved_model_loader.choose_meta_graph_def_and_raise(
      saved_model)
  signature = meta_graph_def.signature_def[constants.TRANSFORM_SIGNATURE]
  # Re-register pyfuncs, if any.
  graph_def = pyfunc_helper.register_pyfuncs_from_saved_transform(
      restored_function.graph, meta_graph_def, loaded_in_tf2=True)
  if graph_def is None:
    return (restored_function, signature.inputs,
            restored_function.structured_outputs)

  inputs = [t.name for t in restored_function.graph.inputs]
  outputs = [t.name for t in restored_function.graph.outputs]
  wrapped = wrap_function.function_from_graph_def(graph_def, inputs, outputs)
  structured_outputs = (
      tf.nest.pack_sequence_as(
          restored_function.structured_outputs,
          wrapped.outputs,
          expand_composites=True))
  wrapped = wrapped.prune(wrapped.inputs, structured_outputs)
  return (wrapped, signature.inputs, wrapped.structured_outputs)


def _as_operation(op_or_tensor: Union[tf.Operation, tf.Tensor]) -> tf.Operation:
  if isinstance(op_or_tensor, tf.Tensor):
    return op_or_tensor.op
  return op_or_tensor


def _get_component_tensors(
    tensor: Union[tf.Tensor, composite_tensor.CompositeTensor]
) -> Iterable[tf.Tensor]:
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
        f'`CompositeTensor`: {tensor}.')


def _get_output_to_inputs_map(
    output_signature: Mapping[str, common_types.TensorType]
) -> Dict[str, Iterable[tf.Tensor]]:
  """Gets all graph inputs that the tensors in output_signature depend on."""
  result = {}
  for name, output in output_signature.items():
    components = _get_component_tensors(output)
    sinks = [_as_operation(component) for component in components]
    # Ignore control dependencies when walking the graph as we only care about
    # which user defined inputs this output depends on.
    result[name] = graph_tools.retrieve_sources(
        sinks, ignore_control_dependencies=True)
  return result


class SavedModelLoader:
  """Handles a SavedModel exported using TF 1.x APIs in TF 2.x."""

  def __init__(self, saved_model_dir: str):
    """Init method for SavedModelLoader.

    Args:
      saved_model_dir: A SavedModel directory providing a transform graph.  The
        MetaGraphDef and signature are selected from the SavedModel using keys
        defined in `../constants.py` ('transform' and 'transform_signature',
        respectively).
    """
    # TODO(b/160294509): Stop using tf.compat.v2 when TF1.15 support is
    # dropped.
    imported = tf.compat.v2.saved_model.load(saved_model_dir)
    load_v2_in_compat = constants.TRANSFORM_SIGNATURE in imported.signatures
    if load_v2_in_compat:
      restored_function = imported.signatures[constants.TRANSFORM_SIGNATURE]
      wrapped, structured_inputs, structured_outputs = (
          _restore_from_v1_saved_model(restored_function, saved_model_dir))
    else:
      # transform_fn is now a ConcreteFunction, but was a tf.function. We need
      # to handle both to maintain backward compatiblity. If it's a tf.function,
      # since `input_signature` was specified when exporting the tf function to
      # `SavedModel`, there should be exactly one concrete function present on
      # loading the `SavedModel`.
      if hasattr(imported.transform_fn, 'concrete_functions'):
        concrete_functions = imported.transform_fn.concrete_functions
        assert len(concrete_functions) == 1, concrete_functions
        wrapped = concrete_functions[0]
      else:
        wrapped = imported.transform_fn
      func_graph = wrapped.graph
      structured_inputs = (
          tf2_utils.get_structured_inputs_from_func_graph(func_graph))
      structured_outputs = tf.nest.pack_sequence_as(
          func_graph.structured_outputs,
          func_graph.outputs,
          expand_composites=True)
    outputs_to_inputs_map = _get_output_to_inputs_map(structured_outputs)
    self._initialize(load_v2_in_compat, imported, wrapped, structured_inputs,
                     structured_outputs, outputs_to_inputs_map)
    saved_transform_io._maybe_register_addon_ops()  # pylint: disable=protected-access

  def _initialize(self, load_v2_in_compat, imported, wrapped, structured_inputs,
                  structured_outputs, outputs_to_inputs_map):
    """Initializes all class arguments."""
    self._load_v2_in_compat = load_v2_in_compat
    self._imported = imported
    self._wrapped_function = wrapped
    self._func_graph = self._wrapped_function.graph
    self._structured_inputs = structured_inputs
    self._structured_outputs = structured_outputs
    self._output_to_inputs_map = outputs_to_inputs_map
    self._sorted_unfed_input_keys = None
    self._wrapped_function_finalized = None
    self._is_finalized = False

  @property
  def load_v2_in_compat(self):
    return self._load_v2_in_compat

  @property
  def structured_outputs(self):
    return self._structured_outputs

  def _get_feeds(self, unfed_input_keys: Iterable[str]) -> Iterable[tf.Tensor]:
    """Returns set of tensors that will be fed."""
    result = object_identity.ObjectIdentitySet(self._func_graph.inputs)
    for input_key in unfed_input_keys:
      unfed_input_components = _get_component_tensors(
          self._structured_inputs[input_key])
      result = result.difference(unfed_input_components)
    return result

  def _get_unfed_input_keys(self,
                            input_tensor_keys: Iterable[str]) -> Iterable[str]:
    return set(self._structured_inputs.keys()).difference(input_tensor_keys)

  def _get_fetches(
      self, feeds: Iterable[tf.Tensor]) -> Dict[str, common_types.TensorType]:
    """Returns set of tensors that can be fetched when `feeds` is supplied."""
    result = {}
    for name, output in self._structured_outputs.items():
      extra_sources = self._output_to_inputs_map[name].difference(feeds)
      # If output does not depend on an input placeholder that is not being fed,
      # add it to fetches.
      if not extra_sources.difference(self._func_graph.internal_captures):
        result[name] = output
    return result

  def _get_fetches_keys(self, feeds: Iterable[tf.Tensor]) -> Iterable[str]:
    return self._get_fetches(feeds).keys()

  def _get_missing_inputs(
      self, unfed_input_keys: Iterable[str],
      batch_size: int) -> Dict[str, common_types.TensorType]:
    """Supplies inputs for `unfed_input_keys`."""
    result = {}
    if unfed_input_keys:
      result = (
          tf2_utils.supply_missing_inputs(self._structured_inputs, batch_size,
                                          unfed_input_keys))
    return result

  def _apply_v1_transform_model_in_v2(
      self, logical_input_map: Mapping[str, common_types.TensorType]
  ) -> Dict[str, common_types.TensorType]:
    """Applies a V1 transform graph to dictionary of (Composite)Tensors.

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
    for name in input_map:
      tensor = self._func_graph.get_tensor_by_name(name)
      try:
        tensor.shape.assert_is_compatible_with(input_map[name].shape)
      except ValueError as e:
        raise ValueError('{}: {}'.format(name, e))
      feeds.append(tensor)
      pruned_input_args.append(input_map[name])

    fetches = self._get_fetches(feeds)
    pruned = self._wrapped_function.prune(feeds, fetches)
    result = pruned(*pruned_input_args)
    # TODO(b/163329414): Remove set_shape when calling pruned no longer produces
    # tensors with unknown shapes.
    for name, output in fetches.items():
      if hasattr(result[name], 'set_shape'):
        result[name].set_shape(output.shape)
    return result

  def _format_input_map_as_tensors(self, input_map):
    """Returns a map from string to `tf.Tensor` or `CompositeTensor`."""
    result = {}
    for key, value in input_map.items():
      if isinstance(value, (tf.Tensor, composite_tensor.CompositeTensor)):
        result[key] = value
      else:
        result[key] = tf.convert_to_tensor(value)
    return result

  def _apply_v2_transform_model_finalized(
      self, logical_input_map: Mapping[str, common_types.TensorType]
  ) -> Dict[str, common_types.TensorType]:
    """Applies a V2 transform graph to dictionary of (Composite)Tensors.

    This method applies the transformation graph to the `logical_input_map` to
    return only outputs that can be computed from the keys provided in
    `logical_input_map`. It assumes that self.finalize has been called before
    this method is invoked.

    Args:
      logical_input_map: a dict of logical name to Tensor.  The logical names
        must be a subset of those in the input signature of the transform graph,
        and the corresponding Tensors must have the expected types and shapes.

    Returns:
      A dict of logical name to Tensor, as provided by the output signature of
      the transform graph.
    """

    # Assert that the same keys are fed as this model was finalized with.
    unfed_input_keys = self._get_unfed_input_keys(logical_input_map.keys())
    assert sorted(unfed_input_keys) == self._sorted_unfed_input_keys

    modified_inputs = self._format_input_map_as_tensors(logical_input_map)
    return self._wrapped_function_finalized(modified_inputs)

  def _apply_v2_transform_model(
      self, logical_input_map: Mapping[str, common_types.TensorType]
  ) -> Dict[str, common_types.TensorType]:
    """Applies a V2 transform graph to dictionary of (Composite)Tensors.

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

    unfed_input_keys = self._get_unfed_input_keys(logical_input_map.keys())
    feeds = self._get_feeds(unfed_input_keys)
    modified_inputs = self._format_input_map_as_tensors(logical_input_map)

    if unfed_input_keys:
      batch_size = 1
      if logical_input_map:
        an_input = next(iter(logical_input_map.values()))
        if isinstance(an_input, tf.RaggedTensor):
          batch_size = an_input.bounding_shape(axis=0)
        elif tf.shape(an_input)[0] is not None:
          batch_size = tf.shape(an_input)[0]

      missing_inputs = self._get_missing_inputs(unfed_input_keys, batch_size)
      modified_inputs.update(missing_inputs)

    flattened_inputs = tf.nest.flatten(modified_inputs, expand_composites=True)

    # self._wrapped_function.inputs may be longer than flattened_inputs as it
    # also contains captured inputs. However, we only want the user inputs here
    # so we don't assert equal length.
    for input_t, wrapped_input in zip(flattened_inputs,
                                      self._wrapped_function.inputs):
      try:
        wrapped_input.shape.assert_is_compatible_with(input_t.shape)
      except ValueError as e:
        raise ValueError('{}: {}'.format(input_t, e))

    transformed_features = self._wrapped_function(*flattened_inputs)
    fetches_keys = self._get_fetches_keys(feeds)
    return {key: transformed_features[key] for key in fetches_keys}

  def apply_transform_model(
      self, logical_input_map: Mapping[str, common_types.TensorType]
  ) -> Dict[str, common_types.TensorType]:
    """Applies a transform graph to dictionary of (Composite)Tensors.

    Args:
      logical_input_map: a dict of logical name to Tensor.  The logical names
        must be a subset of those in the input signature of the transform graph,
        and the corresponding Tensors must have the expected types and shapes.

    Returns:
      A dict of logical name to Tensor, as provided by the output signature of
      the transform graph.
    """
    unexpected_inputs = (
        set(logical_input_map.keys()) - set(self._structured_inputs.keys()))
    if unexpected_inputs:
      raise ValueError(
          'Unexpected inputs to transform: {}'.format(unexpected_inputs))

    if self.load_v2_in_compat:
      return self._apply_v1_transform_model_in_v2(logical_input_map)
    elif self._is_finalized:
      return self._apply_v2_transform_model_finalized(logical_input_map)
    else:
      return self._apply_v2_transform_model(logical_input_map)

  def _finalize_wrapped_function(
      self, unfed_input_keys: Iterable[str],
      fetches_keys: Iterable[str]) -> function.ConcreteFunction:
    """Constructs a function that can be invoked without `unfed_input_keys`."""
    original_input_signature = (
        self._wrapped_function.structured_input_signature[0][0])
    input_signature = {
        k: v
        for k, v in original_input_signature.items()
        if k not in unfed_input_keys
    }

    @tf.function(input_signature=[input_signature], autograph=False)
    def wrapped_finalized(inputs):
      missing_inputs = self._get_missing_inputs(unfed_input_keys, batch_size=1)
      # Directly modifying inputs is not allowed in a tf.function. Hence, we
      # make a deep copy here.
      inputs_copy = tf_utils.copy_tensors(inputs)
      inputs_copy.update(missing_inputs)
      flattened_inputs = tf.nest.flatten(inputs_copy, expand_composites=True)
      transformed_features = self._wrapped_function(*flattened_inputs)
      return {key: transformed_features[key] for key in fetches_keys}

    return wrapped_finalized.get_concrete_function()

  # TODO(b/177672051): Consider calling finalize in the TransformFeaturesLayer.
  def finalize(self, input_tensor_keys: Iterable[str],
               output_tensor_keys: Iterable[str]):
    """Finalizes the set of inputs with which this SavedModel will be called.

    Note: This is not Thread-safe. Should be called prior to any calls to
    `apply_transform_model`.

    Args:
      input_tensor_keys: Set of input keys with which the SavedModel will be
        called.
      output_tensor_keys: Set of output keys that should be returned by the
        SavedModel.
    """
    self._sorted_unfed_input_keys = sorted(
        self._get_unfed_input_keys(input_tensor_keys))
    feeds = self._get_feeds(self._sorted_unfed_input_keys)
    unexpected_outputs = (
        set(output_tensor_keys) - set(self._get_fetches_keys(feeds)))
    if unexpected_outputs:
      raise ValueError(
          'Unexpected output keys requested: {}'.format(unexpected_outputs))
    self._wrapped_function_finalized = self._finalize_wrapped_function(
        self._sorted_unfed_input_keys, sorted(output_tensor_keys))
    self._is_finalized = True


# TODO(b/177606209): Remove once TF supports saving optimized functions.
# TODO(b/169666856): WrappedFunction.prune does not support composite tensors.
# Hence, add additional handling when supporting composite tensors in TFT.
def optimize_concrete_function(
    concrete_function: function.ConcreteFunction,
    strip_control_dependencies: bool) -> wrap_function.WrappedFunction:
  """Returns optimized function with same signature as `concrete_function`."""
  wrapped_fn = wrap_function.WrappedFunction(
      concrete_function.graph,
      variable_holder=wrap_function.VariableHolder(share_variables=True))
  fetches = concrete_function.structured_outputs
  if strip_control_dependencies:
    flat_outputs, _ = tf2_utils.strip_and_get_tensors_and_control_dependencies(
        tf.nest.flatten(fetches, expand_composites=True))
    fetches = tf.nest.pack_sequence_as(
        concrete_function.structured_outputs,
        flat_outputs,
        expand_composites=True)
  result = wrapped_fn.prune(
      feeds=concrete_function.inputs,
      fetches=fetches,
      input_signature=concrete_function.structured_input_signature)
  # TODO(b/163329414): Remove once `prune` retains shape information for all
  # components.
  for original_out, pruned_out in zip(concrete_function.outputs,
                                      result.outputs):
    pruned_out.set_shape(original_out.get_shape())
  return result


def trace_and_update_module(
    module: tf.Module, tf_function: function.Function, name: str,
    strip_control_dependencies: bool) -> function.ConcreteFunction:
  """Traces `tf_function` and saves under attr `name` of `module`.

  Args:
    module: A saveable module which will contain the traced `tf_function` under
      attr `name`.
    tf_function: A tf.function to trace.
    name: A name to same the traced `tf_function` to.
    strip_control_dependencies: Boolean. If True, automatic control dependencies
      will be stripped from the outputs of `tf_function`. This should almost
      always be False. It is useful only if you want to use the structure of the
      TF graph to perform any graph manipulations.

  Returns:
    The concrete function obtained from tracing `tf_function`.
  """
  resource_tracker = tracking.ResourceTracker()
  object_tracker = annotators.ObjectTracker()
  created_variables = []

  def _variable_creator(next_creator, **kwargs):
    var = next_creator(**kwargs)
    created_variables.append(var)
    return var

  # Trace `tf_function` to gather any resources in it using the
  # resource_tracker. These are then assigned to `module.resources` and tracked
  # before exporting to SavedModel.
  with tracking.resource_tracker_scope(resource_tracker), \
       annotators.object_tracker_scope(object_tracker), \
       tf.variable_creator_scope(_variable_creator):
    concrete_fn = tf_function.get_concrete_function()

  # Prior to 2020/10/08, saving a tf.function with a concrete function signature
  # would ensure that the function was not re-traced in a round-trip to a
  # SavedModel. Since this is no longer the case, we save the concrete function
  # directly.
  if tf.compat.forward_compatible(2020, 10, 8):
    pruned_function = optimize_concrete_function(concrete_fn,
                                                 strip_control_dependencies)
    module.pruned_variables = pruned_function.variables
    setattr(module, name, pruned_function)
  else:
    setattr(module, name, tf_function)

  # Any variables created need to be explicitly tracked.
  module.created_variables = created_variables
  # Resources need to be explicitly tracked.
  module.resources = resource_tracker.resources
  module.trackable_objects = object_tracker.trackable_objects
  # TODO(b/158011374) - Stop explicitly tracking initializers. Tracking the
  # table should be sufficient.
  initializers = []
  for resource in module.resources:
    if isinstance(resource, lookup_ops.InitializableLookupTableBase):
      initializers.append(resource._initializer)  # pylint: disable=protected-access
  module.initializers = initializers
  module.assets = [
      common_types.Asset(asset_filepath) for asset_filepath in
      concrete_fn.graph.get_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS)
  ]
  return concrete_fn


def write_v2_saved_model(module: tf.Module, tf_function: function.Function,
                         name: str,
                         saved_model_dir: str) -> function.ConcreteFunction:
  """Writes `tf_function` under attr `name` of `module` to `saved_model_dir`."""
  concrete_fn = trace_and_update_module(
      module, tf_function, name, strip_control_dependencies=False)
  tf.saved_model.save(module, saved_model_dir)
  return concrete_fn
