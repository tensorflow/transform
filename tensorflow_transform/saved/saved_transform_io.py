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
"""Utility functions to build input_fns for use with tf.Learn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

# GOOGLE-INITIALIZATION

import six
import tensorflow as tf
from tensorflow_transform.py_func import pyfunc_helper
from tensorflow_transform.saved import constants
from tensorflow_transform.saved import saved_model_loader
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as tf_saver

_MANGLED_TENSOR_NAME_RE = re.compile(
    r'(.*)\$(indices|values|dense_shape|dense_tensor)$')


def _update_legacy_signature(signature):
  """Update a legacy name-mangled signature in-place.

  Note this code will not work if there are clashes between the old and new
  names, e.g. if x$dense_tensor$dense_tensor and x$dense_tensor are both
  features, but this is an edge case that we do not expect to ever happen.

  Args:
    signature: A SignatureDef.
  """
  for tensor_info_map in [signature.inputs, signature.outputs]:
    # It is necessary to make a copy of tensor_info_map.items() since we need to
    # modify tensor_info_map while iterating it.
    for original_name, original_tensor_info in list(tensor_info_map.items()):
      match = _MANGLED_TENSOR_NAME_RE.match(original_name)
      if not match:
        continue
      tf.logging.warn(
          'Converting feature %s from legacy signature.  New models will '
          'be written without name-mangling in the signature', original_name)
      name = match.group(1)
      if name == 'dense_shape':
        assert name not in tensor_info_map
      else:
        assert (name not in tensor_info_map or
                tensor_info_map[name].WhichOneof('encoding') == 'coo_sparse')
      new_tensor_info = tensor_info_map[name]
      original_tensor_type = match.group(2)
      if original_tensor_type == 'indices':
        new_tensor_info.coo_sparse.indices_tensor_name = (
            original_tensor_info.name)
      elif original_tensor_type == 'values':
        new_tensor_info.dtype = original_tensor_info.dtype
        new_tensor_info.coo_sparse.values_tensor_name = (
            original_tensor_info.name)
      elif original_tensor_type == 'dense_shape':
        new_tensor_info.coo_sparse.dense_shape_tensor_name = (
            original_tensor_info.name)
      else:
        new_tensor_info.CopyFrom(tensor_info_map[original_name])
      del tensor_info_map[original_name]


def _load_transform_saved_model(transform_savedmodel_dir):
  """Load a SavedModel representing a transform function from disk.

  Args:
    transform_savedmodel_dir: a SavedModel directory.

  Returns:
    A tuple with a `MetaGraphDef` proto, the input and outputs of a
    `SignatureDef` proto, and a dict from tensor names to absolute paths for
    asset filepaths.
  """
  saved_model = saved_model_loader.parse_saved_model(
      transform_savedmodel_dir)
  meta_graph_def = saved_model_loader.choose_meta_graph_def(
      saved_model, [constants.TRANSFORM_TAG])

  signature = meta_graph_def.signature_def[constants.TRANSFORM_SIGNATURE]
  # The following code handles models produced prior to CL/200123875.  These
  # models used a non-standard naming convention for features in order to
  # support SparseTensor.
  # TODO(b/34253951): Remove the following code once we no longer want to
  # support the legacy formats.
  _update_legacy_signature(signature)

  # maps name to TensorInfo
  input_signature = signature.inputs
  output_signature = signature.outputs

  # asset_path_dict is {string: string}, mapping tensor names to absolute paths.
  asset_path_dict = saved_model_loader.get_asset_tensors(
      transform_savedmodel_dir, meta_graph_def)

  return meta_graph_def, input_signature, output_signature, asset_path_dict


_PARTITIONED_VARIABLE_NAME_RE = re.compile(r'^(.*)/part_(\d*)$')


def _partially_apply_saved_transform_impl(
    saved_model_dir, logical_input_map, tensor_replacement_map=None,
    fetch_tensor_names=None):
  """Shared code for partially_apply_saved_transform and fetch_tensor_values.

  This adds nodes to a graph that already contains Tensors representing the
  inputs.  These input Tensors may be placeholders that will be fed when the
  graph is executed, or may be the outputs of some Ops.  Most typically, the
  input Tensors are reading and/or parsing Ops, but they could be anything--
  including the outputs of a prior application of this function using another
  transform graph.

  This function operates on the default Graph in the default Session, and so
  must be called within a context where these are provided.

  Args:
    saved_model_dir: A SavedModel directory providing a transform
      graph.  The MetaGraphDef and signature are selected from the SavedModel
      using keys defined in `../constants.py` ('transform' and
      'transform_signature', respectively).
    logical_input_map: a dict of logical name to Tensor.  The logical names must
      be a subset of those in the input signature of the transform graph, and
      the corresponding Tensors must have the expected types and shapes.
    tensor_replacement_map: a dict of tensor names to `Tensors`.
    fetch_tensor_names: a list of tensor names.

  Returns:
    A tuple of (unbound_inputs, outputs, fetched_tensors) where unbound_inputs
    is a dict of logical name to Tensors that are yet to be mapped or fed,
    outputs is a dict of logical name to Tensor, as provided by the output
    signature of the transform graph, and fetched_tensors is a dict of tensor
    names to `Tensor`s where the tensor names are the names given by
    `fetched_tensor_names`.

  Raises:
    ValueError: if the provided input_tensors dict has keys that are not part
      of the input signature, or any of the provided inputs have the wrong
      type or shape.
    RuntimeError: if there is no default graph available to which to apply the
      transform.
  """
  graph = tf.get_default_graph()
  if graph is None:
    raise RuntimeError('apply_saved_transform() requires a default graph.')

  meta_graph_def, input_signature, output_signature, asset_path_dict = (
      _load_transform_saved_model(saved_model_dir))
  asset_tensor_dict = {k: ops.convert_to_tensor(v)
                       for k, v in asset_path_dict.items()}

  # Check for inputs that were not part of the input signature.
  unexpected_inputs = (set(six.iterkeys(logical_input_map)) -
                       set(six.iterkeys(input_signature)))
  if unexpected_inputs:
    raise ValueError('Unexpected inputs '
                     'to transform: {}'.format(unexpected_inputs))

  # Create a map from tensor names in the graph to be imported, to the tensors
  # specified in `input_tensors`.
  input_map = {}
  for logical_name, replacement in six.iteritems(logical_input_map):
    tensor_info = input_signature[logical_name]
    if tensor_info.WhichOneof('encoding') == 'coo_sparse':
      input_map[tensor_info.coo_sparse.indices_tensor_name] = (
          replacement.indices)
      input_map[tensor_info.coo_sparse.values_tensor_name] = replacement.values
      input_map[tensor_info.coo_sparse.dense_shape_tensor_name] = (
          replacement.dense_shape)
    else:
      input_map[tensor_info.name] = replacement

  input_map.update(asset_tensor_dict)
  if tensor_replacement_map:
    input_map.update(tensor_replacement_map)

  # unique_name may produce e.g. transform_5.  The result has no trailing slash.
  scope = graph.unique_name('transform', mark_as_used=False)

  # unique_name returns an "absolute" name while we want a name relative to the
  # current scope.  Therefore, we check if the current name stack is non-empty,
  # and if so, strip out the existing name scope.
  if graph.get_name_scope():
    current_name_scope = graph.get_name_scope() + '/'
    assert scope.startswith(current_name_scope)
    import_scope = scope[len(current_name_scope):]
  else:
    import_scope = scope

  # If the saved_model contained py_funcs, will reinsert them in the graph
  # here and update their associated token in the model.
  pyfunc_helper.register_pyfuncs_from_saved_transform(graph, meta_graph_def)

  # Save the ASSET_FILEPATHS before importing the MetaGraphDef
  current_assets = graph.get_collection(tf.GraphKeys.ASSET_FILEPATHS)

  # Warn user if meta_graph_def has saved variables
  if tf.GraphKeys.TRAINABLE_VARIABLES in meta_graph_def.collection_def:
    trainable_vars = meta_graph_def.collection_def[
        tf.GraphKeys.TRAINABLE_VARIABLES].bytes_list.value
    if trainable_vars:
      raise ValueError(
          'The SavedModel contained trainable variables {}.  Because this '
          'function is typically called in the input_fn, trainable variables '
          'are disallowed'.format(trainable_vars))

  # Load the transform graph, applying it to existing Tensors via input_map.
  # Throws ValueError if the input_map gives mismatched types or shapes.
  saver = tf_saver.import_meta_graph(meta_graph_def,
                                     import_scope=import_scope,
                                     input_map=input_map)

  # Wipe out AssetFileDef collection; it is obsolete after loading
  graph.clear_collection(tf.saved_model.constants.ASSETS_KEY)

  # The import may have added Tensors to the ASSET_FILEPATHS collection that
  # were substituted via input_map.  To account for this, wipe out the
  # collection, restore the preexisting collection values, and then write in
  # the new substituted Tensors.
  graph.clear_collection(tf.GraphKeys.ASSET_FILEPATHS)
  for asset_path_tensor in current_assets:
    graph.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_path_tensor)
  for asset_path_tensor in asset_tensor_dict.values():
    graph.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_path_tensor)

  if saver:
    checkpoint_path = os.path.join(
        tf.compat.as_bytes(saved_model_dir),
        tf.compat.as_bytes(tf.saved_model.constants.VARIABLES_DIRECTORY),
        tf.compat.as_bytes(tf.saved_model.constants.VARIABLES_FILENAME))

    # We can't use the scope rename from init_from_checkpoint because it relies
    # on var scopes not rebuilt by import_meta_graph. So we need to construct it
    # explicitly by iterating over the variables.
    # TODO(b/78624684): remove this workaround.
    var_map = {}
    for var in tf.global_variables():
      var_name = var.op.name
      if not var_name.startswith(scope):
        continue

      # Generate original name before importing into scope.
      original_var_name = var_name[len(scope)+1:]

      match = _PARTITIONED_VARIABLE_NAME_RE.match(original_var_name)
      if match:
        # If the variable is partitioned, extract the base variable name and
        # the index in the partition, then update var_map[base_name] to have
        # var_map[base_name][partition_index] = var.
        base_name = match.group(1)
        partition_index = int(match.group(2))
        if base_name not in var_map:
          var_map[base_name] = []
        while not partition_index < len(var_map[base_name]):
          var_map[base_name].append(None)
        assert var_map[base_name][partition_index] is None
        var_map[base_name][partition_index] = var
      else:
        var_map[original_var_name] = var

    if var_map:
      tf.train.init_from_checkpoint(checkpoint_path, var_map)

  # Add computed output tensors to the output.  There are two cases.  When the
  # output is not in the input_map, then we look up the tensor in the imported
  # graph by prepending the import scope and looking up the tensor by name.
  # This will fail if the expected output tensor is not now in the graph
  # under the expected name scope.  When the output is in the input map, then
  # that tensor will have been re-mapped so we use the tensor given in the
  # input_map.
  def lookup_remapped_tensor(tensor_name):
    if tensor_name in input_map:
      return input_map[tensor_name]
    else:
      return graph.get_tensor_by_name(
          ops.prepend_name_scope(tensor_name, scope))
  def lookup_tensor_or_sparse_tensor(tensor_info):
    if tensor_info.WhichOneof('encoding') == 'coo_sparse':
      return tf.SparseTensor(
          lookup_remapped_tensor(tensor_info.coo_sparse.indices_tensor_name),
          lookup_remapped_tensor(tensor_info.coo_sparse.values_tensor_name),
          lookup_remapped_tensor(
              tensor_info.coo_sparse.dense_shape_tensor_name))
    else:
      return lookup_remapped_tensor(tensor_info.name)
  outputs = {
      logical_name: lookup_tensor_or_sparse_tensor(tensor_info)
      for logical_name, tensor_info in six.iteritems(output_signature)}
  # Do the same for input tensors, although such tensors should never be in the
  # input_map since identical tensors in an input_map would be an error.
  unbound_inputs = {
      logical_name: lookup_tensor_or_sparse_tensor(tensor_info)
      for logical_name, tensor_info in six.iteritems(input_signature)
      if logical_name not in logical_input_map}
  if fetch_tensor_names is None:
    fetch_tensor_names = []
  fetched_tensors = {
      name: lookup_remapped_tensor(name) for name in fetch_tensor_names}

  return unbound_inputs, outputs, fetched_tensors


def partially_apply_saved_transform(saved_model_dir, logical_input_map,
                                    tensor_replacement_map=None):
  """Deprecated alias for partially_apply_saved_transform_internal."""
  tf.logging.warn(
      'partially_apply_saved_transform is deprecated.  Use the '
      'transform_raw_features method of the TFTrandformOutput class instead.')
  return partially_apply_saved_transform_internal(
      saved_model_dir, logical_input_map, tensor_replacement_map)


def partially_apply_saved_transform_internal(saved_model_dir, logical_input_map,
                                             tensor_replacement_map=None):
  """Apply a transform graph, represented as a SavedModel, to existing Tensors.

  For internal use only.  Users should use the transform_raw_features method
  of the TFTrandformOutput class.

  This adds nodes to a graph that already contains Tensors representing the
  inputs.  These input Tensors may be placeholders that will be fed when the
  graph is executed, or may be the outputs of some Ops.  Most typically, the
  input Tensors are reading and/or parsing Ops, but they could be anything--
  including the outputs of a prior application of this function using another
  transform graph.

  This function operates on the default Graph in the default Session, and so
  must be called within a context where these are provided.

  Args:
    saved_model_dir: A SavedModel directory providing a transform
      graph.  The MetaGraphDef and signature are selected from the SavedModel
      using keys defined in `../constants.py` ('transform' and
      'transform_signature', respectively).
    logical_input_map: a dict of logical name to Tensor.  The logical names must
      be a subset of those in the input signature of the transform graph, and
      the corresponding Tensors must have the expected types and shapes.
    tensor_replacement_map: a dict of tensor names to `Tensors`.

  Returns:
    A pair of (unbound_inputs, outputs) where unbound_inputs is a dict of
    logical name to Tensors that are yet to be mapped or fed, and outputs is
    a dict of logical name to Tensor, as provided by the output signature
    of the transform graph

  Raises:
    ValueError: if the provided input_tensors dict has keys that are not part
      of the input signature, or any of the provided inputs have the wrong
      type or shape.
    RuntimeError: if there is no default graph available to which to apply the
      transform.
  """
  unbound_inputs, outputs, _ = _partially_apply_saved_transform_impl(
      saved_model_dir, logical_input_map, tensor_replacement_map)
  return unbound_inputs, outputs


def write_saved_transform_from_session(
    session, inputs, outputs, export_path, as_text=False):
  """Write the current session as a SavedModel."""
  predict_signature_def = (
      tf.saved_model.signature_def_utils.predict_signature_def(inputs, outputs))

  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  builder.add_meta_graph_and_variables(
      session, [constants.TRANSFORM_TAG],
      signature_def_map={'transform_signature': predict_signature_def},
      assets_collection=tf.get_collection(
          tf.GraphKeys.ASSET_FILEPATHS))
  builder.save(as_text)

