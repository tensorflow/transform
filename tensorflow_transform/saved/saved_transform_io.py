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

import re


import six
import tensorflow as tf
from tensorflow_transform.saved import constants
from tensorflow_transform.saved import saved_model_loader
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as tf_saver


def _load_transform_saved_model(transform_savedmodel_dir):
  """Load a SavedModel representing a transform function from disk.

  Args:
    transform_savedmodel_dir: a SavedModel directory.

  Returns:
    A `SavedModel` protocol buffer.
  """
  saved_model = saved_model_loader.parse_saved_model(
      transform_savedmodel_dir)
  meta_graph_def = saved_model_loader.choose_meta_graph_def(
      saved_model, [constants.TRANSFORM_TAG])

  signature = meta_graph_def.signature_def[constants.TRANSFORM_SIGNATURE]

  # maps name to TensorInfo
  input_signature = {logical_name: tensor_info.name
                     for logical_name, tensor_info
                     in six.iteritems(signature.inputs)}
  output_signature = {logical_name: tensor_info.name
                      for logical_name, tensor_info
                      in six.iteritems(signature.outputs)}

  # asset_path_dict is {string: string}, mapping tensor names to absolute paths.
  asset_path_dict = saved_model_loader.get_asset_tensors(
      transform_savedmodel_dir, meta_graph_def)

  return meta_graph_def, input_signature, output_signature, asset_path_dict


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

  decomposed_input_tensors = _decompose_sparse_tensors(logical_input_map)

  meta_graph_def, input_signature, output_signature, asset_path_dict = (
      _load_transform_saved_model(saved_model_dir))
  asset_tensor_dict = {k: ops.convert_to_tensor(v)
                       for k, v in asset_path_dict.items()}

  # Check for inputs that were not part of the input signature.
  unexpected_inputs = (set(six.iterkeys(decomposed_input_tensors)) -
                       set(six.iterkeys(input_signature)))
  if unexpected_inputs:
    raise ValueError('Unexpected inputs '
                     'to transform: {}'.format(unexpected_inputs))

  # Create a map from tensor names in the graph to be imported, to the tensors
  # specified in `input_tensors`.
  input_map = {
      input_signature[decomposed_logical_name]:
      decomposed_input_tensors[decomposed_logical_name]
      for decomposed_logical_name in decomposed_input_tensors}
  input_map.update(asset_tensor_dict)
  if tensor_replacement_map:
    input_map.update(tensor_replacement_map)

  # unique_name may produce e.g. transform_5.  The result has no trailing slash.
  scope = graph.unique_name('transform', mark_as_used=False)


  # Save the ASSET_FILEPATHS before importing the MetaGraphDef
  current_assets = graph.get_collection(ops.GraphKeys.ASSET_FILEPATHS)

  # Load the transform graph, applying it to existing Tensors via input_map.
  # Throws ValueError if the input_map gives mismatched types or shapes.
  saver = tf_saver.import_meta_graph(meta_graph_def,
                                     import_scope=scope,
                                     input_map=input_map)

  # Wipe out AssetFileDef collection; it is obsolete after loading
  graph.clear_collection(tf.saved_model.constants.ASSETS_KEY)

  # The import may have added Tensors to the ASSET_FILEPATHS collection that
  # were substituted via input_map.  To account for this, wipe out the
  # collection, restore the preexisting collection values, and then write in
  # the new substituted Tensors.
  graph.clear_collection(ops.GraphKeys.ASSET_FILEPATHS)
  for asset_path_tensor in current_assets:
    graph.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset_path_tensor)
  for asset_path_tensor in asset_tensor_dict.values():
    graph.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset_path_tensor)

  if saver:
    tf.logging.warn(
        'Transform graphs should not have saved Variables, but this '
        'one does.  Variable values will *not* be restored.')

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
  decomposed_output_tensors = {
      decomposed_logical_name: lookup_remapped_tensor(tensor_name)
      for decomposed_logical_name, tensor_name
      in six.iteritems(output_signature)
  }
  # Do the same for input tensors, where we assume such tensors are not in the
  # input_map since identical tensors in an input_map would be an error.
  decomposed_unbound_input_tensors = {
      decomposed_logical_name: graph.get_tensor_by_name(
          ops.prepend_name_scope(tensor_name, scope))
      for decomposed_logical_name, tensor_name in six.iteritems(input_signature)
      if decomposed_logical_name not in decomposed_input_tensors
  }
  if fetch_tensor_names is None:
    fetch_tensor_names = []
  fetched_tensors = {
      name: lookup_remapped_tensor(name) for name in fetch_tensor_names}

  outputs = _recompose_sparse_tensors(decomposed_output_tensors)
  unbound_inputs = _recompose_sparse_tensors(decomposed_unbound_input_tensors)
  return unbound_inputs, outputs, fetched_tensors


def fetch_tensor_values(saved_model_dir, tensor_replacement_map,
                        fetch_tensor_names):
  _, _, fetched_tensors = _partially_apply_saved_transform_impl(
      saved_model_dir, {}, tensor_replacement_map, fetch_tensor_names)
  return fetched_tensors


def partially_apply_saved_transform(saved_model_dir, logical_input_map,
                                    tensor_replacement_map=None):
  """Apply a transform graph, represented as a SavedModel, to existing Tensors.

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


def apply_saved_transform(saved_model_dir, input_tensors):
  """Apply a transform graph, represented as a SavedModel, to existing Tensors.

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
    input_tensors: a dict of logical name to Tensor.  The logical names must
      match those in the input signature of the transform graph, and the
      corresponding Tensors must have the expected types and shapes.

  Returns:
    A dict of logical name to Tensor, as provided by the output signature
    of the transform graph.

  Raises:
    ValueError: if the provided input_tensors dict does not provide exactly the
      required inputs, or any of the provided inputs have the wrong type or
      shape.
  """
  unbound_inputs, outputs = partially_apply_saved_transform(
      saved_model_dir, input_tensors)
  if unbound_inputs:
    raise ValueError('Missing required inputs '
                     'to transform: {}'.format(unbound_inputs.keys()))
  return outputs


def write_saved_transform_from_session(
    session, inputs, outputs, export_path, as_text=False):
  """Write the current session as a SavedModel."""
  predict_signature_def = _predict_signature_def(
      _decompose_sparse_tensors(inputs),
      _decompose_sparse_tensors(outputs))

  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  builder.add_meta_graph_and_variables(
      session, [constants.TRANSFORM_TAG],
      signature_def_map={'transform_signature': predict_signature_def},
      assets_collection=tf.get_collection(
          tf.GraphKeys.ASSET_FILEPATHS))
  builder.save(as_text)


_SPARSE_TENSOR_NAME_RE = re.compile(r'(.*)\$(indices|values|dense_shape)$')

_DENSE_TENSOR_NAME_RE = re.compile(r'(.*)\$dense_tensor$')


def _decompose_sparse_tensors(tensor_map):
  """Separates out `SparseTensor`s into their constituent parts.

  Takes a map from column names to `Tensor`s or `SparseTensor`s, and
  decomposes each `SparseTensor` into its parts, assigning each part a new
  column name in the returned map.

  Note that there is never any possibility of name collision, as every column
  name gets some suffix such as "$values" added to it.  Therefore every expanded
  name can be uniquely mapped back to the original column name.

  Args:
    tensor_map: A map from strings to `Tensor`s,

  Returns:
    A map from strings to `Tensor`s.
  """
  result = {}

  for key, tensor in six.iteritems(tensor_map):
    if isinstance(tensor, tf.SparseTensor):
      result[key + '$indices'] = tensor.indices
      result[key + '$values'] = tensor.values
      result[key + '$dense_shape'] = tensor.dense_shape
    else:
      result[key + '$dense_tensor'] = tensor

  return result


def _recompose_sparse_tensors(tensor_map):
  """Undoes the function _decompose_sparse_tensors."""
  result = {}

  sparse_keys = set()
  dense_keys = set()
  for key in six.iterkeys(tensor_map):
    match = _SPARSE_TENSOR_NAME_RE.match(key)
    if match:
      sparse_keys.add(match.group(1))
      continue
    match = _DENSE_TENSOR_NAME_RE.match(key)
    if match:
      dense_keys.add(match.group(1))
      continue
    raise ValueError('Unexpected key: {}'.format(key))

  for key in sparse_keys:
    result[key] = tf.SparseTensor(tensor_map[key + '$indices'],
                                  tensor_map[key + '$values'],
                                  tensor_map[key + '$dense_shape'])
  for key in dense_keys:
    result[key] = tensor_map[key + '$dense_tensor']

  return result


# forked from saved_model/signature_def_utils_impl.py to avoid renaming with
# standardized keys, which breaks our decomposed naming standard.
def _predict_signature_def(inputs, outputs):
  """Creates prediction signature from given inputs and outputs.

  Args:
    inputs: dict of string to `Tensor`.
    outputs: dict of string to `Tensor`.

  Returns:
    A prediction-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError('inputs cannot be None or empty for prediction.')
  if outputs is None:
    raise ValueError('outputs cannot be None or empty for prediction.')

  signature_inputs = {
      key: tf.saved_model.utils.build_tensor_info(tensor)
      for key, tensor in six.iteritems(inputs)
  }
  signature_outputs = {
      key: tf.saved_model.utils.build_tensor_info(tensor)
      for key, tensor in six.iteritems(outputs)
  }

  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
