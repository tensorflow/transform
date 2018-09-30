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
"""Functions to apply pretrained tensorflow models.

These functions allow the use of pretrained models in the preproceesing
function. Users can use a SavedModel specified by model dir, metagraph tags,
signatures, or use a tensor-in-tensor-out function which includes variables
together with a specified checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import six
import tensorflow as tf

from tensorflow.contrib.session_bundle import bundle_shim


def apply_saved_model(model_dir, inputs, tags, signature_name=None,
                      output_keys_in_signature=None):
  """Applies a SavedModel to some `Tensor`s.

  Applies a SavedModel to `inputs`. The SavedModel is specified with
  `model_dir`, `tags` and `signature_name`. Note that the SavedModel will be
  converted to an all-constants graph.

  Args:
    model_dir: A path containing a SavedModel.
    inputs: A dict whose keys are the names from the input signature and whose
        values are `Tensor`s. If there is only one input in the model's input
        signature then `inputs` can be a single `Tensor`.
    tags: The tags specifying which metagraph to load from the SavedModel.
    signature_name: Specify signature of the loaded model. The default value
        None can be used if there is only one signature in the MetaGraphDef.
    output_keys_in_signature: A list of strings which should be a subset of
        the outputs in the signature of the SavedModel. The returned `Tensor`s
        will correspond to specified output `Tensor`s, in the same order. The
        default value None can be used if there is only one output from
        signature.

  Returns:
    A `Tensor` or list of `Tensor`s representing the application of the
        SavedModel.

  Raises:
    ValueError: if
    `inputs` is invalid type, or
    `signature_name` is None but the SavedModel contains multiple signature, or
    `inputs` do not match the signature inputs, or
    `output_keys_in_signature` is not a subset of the signature outputs.
  """
  # Load model, get graph, inputs and outputs.
  loaded_graph = tf.Graph()
  loaded_initializer_op_names = []

  with loaded_graph.as_default():
    session, meta_graph = (
        bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
            model_dir, tags=tags))
    loaded_initializer_op_names = [op.name for op in tf.get_collection(
        tf.GraphKeys.TABLE_INITIALIZERS)]

    if signature_name:
      signature = meta_graph.signature_def[signature_name]
    elif len(meta_graph.signature_def) > 1:
      raise ValueError(
          'The SavedModel contains multiple signatures (%r) but signature_name '
          'was not specified.' % (meta_graph.signature_def.keys(),))
    else:
      signature = next(six.itervalues(meta_graph.signature_def))

  # Generate mapping from tensors in the graph to the input tensors.
  if isinstance(inputs, dict):
    if set(signature.inputs.keys()) != set(inputs.keys()):
      raise ValueError(
          'The keys in `inputs` (%r) do not match inputs of the SavedModel '
          '(%r).' % (inputs.keys(), signature.inputs.keys()))
    input_name_to_tensor_map = {
        signature.inputs[key].name: inputs[key]
        for key in inputs.keys()}
  elif len(signature.inputs) != 1:
    raise ValueError(
        'The SavedModel does not have exactly one input (had inputs %r) but '
        '`inputs` was not a dict.' % (signature.inputs.keys(),))
  else:
    input_name_to_tensor_map = {
        next(six.itervalues(signature.inputs)).name: inputs
    }

  # Get output tensor names.
  if output_keys_in_signature:
    if not set(output_keys_in_signature) <= set(signature.outputs.keys()):
      raise ValueError(
          'output_keys_in_signature (%r) is not a subset of outputs of the '
          'SavedModel (%r).'
          % (output_keys_in_signature, signature.outputs.keys()))

    output_tensor_names = [
        signature.outputs[key].name for key in output_keys_in_signature
    ]
    output_single_tensor = False
  elif len(signature.outputs) != 1:
    raise ValueError(
        'The SavedModel does not have exactly one output (had outputs %r) but '
        'output_keys_in_signature was not specified.'
        % (signature.outputs.keys(),))
  else:
    output_tensor_names = [next(six.itervalues(signature.outputs)).name]
    output_single_tensor = True

  # Convert_variables_to_constants() requires op name.
  output_op_names = [loaded_graph.get_tensor_by_name(tensor_name).op.name
                     for tensor_name in output_tensor_names]
  constant_graph_def = tf.graph_util.convert_variables_to_constants(
      session,
      loaded_graph.as_graph_def(),
      output_op_names + loaded_initializer_op_names)

  returned_elements = tf.import_graph_def(
      constant_graph_def,
      input_map=input_name_to_tensor_map,
      return_elements=output_tensor_names + loaded_initializer_op_names)
  returned_output_tensors = returned_elements[:len(output_tensor_names)]
  returned_initializer_ops = returned_elements[len(output_tensor_names):]

  for initializer_op in returned_initializer_ops:
    tf.add_to_collection(
        tf.GraphKeys.TABLE_INITIALIZERS,
        initializer_op)

  if output_single_tensor:
    assert len(output_tensor_names) == 1
    return returned_output_tensors[0]
  else:
    return returned_output_tensors


def apply_function_with_checkpoint(fn, inputs, checkpoint, include=None,
                                   exclude=None):
  """Applies a tensor-in-tensor-out function with variables to some `Tensor`s.

  Variable values are loaded from the given checkpoint path. Note that the
  input_tensor_func, together with the checkpoint, will be converted to an
  all-constants graph, so ops requiring graph collections, such as table lookup
  (which requires a table init op being added to TABLE_INITIALIZERS collection),
  are not supported.

  Args:
    fn: A tensor-in-tensor-out function that may contain variables.
    inputs: A list of `Tensor`s to apply `fn` to.
    checkpoint: The checkpoint path to load variables from.
    include: An optional list/tuple of scope strings for filtering which
        variables from the VARIABLES collection to include. If None, all
        variables will be included.
    exclude: An optional list/tuple of scope strings for filtering which
        variables from the VARIABLES collection to exclude. If None, no
        variables will be excluded.

  Returns:
    A `Tensor` or list of `Tensor`s representing the application of `fn`.

  Raises:
    ValueError: if the input tensor-in-tensor-out function adds to
        TABLE_INITIALIZERS collections.
  """
  loaded_graph = tf.Graph()
  with loaded_graph.as_default():
    input_placeholders = [
        tf.placeholder(dtype=tensor.dtype, shape=tensor.shape,
                       name=tensor.op.name)
        for tensor in inputs]
    output = fn(*input_placeholders)
    if isinstance(output, tf.Tensor):
      output_tensors = [output]
      output_single_tensor = True
    else:
      output_tensors = output
      output_single_tensor = False

    if tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS):
      raise ValueError('Models with table init ops are not supported.')

    vars_to_restore = tf.contrib.slim.get_variables_to_restore(
        include=include, exclude=exclude)
    saver = tf.train.Saver(vars_to_restore)
    with tf.Session() as sess:
      saver.restore(sess, checkpoint)
      output_graph_def = tf.graph_util.convert_variables_to_constants(
          sess, loaded_graph.as_graph_def(),
          [tensor.op.name for tensor in output_tensors])

  input_map = {tensor.name: tensor for tensor in inputs}
  output_tensors = tf.import_graph_def(
      output_graph_def, input_map=input_map,
      return_elements=[tensor.name for tensor in output_tensors])

  if output_single_tensor:
    assert len(output_tensors) == 1
    return output_tensors[0]
  else:
    return output_tensors
