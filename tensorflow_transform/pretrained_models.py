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

import tensorflow as tf


# TODO(b/141936246) Replace this function with a V2-safe way to load models.
def _get_variables(scope=None,
                   suffix=None,
                   collection=tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
  """Gets the list of variables, filtered by scope and/or suffix.

  Taken from tensorflow/contrib/framework/python/ops/variables.py.

  Args:
    scope: an optional scope for filtering the variables to return. Can be a
      variable scope or a string.
    suffix: an optional suffix for filtering the variables to return.
    collection: in which collection search for. Defaults to
      `GraphKeys.GLOBAL_VARIABLES`.

  Returns:
    a list of variables in collection with scope and suffix.
  """
  if scope is not None and isinstance(scope, tf.compat.v1.VariableScope):
    scope = scope.name
  if suffix is not None:
    if ':' not in suffix:
      suffix += ':'
    scope = (scope or '') + '.*' + suffix
  return tf.compat.v1.get_collection(collection, scope)


# TODO(b/141936246) Replace this function with a V2-safe way to load models.
def _get_variables_to_restore(include=None, exclude=None):
  """Gets the list of the variables to restore.

  Taken from tensorflow/contrib/framework/python/ops/variables.py.

  Args:
    include: an optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to include. None would include all
      the variables.
    exclude: an optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to exclude. None it would not
      exclude any.

  Returns:
    a list of variables to restore.

  Raises:
    TypeError: include or exclude is provided but is not a list or a tuple.
  """
  if include is None:
    # Include all variables.
    vars_to_include = _get_variables()
  else:
    if not isinstance(include, (list, tuple)):
      raise TypeError('include is provided but is not a list or a tuple.')
    vars_to_include = []
    for scope in include:
      vars_to_include += _get_variables(scope)
  vars_to_exclude = set()
  if exclude is not None:
    if not isinstance(exclude, (list, tuple)):
      raise TypeError('exclude is provided but is not a list or a tuple.')
    for scope in exclude:
      vars_to_exclude |= set(_get_variables(scope))
  # Exclude the variables in vars_to_exclude
  return [v for v in vars_to_include if v not in vars_to_exclude]


def apply_saved_model(model_dir, inputs, tags, signature_name=None,
                      output_keys_in_signature=None):
  """Applies a SavedModel to some `Tensor`s.

  Applies a SavedModel to `inputs`. The SavedModel is specified with
  `model_dir`, `tags` and `signature_name`. Note that the SavedModel will be
  converted to an all-constants graph.

  Note: This API can only be used when TF2 is disabled or
  `tft_beam.Context.force_tf_compat_v1=True`.

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
  loaded_graph = tf.compat.v1.Graph()
  loaded_initializer_op_names = []

  with loaded_graph.as_default():
    sess = tf.compat.v1.Session()
    meta_graph = tf.compat.v1.saved_model.load(sess,
                                               export_dir=model_dir,
                                               tags=tags)
    loaded_initializer_op_names = [
        op.name for op in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)
    ]

    if signature_name:
      signature = meta_graph.signature_def[signature_name]
    elif len(meta_graph.signature_def) > 1:
      raise ValueError(
          'The SavedModel contains multiple signatures (%r) but signature_name '
          'was not specified.' % (meta_graph.signature_def.keys(),))
    else:
      signature = next(iter(meta_graph.signature_def.values()))

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
        next(iter(signature.inputs.values())).name: inputs
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
    output_tensor_names = [next(iter(signature.outputs.values())).name]
    output_single_tensor = True

  # Convert_variables_to_constants() requires op name.
  output_op_names = [loaded_graph.get_tensor_by_name(tensor_name).op.name
                     for tensor_name in output_tensor_names]
  constant_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      sess, loaded_graph.as_graph_def(),
      output_op_names + loaded_initializer_op_names)
  sess.close()

  returned_elements = tf.import_graph_def(
      constant_graph_def,
      input_map=input_name_to_tensor_map,
      return_elements=output_tensor_names + loaded_initializer_op_names)
  returned_output_tensors = returned_elements[:len(output_tensor_names)]
  returned_initializer_ops = returned_elements[len(output_tensor_names):]

  for initializer_op in returned_initializer_ops:
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS,
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

  Note: This API can only be used when TF2 is disabled or
  `tft_beam.Context.force_tf_compat_v1=True`.

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
  loaded_graph = tf.compat.v1.Graph()
  with loaded_graph.as_default():
    input_placeholders = [
        tf.compat.v1.placeholder(
            dtype=tensor.dtype, shape=tensor.shape, name=tensor.op.name)
        for tensor in inputs
    ]
    output = fn(*input_placeholders)
    if isinstance(output, tf.Tensor):
      output_tensors = [output]
      output_single_tensor = True
    else:
      output_tensors = output
      output_single_tensor = False

    # TODO(qimingj/kestert): Copy table initializers to the composed graph.
    if tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS):
      raise ValueError('Models with table init ops are not supported.')

    vars_to_restore = _get_variables_to_restore(include=include,
                                                exclude=exclude)
    saver = tf.compat.v1.train.Saver(vars_to_restore)
    with tf.compat.v1.Session() as sess:
      saver.restore(sess, checkpoint)
      output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
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
