"""Helper/utility functions that a tf-transform implementation would find handy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
import tensorflow as tf
import tensorflow_transform.api as api

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.util import nest

_ENCODED_FUNCTION_SIGNATURE_NAME = 'encoded_function'

_SPARSE_TENSOR_NAME_RE = re.compile(r'(.*)\$(indices|values|dense_shape)$')

_DENSE_TENSOR_NAME_RE = re.compile(r'(.*)\$dense_tensor$')


def infer_feature_schema(tensors):
  """Given a dict of tensors, creates a schema.

  Infers a schema, in the format of a TensorFlow "feature spec", for the given
  dictionary of tensors. The resulting schema is suitable for use wherever a
  schema is required in tf-transform. However it may not be valid for passing to
  `tf.parse_example`, as it may be missing shape information.

  Args:
    tensors: A dict whose keys are column names and whose values are `Tensor`s
        or `SparseTensor`s. In either case the tensor's 0'th dimension is
        interpreted as the batch dimension. In order to pass a tensor
        representing a single instance, it must be wrapped in a batch of size 1.

  Returns:
    A dict whose keys are column names and whose values are `FixedLenFeature`
    or `VarLenFeature` objects.
  """
  def remove_batch_dimension(dims):
    """Removes the batch dimension from the given dimensions.

    Given the dimensions representing the shape of a batch of data, returns the
    shape of a corresponding instance, by removing the (initial) batch
    dimension.

    Args:
      dims: A tuple of dimensions, or None if the shape is unspecified.

    Returns:
      A tuple of dimensions representing the shape of an instance, or None if
      the shape is unspecified.

    Raises:
      ValueError: if `dims` doesn't have rank at least 1.
    """
    if dims is None:
      # A batch of tensors with unknown static shape yields an instance of
      # unknown static shape.
      return None
    elif not dims:
      raise ValueError('Expected shape of rank at least 1, but got %r' % dims)
    return tuple(dims)[1:]

  result = {}
  for name, tensor in tensors.items():
    if isinstance(tensor, tf.SparseTensor):
      # TODO(kestert): This won't work with serialization.
      # We need to distinguish between "true" sparse tensors and tensors that
      # should be represented as VarLenFeature's, and how to serialize both
      # kinds.
      feature = tf.VarLenFeature(tensor.dtype)
    else:
      feature = tf.FixedLenFeature(
          remove_batch_dimension(tensor.get_shape().dims), tensor.dtype)
    result[name] = feature
  return result


def make_feed_dict(input_tensors, schema, instance):
  """Creates a feed dict for passing data to the graph.

  Converts an instance from the in-memory representation to a batch (of size 1)
  suitable for passing to `tf.Session.run`.

  Args:
    input_tensors: A map from column names to `Tensor`s or `SparseTensor`s.
    schema: A map from column names to `FixedLenFeature`, `VarLenFeature` or
      `SparseFeature` objects.
    instance: A map from column names to a python primitive, list, or ndarray.

  Returns:
    A map from `Tensor`s or `SparseTensor`s to batches in the format required by
    `tf.Session.run`.

  Raises:
    ValueError: If `schema` contains an invalid feature spec.
  """
  def add_batch_dimension_to_indices(indices):
    """Adds a batch dimension to the indices of a sparse tensor.

    Takes indices of a sparse tensor representing a single instance, and returns
    the indices for a sparse tensor where the first (batch) dimension is 1.

    Args:
      indices: A list of sparse tensor indices.

    Returns:
      A list of indices with a batch dimension prepended.
    """
    if not indices:
      # Indices must have shape (?, 2). Therefore if we encounter an empty
      # sparse tensor, we return an empty ndarray with shape (0, 2).
      return np.empty([0, 2], dtype=np.int64)
    else:
      return [(0, i) for i in indices]

  feed_dict = {}
  for key, input_tensor in input_tensors.items():
    spec = schema[key]
    # TODO(abrao): Validate dtypes, shapes etc.
    if isinstance(spec, tf.FixedLenFeature):
      feed_value = [instance[key]]
    elif isinstance(spec, tf.VarLenFeature):
      max_index = len(instance[key])
      feed_value = tf.SparseTensorValue(
          indices=add_batch_dimension_to_indices(range(max_index)),
          values=instance[key],
          dense_shape=(1, max_index)
      )
    elif isinstance(spec, tf.SparseFeature):
      feed_value = tf.SparseTensorValue(
          indices=add_batch_dimension_to_indices(instance[spec.index_key]),
          values=instance[spec.value_key],
          dense_shape=(1, spec.size)
      )
    else:
      raise ValueError('Invalid feature spec %r.' % spec)
    feed_dict[input_tensor] = feed_value
  return feed_dict


def make_output_dict(schema, fetches):
  """Maps the values fetched by `tf.Session.run` to the in-memory format.

  Args:
    schema: A map from column names to `FixedLenFeature`, `VarLenFeature` or
      `SparseFeature` objects.
    fetches: A dict representing a batch of data, as returned by `Session.run`.

  Returns:
    A map from keys to a python primitive, list or ndarray.

  Raises:
    ValueError: If `schema` contains an invalid feature spec.
  """
  def remove_batch_dimension_from_indices(indices):
    """Removes the batch dimension from the indices of a sparse tensor.

    Takes indices for a sparse tensor representing a batch of size 1, and
    returns indices for a single instance within that batch.

    Args:
      indices: A list of sparse tensor indices.

    Returns:
      A list of indices with the batch dimension removed.
    """
    assert all(index[0] == 0 for index in indices), 'Expected batch of size 1.'
    return [tuple(index)[1] for index in indices]

  output_dict = {}
  for key, value in fetches.items():
    spec = schema[key]
    if isinstance(spec, tf.FixedLenFeature):
      assert len(value) == 1, 'Expected batch of size 1.'
      output_dict[key] = value[0]
    elif isinstance(spec, tf.VarLenFeature):
      if (remove_batch_dimension_from_indices(value.indices) !=
          range(len(value.values))):
        raise ValueError('Encountered a SparseTensorValue that cannot be '
                         'decoded as a VarLenFeature.')
      output_dict[key] = list(value.values)
    elif isinstance(spec, tf.SparseFeature):
      output_dict[spec.index_key] = remove_batch_dimension_from_indices(
          value.indices)
      output_dict[spec.value_key] = value.values
    else:
      raise ValueError('Invalid feature spec %r.' % spec)
  return output_dict


# TODO(b/34253951): remove decompose/recompose once MetaGraphDef supports
# SparseTensor
def _decompose_sparse_tensors(tensor_map):
  """Separates out `SparseTensor`s into their constituent parts.

  Takes a map from column names to `Tensor`s or `SparseTensor`s, and
  decomposes each `SparseTensor` into its parts, assigning each part a new
  column name in the returned map.

  Note that there is never any possiblity of name collision, as every column
  name gets some suffix such as "$values" added to it.  Therefore every expanded
  name can be uniquely mapped back to the original column name.

  Args:
    tensor_map: A map from strings to `Tensor`s or `SparseTensor`s.

  Returns:
    A map from strings to `Tensor`s.
  """
  result = {}

  for key, tensor in tensor_map.items():
    if isinstance(tensor, tf.SparseTensor):
      result[key + '$indices'] = tensor.indices
      result[key + '$values'] = tensor.values
      result[key + '$dense_shape'] = tensor.dense_shape
    else:
      result[key + '$dense_tensor'] = tensor

  return result


def _recompose_sparse_tensors(tensor_map):
  """Undoes the function _decompose_sparse_tensors."""
  # TODO(kestert): Add better validation so any malformed input is caught.
  result = {}

  sparse_keys = set()
  dense_keys = set()
  for key in tensor_map.keys():
    match = _SPARSE_TENSOR_NAME_RE.match(key)
    if match:
      sparse_keys.add(match.group(1))
      continue
    match = _DENSE_TENSOR_NAME_RE.match(key)
    if match:
      dense_keys.add(match.group(1))
      continue
    raise ValueError('Unexpected key: %d' % key)

  for key in sparse_keys:
    result[key] = tf.SparseTensor(tensor_map[key + '$indices'],
                                  tensor_map[key + '$values'],
                                  tensor_map[key + '$dense_shape'])
  for key in dense_keys:
    result[key] = tensor_map[key + '$dense_tensor']

  return result


def _flatten(structure):
  """Like nest.flatten but can handle dicts with string keys."""
  def yield_flattened(structure):
    if nest.is_sequence(structure):
      for substructure in structure:
        for element in yield_flattened(substructure):
          yield element
    elif isinstance(structure, dict):
      for key in sorted(structure.keys()):
        for element in yield_flattened(structure[key]):
          yield element
    else:
      yield structure
  return list(yield_flattened(structure))


def _pack_sequence_as(structure, flat_sequence):
  """Like nest.pack_sequence_as but can handle dicts with string keys."""
  flat_sequence_iter = iter(flat_sequence)
  def unflatten(structure):
    if nest.is_sequence(structure):
      return [unflatten(substructure) for substructure in structure]
    elif isinstance(structure, dict):
      return {key: unflatten(structure[key])
              for key in sorted(structure.keys())}
    else:
      return flat_sequence_iter.next()
  return unflatten(structure)


class TransformFnDef(object):
  """An opaque representation of a TransformFn.

  This representation is used directly by test implementations as the
  type of a TransformFn.  It may also be used by non-test implementations
  e.g. the beam implementation represents the TransformFn as a PCollection
  wrapping a TransformFnDef.

  This format should not be relied on and is only for use in
  make_transform_fn_def and load_transform_fn_def.

  Args:
    meta_graph_def: The underlying representation of this TransformFn as
        a MetaGraphDef.
  """

  def __init__(self, meta_graph_def):
    self._meta_graph_def = None
    # We need to serialize the meta_graph_def here since we cannot use a coder
    # directly due to b/34628545.
    self._meta_graph_def_str = meta_graph_def.SerializeToString()

  @property
  def meta_graph_def(self):
    """Returns the internal representation (don't use outside this module)."""
    # Lazy initialize the meta_graph_def from its serialized form.
    if self._meta_graph_def:
      return self._meta_graph_def
    else:
      self._meta_graph_def = meta_graph_pb2.MetaGraphDef()
      self._meta_graph_def.ParseFromString(self._meta_graph_def_str)
      return self._meta_graph_def


def _generate_constant_tensors(graph_def, tensor_value_mapping):
  """Converts values to constant tensors, using shape information in a graph.

  Takes a graph_def and a mapping from tensor names to values.  Returns a
  dictionary of constant tensors that represent these same values but with shape
  inferred from the shape of the tensors in the graph_def.

  This is done by feeding the values in tensor_value_mapping into the
  graph defined by the graph_def, and then fetching the same values to obtain
  a new representation of the values.  This representation contains potentially
  more shape information than the original values, and is used to construct a
  constant tensor.

  Args:
    graph_def: A `GraphDef`.
    tensor_value_mapping: A map from tensor names to values.

  Returns:
    A map from tensor names to tf.constant value representing the values in
        `tensor_value_mapping`.
  """
  g = tf.Graph()
  with g.as_default():
    names = tensor_value_mapping.keys()
    tensors = tf.import_graph_def(graph_def, return_elements=names)
    dtypes = [tensor.dtype for tensor in tensors]
    with tf.Session(graph=g) as sess:
      new_values = sess.run(
          tensors,
          {tensor: tensor_value_mapping[name]
           for name, tensor in zip(names, tensors)})

  # In the default graph, construct dict mapping names to constant tensors.
  return {name: tf.constant(new_value, dtype=dtype)
          for name, new_value, dtype in zip(names, new_values, dtypes)}


def replace_tensors_with_constant_values(
    transform_fn_def, tensor_value_mapping=None):
  """Takes a TransformFnDef and replaces some tensors with constant values.

  Replaces the tensors referenced in `tensor_value_mapping` with the
  corresponding constants.  These tensors will be replaced in the GraphDef of
  the MetaGraphDef.  Since this involves deserializing and serializing the
  graph, the names of some tensors may change.  Thus we also rewrite the
  signatures in the MetaGraphDef to refer to the new tensor names.

  Only for use with MetaGraphDef's created with
  export_function_as_meta_graph_def.

  NOTE: The keys in tensor_value_mapping should not overlap with the tensor
  names of the inputs to the transform function (i.e. the inputs in the
  signature of transform_fn_def.meta_graph_def).  This would not make sense
  since that constant would be overridden when the graph was fed by these
  inputs.

  Args:
    transform_fn_def: A `TransformFnDef`.
    tensor_value_mapping: A map from tensor names to values, which will be
        replaced in the serialized graph.

  Returns:
    A TransformFnDef representing the new graph.

  Raises:
    ValueError: When the keys of tensor_value_mapping and the input signature
        overlap.
  """
  old_meta_graph_def = transform_fn_def.meta_graph_def
  old_signature = old_meta_graph_def.signature_def[
      _ENCODED_FUNCTION_SIGNATURE_NAME]

  # Represent signature as pair of dicts of tensor names.
  old_signature_dict = {
      'inputs': {
          key: tensor_info.name
          for key, tensor_info in old_signature.inputs.items()
      },
      'outputs': {
          key: tensor_info.name
          for key, tensor_info in old_signature.outputs.items()
      }
  }

  # Check that no input to the transform function will get overridden by a
  # constant
  for input_tensor_name in old_signature_dict['inputs'].values():
    if input_tensor_name in tensor_value_mapping:
      raise ValueError('The tensor %s appeared both in the input signature to '
                       'the transform function and in '
                       'tensor_value_mapping.  This is invalid as '
                       'overriding an input with a constant would cause that '
                       'constant to be overridden when the graph was fed.' %
                       input_tensor_name)

  g = tf.Graph()
  with g.as_default():
    # Create input map to replace values with a constant tensor.
    input_map = _generate_constant_tensors(
        old_meta_graph_def.graph_def, tensor_value_mapping)

    # Import the graph def, keeping track of how all the tensors referenced in
    # the signature get (potentially) renamed.  We pack and unpack return
    # elements since import_graph_def only accepts and returns a list of return
    # elements.
    flattened_old_signature = _flatten(old_signature_dict)
    flattened_new_signature = tf.import_graph_def(
        old_meta_graph_def.graph_def, input_map=input_map,
        return_elements=flattened_old_signature)
    new_signature_dict = _pack_sequence_as(
        old_signature_dict, flattened_new_signature)

  meta_graph_def = meta_graph_pb2.MetaGraphDef()
  # Replace graph with new graph that has bound values replaced by constants.
  meta_graph_def.graph_def.CopyFrom(g.as_graph_def())
  # Replace old column names in the signature with new column names.  Copy over
  # old signature because the rest of the `TensorInfo`s in the signature will be
  # the same.
  new_signature = meta_graph_def.signature_def[_ENCODED_FUNCTION_SIGNATURE_NAME]
  new_signature.CopyFrom(old_signature)
  for key, tensor in new_signature_dict['inputs'].items():
    new_signature.inputs[key].name = tensor.name
  for key, tensor in new_signature_dict['outputs'].items():
    new_signature.outputs[key].name = tensor.name

  return TransformFnDef(meta_graph_def)


def make_transform_fn_def(graph, inputs, outputs):
  """Creates a TransformFnDef representing a graph.

  Exports a function, represented as a graph with input and output tensors, as
  a TransformFnDef.  The arguments are a graph, together with a dictionary
  mapping input names to tensors in the graph, and output names to tensors in
  the graph, where tensors may be `Tensor`s or `SparseTensor`s.

  This function creates a MetaGraphDef that encodes the transform function.
  The MetaGraphDef contains the serialized graph as its graph_def and also
  a signature which maps the input and output columns to tensors in the graph.
  Because MetaGraphDef doesn't have native support for sparse tensors, we
  rely on a naming convention to encode sparse tensors in the input/output
  signatures.

  Args:
    graph: The `Graph` containing the function.
    inputs: A dict from strings to `Tensor`s or `SparseTensor`s representing
        the inputs to the function.
    outputs: A dict from strings to `Tensor`s or `SparseTensor`s representing
        the outputs of the function.

  Returns:
    A TransformFnDef containing the serialized graph and the signature for
    the function.
  """
  meta_graph_def = meta_graph_pb2.MetaGraphDef()

  # Serialize the graph
  meta_graph_def.graph_def.CopyFrom(graph.as_graph_def())

  # Encode the inputs and outputs of the graph in a signature
  signature = meta_graph_def.signature_def[_ENCODED_FUNCTION_SIGNATURE_NAME]
  signature.method_name = _ENCODED_FUNCTION_SIGNATURE_NAME
  for tensor_info_map, tensor_map in [(signature.inputs, inputs),
                                      (signature.outputs, outputs)]:
    for key, tensor in _decompose_sparse_tensors(tensor_map).items():
      tensor_info_map[key].name = tensor.name
      tensor_info_map[key].dtype = tensor.dtype.as_datatype_enum
      tensor_info_map[key].tensor_shape.CopyFrom(tensor.get_shape().as_proto())

  return TransformFnDef(meta_graph_def)


def apply_transform_fn_def(transform_fn_def, inputs):
  """Loads a TransformFnDef into a graph.

  Args:
    transform_fn_def: A representation of the preprocessing graph as a
        TransformFnDef.
    inputs: A dict whose keys are the names of the input columns, and whose
        values are `Tensor`s or `SparseTensor`s representing these columns.

  Returns:
    A dict whose keys are output column names and whose values are `Tensor`s or
    `SparseTensor`s representing these columns.
  """
  meta_graph_def = transform_fn_def.meta_graph_def
  signature = meta_graph_def.signature_def[
      _ENCODED_FUNCTION_SIGNATURE_NAME]

  # Build an input_map which maps tensor names in the meta_graph_def, to tensors
  # in the graph, as specified by `inputs`.
  input_map = {}
  for key, tensor in _decompose_sparse_tensors(inputs).items():
    input_map[signature.inputs[key].name] = tensor

  # Import the preprocessing graph, building a map from the column names in the
  # meta_graph_def to the tensors in the graph.  There will still be three
  # column names for each sparse tensor.  We pack and unpack return_elements
  # since import_graph_def only accepts and returns a list of return elements.
  output_column_names = {
      key: tensor_info.name for key, tensor_info in signature.outputs.items()
  }
  flattened_output_column_names = _flatten(output_column_names)
  flattened_output_column_tensors = tf.import_graph_def(
      meta_graph_def.graph_def, input_map=input_map,
      return_elements=flattened_output_column_names)
  output_column_tensors = _pack_sequence_as(
      output_column_names, flattened_output_column_tensors)
  # Merge columns that represent sparse tensors.
  return _recompose_sparse_tensors(output_column_tensors)


def load_transform_fn_def(transform_fn_def):
  """Loads a TransformFnDef into a graph.

  Similar to apply_transform_fn_def except it loads input placeholders and
  returns a column to tensor mapping for inputs.

  Args:
    transform_fn_def: A representation of the preprocessing graph as a
        TransformFnDef.

  Returns:
    A pair of dicts, for inputs and outputs, whose keys are column names and
    whose values are `Tensor`s or `SparseTensor`s representing these columns.
  """
  meta_graph_def = transform_fn_def.meta_graph_def
  signature = meta_graph_def.signature_def[
      _ENCODED_FUNCTION_SIGNATURE_NAME]

  # Construct placeholders for input columns.  We don't load the original
  # placeholders because doing so causes the loaded graph to lose all tensor
  # shape information.
  inputs = _recompose_sparse_tensors({
      key: tf.placeholder(tensor_info.dtype, tensor_info.tensor_shape)
      for key, tensor_info in signature.inputs.items()})

  return (inputs, apply_transform_fn_def(transform_fn_def, inputs))


def run_preprocessing_fn(preprocessing_fn, schema, graph):
  """Constructs the preprocessing graph.

  Args:
    preprocessing_fn: A function that takes a dict of `Column`s as input and
      returns a dict of `Column`s as output.
    schema: A dict mapping column names to `tf.FixedLenFeature`,
      `tf.VarLenFeature` or `tf.SparseFeature` objects.
    graph: A `tf.Graph` object.

  Returns:
    A tuple of input columns and output columns.

  Raises:
    ValueError: If `schema` contains unsupported feature types.
  """
  for key, feature_spec in schema.items():
    if not isinstance(feature_spec, (tf.VarLenFeature, tf.SparseFeature,
                                     tf.FixedLenFeature)):
      raise ValueError('Invalid type for column "%r". Was expecting one of: '
                       '(tf.FixedLenFeature, tf.VarLenFeature, '
                       'tf.SparseFeature) but got: %r' % (key, feature_spec))

  with graph.as_default():
    inputs = {}
    # Given the schema, we need to construct the tensors that we expect to be
    # fed at runtime. To do this we rely on the canonical implementation, namely
    # parse_example, which takes a dict of feature specs and returns a dict of
    # tensors with dtypes and shapes corresponding to a batch of data. We then
    # use this information to construct appropriate placeholders. Note: the
    # tensors output by parse_examples are only used for their type information
    # and never actually wired into the graph, therefore the parse_example op
    # will not get executed.
    batched_tensors = tf.parse_example(tf.placeholder(tf.string), schema)
    for key, tensor in batched_tensors.items():
      if isinstance(tensor, tf.SparseTensor):
        # TODO(abrao): Figure out how to create sparse placeholders with
        # partially known shapes such as (None, None). For now we allow values
        # of any shape to be fed.
        placeholder = tf.sparse_placeholder(tensor.dtype, None)
      else:
        placeholder = tf.placeholder(tensor.dtype, tensor.get_shape())
      # pylint: disable=protected-access
      inputs[key] = api._InputColumn(placeholder, key)

    # Construct the deferred preprocessing graph by calling preprocessing_fn on
    # the inputs.
    outputs = preprocessing_fn(inputs)

  return inputs, outputs
