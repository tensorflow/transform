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
"""Nodes that define the Beam execution graph.

`OperationNode`s are objects that define a graph of operations similar to the
graph of `beam.PTransform`s.
`tensorflow_transform.beam.analysis_graph_builder.build` converts the graph
defined the user's preprocessing_fn into a new graph of `OperationNode`s which
in turn is in turn implemented as a graph of `beam.PTransform`s by
`tensorflow_transform.beam.common.ConstructBeamPipelineVisitor`.  A registration
system is used to register implementations of each individual `OperationDef`.
The `OperationDef`s defined by the user in their preprocessing_fn are all
subclasses of `AnayzerDef` (except `TensorSource`, which gets converted to
`ExtractFromDict` in `tensorflow_transform.beam.analysis_graph_builder.build`).
The subclasses of `AnalyzerDef` are defined in
`tensorflow_transform.analyzer_nodes` and are implemented in
`tensorflow_transform.beam.analyzer_impls`.

This module contains the nodes that are created by
`tensorflow_transform.beam.analysis_graph_builder.build`.  These nodes define
the parts of the beam graph that run a TensorFlow graph in a `beam.ParDo`,
extract `PCollections` containing tuples of tensors required by analyzers,
run the analyzers, and then create a new (deferred) TensorFlow graph where
the results of analyzers are replaced by constant tensors.  This happens in a
number of phases, since an analyzer might depend on a tensor that in turn
depends on the result of another analyzer.

The `OperationDef` subclasses defined here are implemented in
`tensorflow_transform.beam.impl`.
"""

import tensorflow as tf
from tensorflow_transform import nodes
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple


class CreateTensorBinding(
    tfx_namedtuple.namedtuple(
        'CreateTensorBinding',
        ['tensor_name', 'dtype_enum', 'is_asset_filepath', 'label']),
    nodes.OperationDef):
  """An operation that represents creating a tensor binding from a value.

  This `OperationDef` represents a `beam.PTransform` that applies a ParDo
  (where the input PCollection is assumed to contain a single element), which
  combines the single element with the a tensor name and `is_asset_filepath`
  to create a tensor binding.

  Attributes:
    tensor_name: The name of the tensor that the given value should replace as a
        constant tensor.
    dtype_enum: The Dtype of the tensor as a TF `types_pb2.DataType`.
    is_asset_filepath: If true, then the replaced value will be added to the
        ASSET_FILEPATHS collection if exporting a TF1 Graph.
    label: A unique label for this operation.
  """
  __slots__ = ()


class CreateSavedModel(
    tfx_namedtuple.namedtuple(
        'CreateSavedModel',
        ['table_initializers', 'output_signature', 'label']),
    nodes.OperationDef):
  """An operation that represents creating a SavedModel with bound values.

  This operation represents creating a SavedModel.  Its output is a
  PCollection containing a single element which is the directory containing the
  `SavedModel`.  The inputs are a PCollection of tensor bindings.  A tensor
  binding is the specification of a tensor and a value that it should be
  replaced with in the graph.

  This allows us to create a `SavedModel` in a deferred manner, which depends on
  deferred values (the tensor bindings) which were not known when the Beam graph
  was constructed.


  Attributes:
    table_initializers: A list of table initializer ops that should be run as
        part of this SavedModel.
    output_signature: The output signature of this `SavedModel`, as a dictionary
        whose keys are feature names and values are `Tensor`s or
        `SparseTensor`s.
    label: A unique label for this operation.
  """
  __slots__ = ()

  def _get_tensor_type_name(self, tensor):
    if isinstance(tensor, tf.Tensor):
      return 'Tensor'
    elif isinstance(tensor, tf.SparseTensor):
      return 'SparseTensor'
    raise ValueError('Got a {}, expected a Tensor or SparseTensor'.format(
        type(tensor)))

  def get_field_str(self, field_name):
    # Overriding the str representation of table initializers since it may be
    # different for various versions of TF.
    if field_name == 'table_initializers':
      return '{}'.format(len(self.table_initializers))
    elif field_name == 'output_signature':
      copied = self.output_signature.copy()
      for key in copied:
        value = self.output_signature[key]
        copied[key] = '{}<shape: {}, {}>'.format(
            self._get_tensor_type_name(value), value.shape.as_list(),
            value.dtype)
      return str(copied)
    return super().get_field_str(field_name)


class ExtractInputForSavedModel(
    tfx_namedtuple.namedtuple('ExtractInputForSavedModel',
                              ['dataset_key', 'label']), nodes.OperationDef):
  """An operation that forwards the requested dataset in PCollection form.

  The resulting PCollection is either the dataset corresponding to
  `dataset_key`, or a flattened PCollection if `dataset_key` is not specified.

  Attributes:
    dataset_key: (Optional) dataset key str.
    label: A unique label for this operation.
  """
  __slots__ = ()


class ApplySavedModel(
    tfx_namedtuple.namedtuple('ApplySavedModel', ['phase', 'label']),
    nodes.OperationDef):
  """An operation that represents applying a SavedModel as a `beam.ParDo`.

  This operation represents applying a `SavedModel`, which is the input to this
  operation, to the input values.  The inputs values are not an input to this
  operation, but are provided to the implementation by
  `tensorflow_transform.beam.common.ConstructBeamPipelineVisitor.ExtraArgs`.

  The input should be a PCollection containing a single element which is the
  directory containing the SavedModel to be run.

  Attributes:
    phase: An integer which is the phase that this operation is run as part of.
    label: A unique label for this operation.
  """
  __slots__ = ()

  @property
  def is_partitionable(self):
    return True


class ExtractFromDict(
    tfx_namedtuple.namedtuple('ExtractFromDict', ['keys', 'label']),
    nodes.OperationDef):
  """An operation that represents extracting values from a dictionary.

  This operation represents a `beam.ParDo` that is applied to a PCollection
  whose elements are assumed to be a dictionary of values.  For each element of
  the PCollection, this corresponding element of the output PCollection is a
  tuple of values, one for each key.

  Attributes:
    keys: The keys whose values should be extracted from each element of the
        input PCollection. keys should either be a tuple or a string.
    label: A unique label for this operation.
  """
  __slots__ = ()

  @property
  def is_partitionable(self):
    return True


class Flatten(
    tfx_namedtuple.namedtuple('Flatten', ['label']), nodes.OperationDef):
  __slots__ = ()

  @property
  def is_partitionable(self):
    return True
