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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


from tensorflow_transform import nodes


class CreateTensorBinding(
    collections.namedtuple(
        'CreateTensorBinding', ['tensor', 'is_asset_filepath', 'label']),
    nodes.OperationDef):
  """An operation that represents creating a tensor binding from a value.

  This `OperationDef` represents a `beam.PTransform` that applies a ParDo
  (where the input PCollection is assumed to contain a single element), which
  combines the single element with the a tensor name and `is_asset_filepath`
  to create a tensor binding.

  Fields:
    tensor: The name of the tensor that the given value should replace as a
        constant tensor.
    is_asset_filepath: If true, then the replaced value will be added to the
        ASSET_FILEPATHS collection.
    label: A unique label for this operation.
  """
  pass


class CreateSavedModel(
    collections.namedtuple(
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


  Fields:
    table_initializers: A list of table initializer ops that should be run as
        part of this SavedModel.
    output_signature: The output signature of this `SavedModel`, as a dictionary
        whose keys are feature names and values are `Tensor`s or
        `SparseTensor`s.
    label: A unique label for this operation.
  """
  pass


class ApplySavedModel(
    collections.namedtuple('ApplySavedModel',
                           ['dataset_key', 'phase', 'label']),
    nodes.OperationDef):
  """An operation that represents applying a SavedModel as a `beam.ParDo`.

  This operation represents applying a `SavedModel`, which is theinput to this
  operation, to the input values.  The inputs values are not an input to this
  operation, but are provided to the implementation by
  `tensorflow_transform.beam.common.ConstructBeamPipelineVisitor.ExtraArgs`.

  The input should be a PCollection containing a single element which is the
  directory containing the SavedModel to be run.

  Args:
    phase: An integer which is the phase that this operation is run as part of.
    label: A unique label for this operation.
  """

  @property
  def is_partitionable(self):
    return True


class ExtractFromDict(
    collections.namedtuple('ExtractFromDict', ['keys', 'label']),
    nodes.OperationDef):
  """An operation that represents extracting values from a dictionary.

  This operation represents a `beam.ParDo` that is applied to a PCollection
  whose elements are assumed to be a dictionary of values.  For each element of
  the PCollection, this corresponding element of the output PCollection is a
  tuple of values, one for each key.

  Args:
    key: The keys whose values should be extracted from each element of the
        input PCollection.
    label: A unique label for this operation.
  """

  @property
  def is_partitionable(self):
    return True


class Flatten(collections.namedtuple('Flatten', ['label']), nodes.OperationDef):

  @property
  def is_partitionable(self):
    return True
