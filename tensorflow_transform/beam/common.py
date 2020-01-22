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
"""Constants and types shared by tf.Transform Beam package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import uuid

# GOOGLE-INITIALIZATION

import apache_beam as beam
from apache_beam.typehints import Union
from six import binary_type
from six import integer_types
from six import string_types
from tensorflow_transform import nodes

NUMERIC_TYPE = Union[float, Union[integer_types]]
PRIMITIVE_TYPE = Union[NUMERIC_TYPE, Union[string_types], binary_type]

METRICS_NAMESPACE = 'tfx.Transform'


def get_unique_temp_path(base_temp_dir):
  """Return a path to a unique temp dir from given base temp dir.

  Note this doesn't create the path that it returns.

  Args:
    base_temp_dir: A base directory

  Returns:
    The path name of a subdirectory of base_temp_dir, where the subdirectory is
        unique.
  """
  return os.path.join(base_temp_dir, uuid.uuid4().hex)


_PTRANSFORM_BY_OPERATION_DEF_SUBCLASS = {}


def register_ptransform(operation_def_subclass):
  """Decorator to register a PTransform as the implementation for an analyzer.

  This function is used to define implementations of the analyzers defined in
  tensorflow_transform/analyzer_nodes.py and also the internal operations
  defined in tensorflow_transform/beam/beam_nodes.py.  The registered PTransform
  will be invoked as follows:

  outputs = inputs | operation.label >> MyPTransform(operation, extra_args)

  where operation is a the instance of the subclass that was registered,
  extra_args are global arguments available to each PTransform (see
  ConstructBeamPipelineVisitor.extra_args) and `inputs` is a tuple of
  PCollections correpsonding to the inputs of the OperationNode being
  implemented.  The return value `outputs` should be a a tuple of PCollections
  corresponding to the outputs of the OperationNode.  If the OperationNode has
  a single output then the return value can also be a PCollection instead of a
  tuple.

  In some cases the implementation cannot be a PTransform and so instead the
  value being registered may also be a function.  The registered function will
  be invoked as follows:

  outputs = my_function(inputs, operation, extra_args)

  where inputs, operation, extra_args and outputs are the same as for the
  PTransform case.

  Args:
    operation_def_subclass: The class of attributes that is being registered.
        Should be a subclass of `tensorflow_transform.nodes.OperationDef`.

  Returns:
    A class decorator that registers a PTransform or function as an
        implementation of the OperationDef subclass.
  """

  def register(ptransform_class):
    assert isinstance(ptransform_class, type)
    assert issubclass(ptransform_class, beam.PTransform)
    assert operation_def_subclass not in _PTRANSFORM_BY_OPERATION_DEF_SUBCLASS
    _PTRANSFORM_BY_OPERATION_DEF_SUBCLASS[operation_def_subclass] = (
        ptransform_class)
    return ptransform_class

  return register


class ConstructBeamPipelineVisitor(nodes.Visitor):
  """Visitor that constructs the beam pipeline from the node graph."""

  ExtraArgs = collections.namedtuple(  # pylint: disable=invalid-name
      'ExtraArgs', [
          'base_temp_dir',
          'pipeline',
          'flat_pcollection',
          'pcollection_dict',
          'tf_config',
          'graph',
          'input_signature',
          'input_schema',
          'input_tensor_adapter_config',
          'use_tfxio',
          'cache_pcoll_dict',
      ])

  def __init__(self, extra_args):
    self._extra_args = extra_args

  def visit(self, operation, inputs):
    try:
      ptransform = _PTRANSFORM_BY_OPERATION_DEF_SUBCLASS[operation.__class__]
    except KeyError:
      raise ValueError('No implementation for {} was registered'.format(
          operation))

    # TODO(zoyahav): Consider extracting a single PCollection before passing to
    # ptransform if len(inputs) == 1.
    outputs = ((inputs or beam.pvalue.PBegin(self._extra_args.pipeline))
               | operation.label >> ptransform(operation, self._extra_args))

    if isinstance(outputs, beam.pvalue.PCollection):
      return (outputs,)
    else:
      return outputs

  def validate_value(self, value):
    if not isinstance(value, beam.pvalue.PCollection):
      raise TypeError('Expected a PCollection, got {} of type {}'.format(
          value, type(value)))


class IncrementCounter(beam.PTransform):
  """A PTransform that increments a counter once per PCollection.

  The output PCollection is the same as the input PCollection.
  """

  def __init__(self, counter_name):
    self._counter_name = counter_name

  def _make_and_increment_counter(self, unused_element):
    del unused_element
    beam.metrics.Metrics.counter(METRICS_NAMESPACE, self._counter_name).inc()
    return None

  def expand(self, pcoll):
    _ = (
        pcoll.pipeline
        | 'CreateSole' >> beam.Create([None])
        | 'Count' >> beam.Map(self._make_and_increment_counter))
    return pcoll
