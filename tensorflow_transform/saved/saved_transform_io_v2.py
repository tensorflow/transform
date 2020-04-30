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

import itertools

# GOOGLE-INITIALIZATION

import six
import tensorflow as tf
from tensorflow_transform.saved import constants
from tensorflow_transform.saved import saved_model_loader
from tensorflow_transform.saved import saved_transform_io
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
# pylint: enable=g-direct-tensorflow-import


# TODO(varshaan): Move to graph_tools. This util is refactored from
# ops/op_selector.py.
def _map_subgraph(sources, sinks):
  """Captures subgraph between sources and sinks.

  Walk a Graph backwards from `sinks` to `sources` and returns any extra sources
  encountered in the subgraph that were not specified in `sources`.

  Arguments:
    sources:  An iterable of Tensors where subgraph extraction should stop.
    sinks:  An iterable of Operations where the subgraph terminates.

  Returns:
    The set of placeholders upon which `sinks` depend and are not in `sources`.
  """
  stop_at_tensors = object_identity.ObjectIdentitySet(sources)
  ops_to_visit = object_identity.ObjectIdentitySet(sinks)
  visited_ops = object_identity.ObjectIdentitySet()
  potential_extra_sources = object_identity.ObjectIdentitySet()
  while ops_to_visit:
    op = ops_to_visit.pop()
    visited_ops.add(op)

    if op.type == 'Placeholder':
      potential_extra_sources.update(op.outputs)

    input_ops = [t.op for t in op.inputs if t not in stop_at_tensors]
    for input_op in itertools.chain(input_ops, op.control_inputs):
      if input_op not in visited_ops:
        ops_to_visit.add(input_op)

  return potential_extra_sources.difference(sources)


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
    self._imported = tf.compat.v2.saved_model.load(saved_model_dir,
                                                   [constants.TRANSFORM_TAG])
    self._wrapped = self._imported.signatures[constants.TRANSFORM_SIGNATURE]
    self._input_signature = self._get_input_signature(saved_model_dir)

  def _get_input_signature(self, saved_model_dir):
    saved_model = saved_model_loader.parse_saved_model(saved_model_dir)
    meta_graph_def = saved_model_loader.choose_meta_graph_def(
        saved_model, [constants.TRANSFORM_TAG])
    signature = meta_graph_def.signature_def[constants.TRANSFORM_SIGNATURE]
    return signature.inputs

  def _as_operation(self, op_or_tensor):
    if isinstance(op_or_tensor, ops.Tensor):
      return op_or_tensor.op
    return op_or_tensor

  def _get_component_tensor_ops(self, tensor):
    """Get all component tensors as Tensorflow ops.

    Args:
      tensor: A `Tensor` or `CompositeTensor`.

    Returns:
      All `Tensor` component ops of `tensor`.

    Raises:
      ValueError if supplied `tensor` parameter is neither a `Tensor` nor a
      `CompositeTensor`.
    """
    components = None
    if isinstance(tensor, ops.Tensor):
      components = [tensor]
    elif isinstance(tensor, composite_tensor.CompositeTensor):
      components = nest.flatten(tensor, expand_composites=True)
    else:
      raise ValueError(
          'Unsupported tensor. Arg `tensor` is neither a `Tensor` nor a '
          '`CompositeTensor`: {}.'.format(tensor))
    return [self._as_operation(component) for component in components]

  def apply_v1_transform_model_in_v2(self, logical_input_map):
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
    unexpected_inputs = (
        set(six.iterkeys(logical_input_map)) -
        set(six.iterkeys(self._input_signature)))
    if unexpected_inputs:
      raise ValueError(
          'Unexpected inputs to transform: {}'.format(unexpected_inputs))

    input_map = (
        saved_transform_io._expand_input_map(  # pylint: disable=protected-access
            logical_input_map, self._input_signature))
    remapped_inputs = {}
    feeds = []
    for name in six.iterkeys(input_map):
      tensor = self._wrapped.graph.get_tensor_by_name(name)
      tensor.shape.assert_is_compatible_with(input_map[name].shape)
      remapped_inputs[tensor.op.name] = input_map[name]
      feeds.append(tensor)

    output_signature = self._wrapped.structured_outputs
    fetches = {}
    for name, output in six.iteritems(output_signature):
      sinks = self._get_component_tensor_ops(output)
      extra_sources = _map_subgraph(feeds, sinks)
      # If output does not depend on an input placeholder that is not being fed,
      # add it to fetches.
      if not extra_sources.difference(self._wrapped.graph.internal_captures):
        fetches[name] = output

    pruned = self._wrapped.prune(feeds, fetches)
    return pruned(**remapped_inputs)
