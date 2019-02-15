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
"""Logic associated with schema inference and propagation.

This module contains functionality to set the schema assciated with a Tensor,
and to infer the schema for a tensor, including any information that has been
set.  This module will also contain any schema propagation logic, i.e. deducing
the schema of a tensor from its parents in the graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION

import six
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_schema

from tensorflow_metadata.proto.v0 import schema_pb2


def _feature_spec_from_batched_tensors(tensors):
  """Infer `Schema` proto from a dict of tensors.

  Args:
    tensors: A dict whose keys are strings and values are `Tensor` or
        `SparseTensor`s.

  Returns:
    A `Schema` proto inferred from the types and shapes of the tensors.

  Raises:
    ValueError: If the feature spec cannot be inferred.
    TypeError: If any of the values of `tensors` are not a `Tensor` or
        `SparseTensor`.
  """
  result = {}
  for name, tensor in six.iteritems(tensors):
    tensor = tensors[name]
    shape = tensor.get_shape()
    if tensor.dtype not in (tf.string, tf.int64, tf.float32):
      raise ValueError('{} had invalid dtype {} for feature spec'.format(
          tensor, tensor.dtype))
    if isinstance(tensor, tf.SparseTensor):
      if shape.ndims != 2:
        raise ValueError(
            '{} had invalid shape {} for VarLenFeature: must have rank '
            '2'.format(tensor, shape))
      result[name] = tf.VarLenFeature(tensor.dtype)
    elif isinstance(tensor, tf.Tensor):
      if shape.ndims in [None, 0]:
        raise ValueError(
            '{} had invalid shape {} for FixedLenFeature: must have rank '
            'at least 1'.format(tensor, shape))
      if any(dim is None for dim in shape.as_list()[1:]):
        raise ValueError(
            '{} had invalid shape {} for FixedLenFeature: apart from the batch '
            'dimension, all dimensions must have known size'.format(
                tensor, shape))
      result[name] = tf.FixedLenFeature(shape.as_list()[1:], tensor.dtype)
    else:
      raise TypeError(
          'Expected a Tensor or SparseTensor, got {} of type {}'.format(
              tensor, type(tensor)))

  return result


def infer_feature_schema(features, graph, session=None):
  """Given a dict of tensors, creates a `Schema`.

  Infers a schema, in the format of a tf.Transform `Schema`, for the given
  dictionary of tensors.

  If there is an override specified, we override the inferred schema for the
  given feature's tensor.  An override has the meaning that we should set
  is_categorical=True.  If session is not provided then we just set
  is_categorical=True, and if the session is provided then was also compute
  values of the tensors representing the min and max values and set them in the
  schema.

  Args:
    features: A dict mapping column names to `Tensor` or `SparseTensor`s. The
        `Tensor` or `SparseTensor`s should have a 0'th dimension which is
        interpreted as the batch dimension.
    graph: A `tf.Graph` used to determine schema overrides.
    session: (optional) A `tf.Session` used to compute schema overrides.  If
        None, schema overrides will not be computed.

  Returns:
    A `Schema` object.
  """
  tensor_ranges = _get_tensor_schema_overrides(graph)
  if session is None:
    tensor_ranges = {tensor: (None, None) for tensor in tensor_ranges.keys()}
  else:
    tensor_ranges = session.run(tensor_ranges)

  domains = {}
  for name, tensor in six.iteritems(features):
    values = tensor.values if isinstance(tensor, tf.SparseTensor) else tensor
    if values in tensor_ranges:
      assert values.dtype == tf.int64
      min_value, max_value = tensor_ranges[values]
      domains[name] = schema_pb2.IntDomain(
          min=min_value, max=max_value, is_categorical=True)

  feature_spec = _feature_spec_from_batched_tensors(features)

  return dataset_schema.from_feature_spec(feature_spec, domains)


# Names of collections, which should all be the same length and contain tensors.
# Each tensor in the first collection should have its min/max described by the
# tensors in the other two collections.
_TF_METADATA_TENSOR_COLLECTION = 'tft_schema_override_tensor'
_TF_METADATA_TENSOR_MIN_COLLECTION = 'tft_schema_override_min'
_TF_METADATA_TENSOR_MAX_COLLECTION = 'tft_schema_override_max'


def set_tensor_schema_override(tensor, min_value, max_value):
  """Override parts of the schema of a `Tensor`.

  Args:
    tensor: The `Tensor` whose range is being set.  Must have dtype int64.
    min_value: A `Tensor` representing the min value of `tensor`.
    max_value: A `Tensor` representing the max value of `tensor`.

  Raises:
    ValueError: If any arguments are invalid.
  """
  if not isinstance(tensor, tf.Tensor):
    raise ValueError('tensor {} was not a Tensor'.format(tensor))
  if tensor.dtype != tf.int64:
    raise ValueError(
        'Range can only be set for feature of type tf.int64, got {}'.format(
            tensor.dtype))
  if not isinstance(min_value, tf.Tensor):
    raise ValueError('min_vaue {} was not a Tensor'.format(min_value))
  if not isinstance(max_value, tf.Tensor):
    raise ValueError('max_vaue {} was not a Tensor'.format(min_value))
  tf.add_to_collection(_TF_METADATA_TENSOR_COLLECTION, tensor)
  tf.add_to_collection(_TF_METADATA_TENSOR_MIN_COLLECTION, min_value)
  tf.add_to_collection(_TF_METADATA_TENSOR_MAX_COLLECTION, max_value)


def _get_tensor_schema_overrides(graph):
  """Lookup overrides for `Tensor`s  or `SparseTensor`s."""
  tensors = graph.get_collection(_TF_METADATA_TENSOR_COLLECTION)
  min_values = graph.get_collection(_TF_METADATA_TENSOR_MIN_COLLECTION)
  max_values = graph.get_collection(_TF_METADATA_TENSOR_MAX_COLLECTION)
  assert len(tensors) == len(min_values), '{} != {}'.format(tensors, min_values)
  assert len(tensors) == len(max_values), '{} != {}'.format(tensors, max_values)
  return dict(zip(tensors, zip(min_values, max_values)))
