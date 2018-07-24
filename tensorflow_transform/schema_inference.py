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


import six
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_schema


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
    graph: A tf.Graph, used to look up schema overrides even they are not
        computed.
    session: (optional) A `tf.Session` used to compute schema overrides.  If
        None, schema overrides will not be computed.

  Returns:
    A `Schema` object.
  """
  tensor_overrides = _get_tensor_schema_overrides(graph)

  column_schemas = {}
  for name, tensor in six.iteritems(features):
    column_schema = dataset_schema.infer_column_schema_from_tensor(tensor)
    override_min_and_max = tensor_overrides.get(
        tensor.values if isinstance(tensor, tf.SparseTensor) else tensor)
    if override_min_and_max is not None:
      assert column_schema.domain.dtype == tf.int64
      assert isinstance(column_schema.domain, dataset_schema.IntDomain)
      if session is not None:
        min_value, max_value = session.run(override_min_and_max)
      else:
        min_value, max_value = None, None
      column_schemas[name] = dataset_schema.ColumnSchema(
          dataset_schema.IntDomain(tf.int64, min_value, max_value,
                                   is_categorical=True),
          column_schema.axes,
          column_schema.representation)
    else:
      column_schemas[name] = column_schema

  return dataset_schema.Schema(column_schemas)


# Names of collections, which should all be the same length and contain tensors.
# Each tensor in the first collection should have its min/max described by the
# tensors in the other two collections.
_TF_METADATA_TENSOR_COLLECTION = 'tft_schema_override_tensor'
_TF_METADATA_TENSOR_MIN_COLLECTION = 'tft_schema_override_min'
_TF_METADATA_TENSOR_MAX_COLLECTION = 'tft_schema_override_max'


def set_tensor_schema_override(tensor, min_value, max_value):
  """Override parts of the schema of a `Tensor`."""
  if not isinstance(tensor, tf.Tensor):
    raise ValueError('tensor {} was not a Tensor'.format(tensor))
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
