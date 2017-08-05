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
"""Reader for v1 JSON to `Schema`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_schema as sch


def from_schema_json(schema_json):
  """Translate a v1 JSON schema into a `Schema`."""
  schema_dict = json.loads(schema_json)
  feature_column_schemas = {
      feature_dict['name']: _from_feature_dict(feature_dict)
      for feature_dict in schema_dict.get('feature', [])
  }
  sparse_feature_column_schemas = {
      sparse_feature_dict['name']: _from_sparse_feature_dict(
          sparse_feature_dict)
      for sparse_feature_dict in schema_dict.get('sparseFeature', [])
  }
  overlapping_keys = set(six.iterkeys(feature_column_schemas)).intersection(
      six.iterkeys(sparse_feature_column_schemas))
  if overlapping_keys:
    raise ValueError('Keys of dense and sparse features overlapped. '
                     'overlapping keys: %s' % overlapping_keys)
  feature_column_schemas.update(sparse_feature_column_schemas)
  return sch.Schema(feature_column_schemas)


def _from_feature_dict(feature_dict):
  """Translate a JSON feature dict into a `ColumnSchema`."""
  domain = _from_domain_dict(feature_dict['domain'])

  axes = []
  if 'fixedShape' in feature_dict:
    for axis in feature_dict['fixedShape'].get('axis', []):
      # int() is needed because protobuf JSON encodes int64 as string
      axes.append(sch.Axis(int(axis.get('size'))))
  elif 'valueCount' in feature_dict:
    # Value_count always means a 1-D feature of unknown size.
    # We don't support value_count.min and value_count.max yet.
    axes.append(sch.Axis(None))

  tf_options = feature_dict['parsingOptions']['tfOptions']
  if tf_options.get('fixedLenFeature') is not None:
    default_value = None
    try:
      # int() is needed because protobuf JSON encodes int64 as string
      default_value = int(tf_options['fixedLenFeature']['intDefaultValue'])
    except KeyError:
      try:
        default_value = tf_options['fixedLenFeature']['stringDefaultValue']
      except KeyError:
        try:
          default_value = tf_options['fixedLenFeature']['floatDefaultValue']
        except KeyError:
          pass
    representation = sch.FixedColumnRepresentation(default_value)
  elif tf_options.get('varLenFeature') is not None:
    representation = sch.ListColumnRepresentation()
  else:
    raise ValueError('Could not interpret tfOptions: {}'.format(tf_options))

  return sch.ColumnSchema(domain, axes, representation)


def _from_sparse_feature_dict(feature_dict):
  """Translate a JSON sparse feature dict into a ColumnSchema."""
  # assume there is only one value column
  value_feature = feature_dict['valueFeature'][0]
  domain = _from_domain_dict(value_feature['domain'])

  index_feature_dicts = feature_dict['indexFeature']

  # int() is needed because protobuf JSON encodes int64 as string
  axes = [sch.Axis(int(index_feature_dict['size']))
          for index_feature_dict in index_feature_dicts]

  value_field_name = value_feature['name']
  index_fields = [sch.SparseIndexField(index_feature_dict['name'],
                                       index_feature_dict['isSorted'])
                  for index_feature_dict in index_feature_dicts]

  representation = sch.SparseColumnRepresentation(value_field_name,
                                                  index_fields)

  return sch.ColumnSchema(domain, axes, representation)


def _from_domain_dict(domain):
  """Translate a JSON domain dict into a Domain."""
  if domain.get('ints') is not None:
    def maybe_to_int(s):
      return int(s) if s is not None else None
    return sch.IntDomain(
        tf.int64,
        maybe_to_int(domain['ints'].get('min')),
        maybe_to_int(domain['ints'].get('max')),
        domain['ints'].get('isCategorical'),
        domain['ints'].get('vocabularyFile', ''))
  if domain.get('floats') is not None:
    return sch.FloatDomain(tf.float32)
  if domain.get('strings') is not None:
    return sch.StringDomain(tf.string)
  if domain.get('bools') is not None:
    return sch.BoolDomain(tf.bool)
  raise ValueError('Unknown domain: {}'.format(domain))
