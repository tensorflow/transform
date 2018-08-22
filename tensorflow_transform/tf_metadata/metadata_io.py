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
"""Utilities to read and write metadata in standardized versioned formats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os


import six
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema as sch

from tensorflow.python.lib.io import file_io


def read_metadata(path):
  """Load metadata in JSON format from a path into a new DatasetMetadata."""
  # Ensure that the Schema file exists
  schema_file = os.path.join(path, 'v1-json', 'schema.json')
  if not file_io.file_exists(schema_file):
    raise IOError('Schema file {} does not exist'.format(schema_file))

  file_content = file_io.FileIO(schema_file, 'r').read()
  return dataset_metadata.DatasetMetadata(_schema_from_json(file_content))


def _schema_from_json(schema_json):
  """Translate a JSON schema into a `Schema`."""
  schema_dict = json.loads(schema_json)
  feature_column_schemas = {
      feature_dict['name']: _column_schema_from_json(feature_dict)
      for feature_dict in schema_dict.get('feature', [])
  }
  sparse_feature_column_schemas = {
      sparse_feature_dict['name']: _sparse_column_schema_from_json(
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


def _column_schema_from_json(feature_dict):
  """Translate a JSON feature dict into a `ColumnSchema`."""
  domain = _domain_from_json(feature_dict['domain'])

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


def _sparse_column_schema_from_json(feature_dict):
  """Translate a JSON sparse feature dict into a ColumnSchema."""
  # assume there is only one value column
  value_feature = feature_dict['valueFeature'][0]
  domain = _domain_from_json(value_feature['domain'])

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


def _domain_from_json(domain):
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


def write_metadata(metadata, path):
  """Write metadata to given path, in JSON format.

  Args:
    metadata: A `DatasetMetadata` to write.
    path: a path to a directory where metadata should be written.
  """
  schema_dir = os.path.join(path, 'v1-json')
  if not file_io.file_exists(schema_dir):
    file_io.recursive_create_dir(schema_dir)
  schema_file = os.path.join(schema_dir, 'schema.json')

  schema_as_json = _schema_to_json(metadata.schema)
  file_io.write_string_to_file(schema_file, schema_as_json)


_FEATURE_TYPE_INT = 'INT'
_FEATURE_TYPE_FLOAT = 'FLOAT'
_FEATURE_TYPE_BYTES = 'BYTES'


def _schema_to_json(schema):
  """Converts in-memory `Schema` representation to JSON."""
  features = []
  sparse_features = []
  for name, column_schema in sorted(six.iteritems(schema.column_schemas)):
    if isinstance(column_schema.representation,
                  sch.SparseColumnRepresentation):
      sparse_features.append(_sparse_column_schema_to_json(name, column_schema))
    else:
      features.append(_dense_column_schema_to_json(name, column_schema))
  schema_dict = {
      'feature': features,
      'sparseFeature': sparse_features
  }
  return json.dumps(schema_dict, indent=2, separators=(',', ': '),
                    sort_keys=True)


def _dense_column_schema_to_json(name, column_schema):
  """Translate a ColumnSchema for a dense column into JSON feature dict."""
  representation = column_schema.representation

  result = {}
  result['name'] = name
  # Note result['deprecated'] is not populated in v1.
  # Note result['comment'] is not populated in v1.
  # Note result['presence'] is not populated in v1.

  if column_schema.is_fixed_size():
    axes = []
    for axis in column_schema.axes:
      # str() is needed to match protobuf JSON encoding of int64 as string
      axes.append({'size': str(axis.size)})
    result['fixedShape'] = {'axis': axes}
  else:
    # This is a 1-d variable length feature.  We don't track max and min, so
    # just provide an empty value_count.
    result['valueCount'] = {}

  result['type'] = _feature_type_to_json(column_schema.domain.dtype)
  result['domain'] = _domain_to_json(column_schema.domain)

  tf_options = _representation_to_json(representation, result['type'])
  result['parsingOptions'] = {'tfOptions': tf_options}

  return result


def _sparse_column_schema_to_json(name, column_schema):
  """Translate a ColumnSchema for a sparse column into JSON feature dict."""
  representation = column_schema.representation

  result = {}
  result['name'] = name
  # Note result['deprecated'] is not populated in v1.
  # Note result['comment'] is not populated in v1.
  # Note result['presence'] is not populated in v1.

  index_feature_list = []
  # Note axes and index_fields must be in the same order.
  for (axis, index_field) in zip(
      column_schema.axes, representation.index_fields):

    # str() is needed to match protobuf JSON encoding of int64 as string
    index_feature_list.append({'name': index_field.name,
                               'size': str(axis.size),
                               'isSorted': index_field.is_sorted})

  result['indexFeature'] = index_feature_list
  result['valueFeature'] = [{'name': representation.value_field_name,
                             'type': _feature_type_to_json(
                                 column_schema.domain.dtype),
                             'domain': _domain_to_json(column_schema.domain)}]

  return result


def _feature_type_to_json(dtype):
  if dtype.is_integer:
    return _FEATURE_TYPE_INT
  if dtype.is_floating:
    return _FEATURE_TYPE_FLOAT
  if dtype == tf.string:
    return _FEATURE_TYPE_BYTES
  if dtype == tf.bool:
    return _FEATURE_TYPE_INT
  return 'TYPE_UNKNOWN'


def _domain_to_json(domain):
  """Translates a Domain object into a JSON dict."""
  result = {}
  # Domain names and bounds are not populated yet
  if isinstance(domain, sch.IntDomain):
    result['ints'] = {
        'min': str(domain.min_value),
        'max': str(domain.max_value),
        'isCategorical': domain.is_categorical,
        'vocabularyFile': domain.vocabulary_file
    }
  elif isinstance(domain, sch.FloatDomain):
    result['floats'] = {}
  elif isinstance(domain, sch.StringDomain):
    result['strings'] = {}
  elif isinstance(domain, sch.BoolDomain):
    result['bools'] = {}
  return result


def _representation_to_json(representation, type_string):
  """Translate a ColumnRepresentation into JSON format."""
  tf_options = {}
  if isinstance(representation, sch.FixedColumnRepresentation):
    if representation.default_value is None:
      fixed_len_options = {}
    else:
      if type_string == 'BYTES':
        fixed_len_options = {'stringDefaultValue':
                             representation.default_value}
      elif type_string == 'INT':
        int_default = int(representation.default_value)
        # str() is needed to match protobuf JSON encoding of int64 as string
        fixed_len_options = {'intDefaultValue': str(int_default)}
      elif type_string == 'FLOAT':
        fixed_len_options = {'floatDefaultValue':
                             representation.default_value}
      else:
        raise ValueError("v1 Schema can't represent default value {} "
                         "for type {}".format(
                             representation.default_value, type_string))
    tf_options['fixedLenFeature'] = fixed_len_options
    return tf_options

  if isinstance(representation, sch.ListColumnRepresentation):
    tf_options['varLenFeature'] = {}
    return tf_options

  raise TypeError('Cannot represent {} using the Feature representation; '
                  'the SparseFeature representation should have been '
                  'chosen.'.format(representation))
