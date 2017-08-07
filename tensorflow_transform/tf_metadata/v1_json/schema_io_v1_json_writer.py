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
"""Writer for `Schema` to v1 JSON."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import six
import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_schema


_FEATURE_TYPE_INT = 'INT'
_FEATURE_TYPE_FLOAT = 'FLOAT'
_FEATURE_TYPE_BYTES = 'BYTES'


def to_schema_json(schema):
  """Converts in-memory `Schema` representation to v1 Schema JSON."""

  result = {'feature': _get_features(schema),
            'sparseFeature': _get_sparse_features(schema)}

  # not populated yet: Schema.string_domain

  return json.dumps(result, indent=2, separators=(',', ': '), sort_keys=True)


def _get_features(schema):
  result = []
  for name, column_schema in sorted(six.iteritems(schema.column_schemas)):
    if not isinstance(column_schema.representation,
                      dataset_schema.SparseColumnRepresentation):
      result.append(_column_schema_to_dict_dense(name, column_schema))
  return result


def _get_sparse_features(schema):
  result = []
  for name, column_schema in sorted(six.iteritems(schema.column_schemas)):
    if isinstance(column_schema.representation,
                  dataset_schema.SparseColumnRepresentation):
      result.append(_column_schema_to_dict_sparse(name, column_schema))
  return result


def _column_schema_to_dict_dense(name, column_schema):
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

  result['type'] = _to_feature_type_enum(column_schema.domain.dtype)
  result['domain'] = _to_domain(column_schema.domain)

  tf_options = _get_tf_options(representation, result['type'])
  result['parsingOptions'] = {'tfOptions': tf_options}

  return result


def _column_schema_to_dict_sparse(name, column_schema):
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
                             'type': _to_feature_type_enum(
                                 column_schema.domain.dtype),
                             'domain': _to_domain(column_schema.domain)}]

  return result


def _to_feature_type_enum(dtype):
  if dtype.is_integer:
    return _FEATURE_TYPE_INT
  if dtype.is_floating:
    return _FEATURE_TYPE_FLOAT
  if dtype == tf.string:
    return _FEATURE_TYPE_BYTES
  if dtype == tf.bool:
    return _FEATURE_TYPE_INT
  return 'TYPE_UNKNOWN'


def _to_domain(domain):
  """Translates a Domain object into a JSON dict."""
  result = {}
  # Domain names and bounds are not populated yet
  if isinstance(domain, dataset_schema.IntDomain):
    result['ints'] = {
        'min': str(domain.min_value),
        'max': str(domain.max_value),
        'isCategorical': domain.is_categorical,
        'vocabularyFile': domain.vocabulary_file
    }
  elif isinstance(domain, dataset_schema.FloatDomain):
    result['floats'] = {}
  elif isinstance(domain, dataset_schema.StringDomain):
    result['strings'] = {}
  elif isinstance(domain, dataset_schema.BoolDomain):
    result['bools'] = {}
  return result


def _get_tf_options(representation, type_string):
  """Translate a ColumnRepresentation into JSON string for tf_options."""
  tf_options = {}
  if isinstance(representation, dataset_schema.FixedColumnRepresentation):
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

  if isinstance(representation, dataset_schema.ListColumnRepresentation):
    tf_options['varLenFeature'] = {}
    return tf_options

  raise TypeError('Cannot represent {} using the Feature representation; '
                  'the SparseFeature representation should have been '
                  'chosen.'.format(representation))

