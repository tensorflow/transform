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

import json
import os


import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


def read_metadata(path):
  """Load metadata in JSON format from a path into a new DatasetMetadata."""
  schema_file = os.path.join(path, 'schema.pbtxt')
  legacy_schema_file = os.path.join(path, 'v1-json', 'schema.json')
  if file_io.file_exists(schema_file):
    text_proto = file_io.FileIO(schema_file, 'r').read()
    schema_proto = text_format.Parse(text_proto, schema_pb2.Schema(),
                                     allow_unknown_extension=True)
  elif file_io.file_exists(legacy_schema_file):
    schema_json = file_io.FileIO(legacy_schema_file, 'r').read()
    schema_proto = _parse_schema_json(schema_json)
  else:
    raise IOError(
        'Schema file {} does not exist and neither did legacy format file '
        '{}'.format(schema_file, legacy_schema_file))
  return dataset_metadata.DatasetMetadata(schema_proto)


def _parse_schema_json(schema_json):
  """Translate a JSON schema into a Schema proto."""
  schema_dict = json.loads(schema_json)
  feature_spec = {
      feature_dict['name']: _column_schema_from_json(feature_dict)
      for feature_dict in schema_dict.get('feature', [])
  }
  domains = {
      feature_dict['name']: _domain_from_json(feature_dict['domain'])
      for feature_dict in schema_dict.get('feature', [])
  }
  return schema_utils.schema_from_feature_spec(feature_spec, domains)


def _column_schema_from_json(feature_dict):
  """Translate a JSON feature dict into a feature spec."""
  dtype = _dtype_from_json(feature_dict['domain'])
  tf_options = feature_dict['parsingOptions']['tfOptions']
  if tf_options.get('fixedLenFeature') is not None:
    default_value = None
    try:
      # int() is needed because protobuf JSON encodes int64 as string
      default_value = _convert_scalar_or_list(
          int, tf_options['fixedLenFeature']['intDefaultValue'])
    except KeyError:
      try:
        default_value = tf_options['fixedLenFeature']['stringDefaultValue']
      except KeyError:
        try:
          default_value = tf_options['fixedLenFeature']['floatDefaultValue']
        except KeyError:
          pass
    axes = feature_dict['fixedShape'].get('axis', [])
    shape = [int(axis['size']) for axis in axes]
    return tf.io.FixedLenFeature(shape, dtype, default_value)
  elif tf_options.get('varLenFeature') is not None:
    return tf.io.VarLenFeature(dtype)
  else:
    raise ValueError('Could not interpret tfOptions: {}'.format(tf_options))


def _domain_from_json(domain):
  """Translate a JSON domain dict into an IntDomain or None."""
  if domain.get('ints') is not None:
    def maybe_to_int(s):
      return int(s) if s is not None else None
    return schema_pb2.IntDomain(
        min=maybe_to_int(domain['ints'].get('min')),
        max=maybe_to_int(domain['ints'].get('max')),
        is_categorical=domain['ints'].get('isCategorical'))
  return None


def _dtype_from_json(domain):
  """Translate a JSON domain dict into a tf.DType."""
  if domain.get('ints') is not None:
    return tf.int64
  if domain.get('floats') is not None:
    return tf.float32
  if domain.get('strings') is not None:
    return tf.string
  raise ValueError('Unknown domain: {}'.format(domain))


def write_metadata(metadata, path):
  """Write metadata to given path, in JSON format.

  Args:
    metadata: A `DatasetMetadata` to write.
    path: a path to a directory where metadata should be written.
  """
  if not file_io.file_exists(path):
    file_io.recursive_create_dir(path)
  schema_file = os.path.join(path, 'schema.pbtxt')
  ascii_proto = text_format.MessageToString(metadata.schema)
  file_io.atomic_write_string_to_file(schema_file, ascii_proto, overwrite=True)


def _convert_scalar_or_list(fn, scalar_or_list):
  if isinstance(scalar_or_list, list):
    return list(map(fn, scalar_or_list))
  else:
    return fn(scalar_or_list)
