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
"""Tests for dataset_metadata.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile


from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import test_common
import unittest

from tensorflow.python.lib.io import file_io


_SCHEMA_WITH_INVALID_KEYS = """
{
  "feature": [{
    "name": "my_key",
    "fixedShape": {
      "axis": []
    },
    "type": "INT",
    "domain": {
      "ints": {}
    },
    "parsingOptions": {
      "tfOptions": {
        "fixedLenFeature": {}
      }
    }
  }],
  "sparseFeature": [{
    "name": "my_key",
    "indexFeature": [],
    "valueFeature": [{
      "name": "value_key",
      "type": "INT",
      "domain": {
        "ints": {}
      }
    }]
  }]
}
"""


class SchemaIOv1JsonTest(unittest.TestCase):

  def _write_schema_to_disk(self, basedir, schema_string):
    version_basedir = os.path.join(basedir, 'v1-json')

    # Write a proto by hand to disk
    file_io.recursive_create_dir(version_basedir)
    file_io.write_string_to_file(os.path.join(version_basedir, 'schema.json'),
                                 schema_string)

  def test_read_with_invalid_keys(self):
    basedir = tempfile.mkdtemp()
    self._write_schema_to_disk(basedir, _SCHEMA_WITH_INVALID_KEYS)

  def test_read_features_default_axis(self):
    basedir = tempfile.mkdtemp()
    schema_no_sparse_features = """
    {
      "feature": [{
        "name": "my_key",
        "fixedShape": {},
        "type": "INT",
        "domain": {
          "ints": {}
        },
        "parsingOptions": {
          "tfOptions": {
            "fixedLenFeature": {}
          }
        }
      }]
    }
    """
    self._write_schema_to_disk(basedir, schema_no_sparse_features)
    _ = metadata_io.read_metadata(basedir)

  def test_read_features(self):
    basedir = tempfile.mkdtemp()
    schema_no_sparse_features = """
    {
      "feature": [{
        "name": "my_key",
        "fixedShape": {
          "axis": [{
            "size": 2
          }]
        },
        "type": "INT",
        "domain": {
          "ints": {}
        },
        "parsingOptions": {
          "tfOptions": {
            "fixedLenFeature": {}
          }
        }
      }]
    }
    """
    self._write_schema_to_disk(basedir, schema_no_sparse_features)
    _ = metadata_io.read_metadata(basedir)

  def test_read_no_features(self):
    basedir = tempfile.mkdtemp()
    schema_no_features = """
    {
      "sparseFeature": [{
        "name": "my_key",
        "indexFeature": [],
        "valueFeature": [{
          "name": "value_key",
          "type": "INT",
          "domain": {
            "ints": {}
          }
        }]
      }]
    }
    """
    self._write_schema_to_disk(basedir, schema_no_features)
    _ = metadata_io.read_metadata(basedir)

  def test_write_and_read(self):
    basedir = tempfile.mkdtemp()
    original = dataset_metadata.DatasetMetadata(
        schema=test_common.get_test_schema())

    metadata_io.write_metadata(original, basedir)
    reloaded = metadata_io.read_metadata(basedir)

    generated_feature_spec = reloaded.schema.as_feature_spec()
    self.assertEqual(test_common.test_feature_spec, generated_feature_spec)


if __name__ == '__main__':
  unittest.main()
