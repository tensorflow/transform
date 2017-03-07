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

import tempfile


from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import version_api
from tensorflow_transform.tf_metadata.vtest import schema_io_vtest
import unittest


_VTEST = version_api.MetadataVersion('vTest', None,
                                     schema_io_vtest.SchemaIOvTest(),
                                     None, None, None, None)
_test_versions = {'test': _VTEST}.items()  # make immutable


class DatasetMetadataTest(unittest.TestCase):

  def test_write_and_read(self):
    basedir = tempfile.mkdtemp()
    original_schema = schema_io_vtest.TestSchema(
        {'test_feature_1': 'bogus 1', 'test_feature_2': 'bogus 2'})
    original = dataset_metadata.DatasetMetadata(schema=original_schema)

    metadata_io.write_metadata(original, basedir, versions=_test_versions)
    reloaded = metadata_io.read_metadata(basedir, versions=_test_versions)

    self.assertTrue('test_feature_1' in reloaded.schema.column_schemas)
    self.assertTrue('test_feature_2' in reloaded.schema.column_schemas)
    self.assertEqual(2, len(reloaded.schema.column_schemas))


if __name__ == '__main__':
  unittest.main()
