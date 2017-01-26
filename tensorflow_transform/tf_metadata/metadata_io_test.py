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


_VTEST = version_api.MetadataVersion('vTest', schema_io_vtest.SchemaIOvTest(),
                                     None, None, None, None)
_test_versions = {'test': _VTEST}


class DatasetMetadataTest(unittest.TestCase):

  def test_write_and_read(self):
    basedir = tempfile.mkdtemp()
    original_schema = schema_io_vtest.TestSchema(
        ['test_feature_1', 'test_feature_2'])
    original = dataset_metadata.DatasetMetadata(schema=original_schema)

    print(original.schema.features)

    metadata_io.write_metadata(original, basedir, versions=_test_versions)
    reloaded = metadata_io.read_metadata(basedir, versions=_test_versions)

    print(reloaded.schema.features)

    self.assertTrue('test_feature_1' in reloaded.schema.features)
    self.assertTrue('test_feature_2' in reloaded.schema.features)
    self.assertEqual(2, len(reloaded.schema.features))


if __name__ == '__main__':
  unittest.main()
