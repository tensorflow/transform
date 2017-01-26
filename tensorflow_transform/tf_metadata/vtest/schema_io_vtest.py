"""Serialization strategy mapping `Schema` to v1 protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import version_api


class SchemaIOvTest(version_api.SchemaIO):

  def write(self, schema, path):
    with open(path + ".test", "w") as f:
      f.write("\n".join(schema.features.keys()))

  def read(self, path):
    with open(path + ".test") as f:
      all_feature_names = f.read().splitlines()
    return TestSchema(all_feature_names)


class TestSchema(dataset_schema.Schema):

  def __init__(self, feature_names):
    self._features = {feature_name: "Bogus FeatureSchema for %s" % feature_name
                      for feature_name in feature_names}

