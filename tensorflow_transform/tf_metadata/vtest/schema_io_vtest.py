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
"""Serialization strategy mapping `Schema` to v1 protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import version_api


class SchemaIOvTest(version_api.SchemaIO):

  def write(self, schema, path):
    with open(path + ".test", "w") as f:
      f.write("\n".join(schema.column_schemas.keys()))

  def read(self, path):
    with open(path + ".test") as f:
      all_feature_names = f.read().splitlines()
    return TestSchema(all_feature_names)


class TestSchema(dataset_schema.Schema):

  def __init__(self, feature_names):
    features = {feature_name: "Bogus FeatureSchema for %s" % feature_name
                for feature_name in feature_names}
    super(TestSchema, self).__init__(features)

