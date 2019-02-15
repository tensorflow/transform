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
"""Tests for tensorflow_transform.tf_metadata.schema_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION

from absl.testing import parameterized
from tensorflow_transform.tf_metadata import schema_utils_legacy
from tensorflow_transform.tf_metadata import schema_utils_test_cases
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
import unittest
from tensorflow_metadata.proto.v0 import schema_pb2


class SchemaUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *schema_utils_test_cases.EQUIVALENT_FEATURE_SPEC_AND_SCHEMAS)
  def test_schema_from_feature_spec(
      self, ascii_proto, feature_spec, domains=None,
      generate_legacy_feature_spec=False):
    expected_schema_proto = text_format.Parse(ascii_proto, schema_pb2.Schema())
    schema_utils_legacy.set_generate_legacy_feature_spec(
        expected_schema_proto, generate_legacy_feature_spec)
    self.assertEqual(
        schema_utils.schema_from_feature_spec(feature_spec, domains),
        expected_schema_proto)

  @parameterized.named_parameters(
      *(schema_utils_test_cases.EQUIVALENT_FEATURE_SPEC_AND_SCHEMAS +
        schema_utils_test_cases.NON_ROUNDTRIP_SCHEMAS))
  def test_schema_as_feature_spec(
      self, ascii_proto, feature_spec, domains=None,
      generate_legacy_feature_spec=False):
    schema_proto = text_format.Parse(ascii_proto, schema_pb2.Schema())
    schema_utils_legacy.set_generate_legacy_feature_spec(
        schema_proto, generate_legacy_feature_spec)
    self.assertEqual(
        schema_utils.schema_as_feature_spec(schema_proto),
        (feature_spec, domains or {}))

  @parameterized.named_parameters(
      *schema_utils_test_cases.INVALID_SCHEMA_PROTOS)
  def test_schema_as_feature_spec_fails(
      self, ascii_proto, error_msg, error_class=ValueError,
      generate_legacy_feature_spec=False):
    schema_proto = text_format.Parse(ascii_proto, schema_pb2.Schema())
    schema_utils_legacy.set_generate_legacy_feature_spec(
        schema_proto, generate_legacy_feature_spec)
    with self.assertRaisesRegexp(error_class, error_msg):
      schema_utils.schema_as_feature_spec(schema_proto)

  @parameterized.named_parameters(
      *schema_utils_test_cases.INVALID_FEATURE_SPECS)
  def test_schema_from_feature_spec_fails(
      self, feature_spec, error_msg, domain=None, error_class=ValueError):
    with self.assertRaisesRegexp(error_class, error_msg):
      schema_utils.schema_from_feature_spec(feature_spec, domain)


if __name__ == '__main__':
  unittest.main()
