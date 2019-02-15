# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Stubs for handling legacy fields of the Schema proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def should_set_generate_legacy_feature_spec(feature_spec):
  del feature_spec  # unused
  return False


def set_generate_legacy_feature_spec(schema_proto, value):
  del schema_proto  # unused
  if value:
    raise NotImplementedError(
        'The generate_legacy_feature_spec is a legacy field that is not part '
        'of the OSS tf.Transform codebase')


def get_generate_legacy_feature_spec(schema_proto):
  del schema_proto  # unused
  return False


def check_for_unsupported_features(schema_proto):
  del schema_proto  # unused


def get_deprecated(feature):
  del feature  # unused
  return False
