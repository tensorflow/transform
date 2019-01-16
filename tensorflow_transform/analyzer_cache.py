# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Module which allows a pipeilne to define and utilize cached analyzers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re


# This should be advanced whenever a non-backwards compatible change is made
# that affects analyzer cache. For example, changing accumulator format.
_CACHE_VERSION = '__v0__'


def validate_dataset_keys(keys):
  regex = re.compile(r'^[a-zA-Z0-9\.\-_]+$')
  for key in keys:
    if not regex.match(key):
      raise ValueError(
          'Dataset key {!r} does not match allowed pattern: {!r}'.format(
              key, regex.pattern))


def make_cache_file_path(dataset_key, cache_key):
  return os.path.join(
      _make_valid_cache_component(dataset_key), '{}{}'.format(
          _CACHE_VERSION, _make_valid_cache_component(cache_key)))


def _make_valid_cache_component(name):
  return name.replace('/', '-').replace('*', 'STAR').replace('@', 'AT').replace(
      '[', '--').replace(']', '--')
