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
"""Tests for tensorflow_transform.analyzer_cache."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow_transform import analyzer_cache
import unittest


class AnalyzerCacheTest(unittest.TestCase):

  def test_validate_dataset_keys(self):
    analyzer_cache.validate_dataset_keys(
        {'foo', 'Foo', 'A1', 'A_1', 'A.1', 'A-1'})

    for key in {'foo 1', 'foo@1', 'foo*', 'foo[]', 'foo/goo'}:
      with self.assertRaisesRegexp(
          ValueError, 'Dataset key .* does not match allowed pattern:'):
        analyzer_cache.validate_dataset_keys({key})

  def test_make_cache_file_path(self):
    dataset_key = 'foo-1'
    cache_key = 'cache-3'
    path = analyzer_cache.make_cache_file_path(dataset_key, cache_key)
    self.assertIn(dataset_key, path)
    self.assertIn(cache_key, path)
    self.assertIn(analyzer_cache._CACHE_VERSION, path)


if __name__ == '__main__':
  unittest.main()
