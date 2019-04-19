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

import json
import os
import re

# GOOGLE-INITIALIZATION

import apache_beam as beam
import six
import tensorflow as tf

# This should be advanced whenever a non-backwards compatible change is made
# that affects analyzer cache. For example, changing accumulator format.
_CACHE_VERSION = '__v0__'


# TODO(b/37788560): Use artifacts instead.
_MANIFEST_FILE_NAME = 'MANIFEST.txt'


class WriteAnalysisCacheToFS(beam.PTransform):
  """Writes a cache object that can be read by ReadAnalysisCacheFromFS."""

  def __init__(self, cache_base_dir, sink=None):
    """Init method.

    Args:
      cache_base_dir: A str, the path that the cache should be stored in.
      sink: (Optional) A PTransform class that takes a path, and optional
        file_name_suffix arguments in its constructor, and is used to write the
        cache.
    """
    self._cache_base_dir = cache_base_dir
    # TODO(b/37788560): Possibly use Riegeli as a default file format once
    # possible.
    self._sink = sink if sink is not None else beam.io.WriteToTFRecord

  def expand(self, dataset_cache_dict):

    cache_is_written = []
    for dataset_key, cache_dict in six.iteritems(dataset_cache_dict):
      manifest = {}
      dataset_key_dir = os.path.join(self._cache_base_dir,
                                     _make_dataset_key(dataset_key))
      for cache_entry_key, cache_pcoll in six.iteritems(cache_dict):
        cache_entry = _make_valid_cache_component(cache_entry_key)
        path = os.path.join(dataset_key_dir, cache_entry)
        manifest[cache_entry] = cache_entry
        cache_is_written.append(
            cache_pcoll
            | 'WriteCache[{}][{}]'.format(dataset_key, cache_entry_key) >>
            self._sink(path, file_name_suffix='.gz'))

      if not tf.io.gfile.isdir(dataset_key_dir):
        tf.io.gfile.makedirs(dataset_key_dir)
      with tf.io.gfile.GFile(
          os.path.join(dataset_key_dir, _MANIFEST_FILE_NAME), 'w') as f:
        f.write(json.dumps(manifest))

    return cache_is_written


class ReadAnalysisCacheFromFS(beam.PTransform):
  """Reads cache from the FS written by WriteAnalysisCacheToFS."""

  def __init__(self, cache_base_dir, dataset_keys, source=None):
    """Init method.

    Args:
      cache_base_dir: A string, the path that the cache should be stored in.
      dataset_keys: An iterable of strings.
      source: (Optional) A PTransform class that takes a path argument in its
        constructor, and is used to read the cache.
    """
    self._cache_base_dir = cache_base_dir
    self._dataset_keys = dataset_keys
    # TODO(b/37788560): Possibly use Riegeli as a default file format once
    # possible.
    self._source = source if source is not None else beam.io.ReadFromTFRecord

  def expand(self, pvalue):
    cache_dict = {}

    for dataset_key in self._dataset_keys:

      dataset_cache_path = os.path.join(self._cache_base_dir,
                                        _make_dataset_key(dataset_key))
      if not tf.io.gfile.isdir(dataset_cache_path):
        continue
      cache_dict[dataset_key] = {}
      with tf.io.gfile.GFile(
          os.path.join(dataset_cache_path, _MANIFEST_FILE_NAME), 'r') as f:
        manifest = json.loads(f.read())
      for key, value in six.iteritems(manifest):
        cache_dict[dataset_key][key] = (
            pvalue.pipeline
            | 'ReadCache[{}][{}]'.format(dataset_key, key) >>
            self._source('{}{}'.format(
                os.path.join(dataset_cache_path, value), '-*-of-*')))
    return cache_dict


def validate_dataset_keys(keys):
  regex = re.compile(r'^[a-zA-Z0-9\.\-_]+$')
  for key in keys:
    if not regex.match(key):
      raise ValueError(
          'Dataset key {!r} does not match allowed pattern: {!r}'.format(
              key, regex.pattern))


def make_cache_entry_key(cache_key):
  return '{}{}'.format(_CACHE_VERSION, _make_valid_cache_component(cache_key))


def _make_dataset_key(dataset_key):
  return _make_valid_cache_component(dataset_key)


def _make_valid_cache_component(name):
  return name.replace('/', '-').replace('*', 'STAR').replace('@', 'AT').replace(
      '[', '--').replace(']', '--')
