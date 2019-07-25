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

# GOOGLE-INITIALIZATION

import apache_beam as beam
from apache_beam.internal import pickler
import six
import tensorflow as tf

# This should be advanced whenever a non-backwards compatible change is made
# that affects analyzer cache. For example, changing accumulator format.
_CACHE_VERSION = b'__v0__'


def _get_dataset_cache_path(base_dir, dataset_key):
  return os.path.join(base_dir, make_dataset_key(dataset_key))


class _ManifestFile(object):
  """A manifest file wrapper used to read and write tft cache manifest files."""

  # TODO(b/37788560): Use artifacts instead.
  _MANIFEST_FILE_NAME = 'MANIFEST'

  def __init__(self, base_path):
    self._base_path = base_path
    self._manifest_path = os.path.join(base_path, self._MANIFEST_FILE_NAME)
    self._file = None

  def __enter__(self):
    if not tf.io.gfile.isdir(self._base_path):
      tf.io.gfile.makedirs(self._base_path)
    self._file = tf.io.gfile.GFile(self._manifest_path, 'wb+')
    return self

  def __exit__(self, *exn_info):
    self._file.close()
    self._file = None

  def _get_manifest_contents(self, manifest_file_handle):
    manifest_file_handle.seek(0)
    return pickler.loads(manifest_file_handle.read())

  def read(self):
    if not tf.io.gfile.exists(self._manifest_path):
      return {}

    if self._file is not None:
      return self._get_manifest_contents(self._file)
    else:
      with tf.io.gfile.GFile(self._manifest_path, 'rb') as f:
        return self._get_manifest_contents(f)

  def write(self, manifest):
    assert self._file is not None
    try:
      self._file.seek(0)
    except tf.errors.NotFoundError:
      pass
    self._file.write(pickler.dumps(manifest))


class WriteAnalysisCacheToFS(beam.PTransform):
  """Writes a cache object that can be read by ReadAnalysisCacheFromFS.

  Given a cache collection, this writes it to the configured directory.
  If the configured directory already contains cache, this will merge the new
  cache with the old.
  NOTE: This merging of cache is determined at beam graph construction time,
  so the cache must already exist there when constructing this.
  """

  def __init__(self, pipeline, cache_base_dir, sink=None):
    """Init method.

    Args:
      pipeline: A beam Pipeline.
      cache_base_dir: A str, the path that the cache should be stored in.
      sink: (Optional) A PTransform class that takes a path, and optional
        file_name_suffix arguments in its constructor, and is used to write the
        cache.
    """
    self.pipeline = pipeline
    self._cache_base_dir = cache_base_dir
    # TODO(b/37788560): Possibly use Riegeli as a default file format once
    # possible.
    self._sink = sink if sink is not None else beam.io.WriteToTFRecord

  def _write_cache(self, manifest_file, dataset_key, dataset_key_dir,
                   cache_dict):
    manifest = manifest_file.read()
    start_cache_idx = max(manifest.values()) + 1 if manifest else 0

    cache_is_written = []
    for cache_key_idx, (cache_entry_key, cache_pcoll) in enumerate(
        six.iteritems(cache_dict), start_cache_idx):
      path = os.path.join(dataset_key_dir, str(cache_key_idx))
      manifest[cache_entry_key] = cache_key_idx
      cache_is_written.append(
          cache_pcoll
          | 'WriteCache[{}][{}]'.format(dataset_key, cache_key_idx) >>
          self._sink(path, file_name_suffix='.gz'))

    manifest_file.write(manifest)
    return cache_is_written

  def expand(self, dataset_cache_dict):

    cache_is_written = []
    for dataset_key, cache_dict in six.iteritems(dataset_cache_dict):
      dataset_key_dir = _get_dataset_cache_path(self._cache_base_dir,
                                                dataset_key)

      with _ManifestFile(dataset_key_dir) as manifest_file:
        cache_is_written.extend(
            self._write_cache(manifest_file, dataset_key, dataset_key_dir,
                              cache_dict))

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

      dataset_cache_path = _get_dataset_cache_path(self._cache_base_dir,
                                                   dataset_key)
      manifest_file = _ManifestFile(dataset_cache_path)
      manifest = manifest_file.read()
      if not manifest:
        continue
      cache_dict[dataset_key] = {}
      for key, value in six.iteritems(manifest):
        cache_dict[dataset_key][key] = (
            pvalue.pipeline
            | 'ReadCache[{}][{}]'.format(dataset_key, value) >>
            self._source('{}{}'.format(
                os.path.join(dataset_cache_path, str(value)), '-*-of-*')))
    return cache_dict


def validate_dataset_keys(keys):
  regex = re.compile(r'^[a-zA-Z0-9\.\-_]+$')
  for key in keys:
    if not regex.match(key):
      raise ValueError(
          'Dataset key {!r} does not match allowed pattern: {!r}'.format(
              key, regex.pattern))


def make_cache_entry_key(cache_key):
  return _CACHE_VERSION + tf.compat.as_bytes(cache_key)


def make_dataset_key(dataset_key):
  return _make_valid_cache_component(dataset_key)


def _make_valid_cache_component(name):
  return name.replace('/', '-').replace('*', 'STAR').replace('@', 'AT').replace(
      '[', '--').replace(']', '--').replace(':', 'P').replace('=', 'E')
