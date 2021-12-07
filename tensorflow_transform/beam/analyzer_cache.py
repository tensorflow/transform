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

import os
import pickle
import re
import sys

import apache_beam as beam
import tensorflow as tf
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple

# This should be advanced whenever a non-backwards compatible change is made
# that affects analyzer cache. For example, changing accumulator format.
_CACHE_VERSION_NUMBER = 1
_PYTHON_VERSION = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
_CACHE_VERSION = tf.compat.as_bytes('__v{}__{}_'.format(_CACHE_VERSION_NUMBER,
                                                        _PYTHON_VERSION))

_CACHE_COMPONENT_CHARACTER_REPLACEMENTS = (
    ('/', '-'),
    ('\\', '-'),
    ('*', 'STAR'),
    ('@', 'AT'),
    ('[', '--'),
    (']', '--'),
    (':', 'P'),
    ('=', 'E'),
)


def _make_valid_cache_component(name):
  result = name
  for unsupported_char, replacement in _CACHE_COMPONENT_CHARACTER_REPLACEMENTS:
    result = result.replace(unsupported_char, replacement)
  return result


class DatasetKey(tfx_namedtuple.namedtuple('DatasetKey', ['key'])):
  """A key for a dataset used for analysis."""
  __slots__ = ()
  _FLATTENED_DATASET_KEY = object()

  def __new__(cls, dataset_key):
    if dataset_key is not DatasetKey._FLATTENED_DATASET_KEY:
      dataset_key = _make_valid_cache_component(dataset_key)
    return super(DatasetKey, cls).__new__(cls, key=dataset_key)

  def __str__(self):
    if self.is_flattened_dataset_key():
      return str(DatasetKey('FlattenedDataset'))
    else:
      return super().__str__()

  def __hash__(self):
    return hash(self.key)

  def __eq__(self, other):
    if self.key == other:
      return True
    return isinstance(other, DatasetKey) and self.key == other.key

  def is_flattened_dataset_key(self):
    return self.key == self._FLATTENED_DATASET_KEY


def _make_flattened_dataset_key():
  return DatasetKey(DatasetKey._FLATTENED_DATASET_KEY)  # pylint: disable=protected-access


def _get_dataset_cache_path(base_dir, dataset_key):
  return os.path.join(base_dir, dataset_key.key)


class _ManifestFile:
  """A manifest file wrapper used to read and write tft cache manifest files."""

  # TODO(b/37788560): Use artifacts instead.
  _MANIFEST_FILE_NAME = 'MANIFEST'

  def __init__(self, base_path):
    self._base_path = base_path
    self._manifest_path = os.path.join(base_path, self._MANIFEST_FILE_NAME)
    self._file = None

  def _open(self):
    assert self._file is None
    if not tf.io.gfile.isdir(self._base_path):
      tf.io.gfile.makedirs(self._base_path)
    self._file = tf.io.gfile.GFile(self._manifest_path, 'wb+')

  def _close(self):
    if self._file:
      self._file.close()
    self._file = None

  def _delete(self):
    self._close()
    tf.io.gfile.remove(self._manifest_path)

  def __enter__(self):
    self._open()
    return self

  def __exit__(self, *exn_info):
    self._close()

  def _get_manifest_contents(self, manifest_file_handle):
    manifest_file_handle.seek(0)
    try:
      result = pickle.loads(manifest_file_handle.read())
      assert isinstance(result, dict)
      return result
    except Exception as e:  # pylint: disable=broad-except
      # Any exception at this point would be an indication that the cache is
      # likely invalidated. Returning an empty dict allows the pipeline to
      # "gracefully" recover (by proceeding without cache) as opposed to
      # entering a crash-loop it can't recover from.
      tf.compat.v1.logging.error('Can\'t load cache manifest contents: %s',
                                 str(e))
      return {}

  def read(self):
    if not tf.io.gfile.exists(self._manifest_path):
      return {}

    if self._file is not None:
      return self._get_manifest_contents(self._file)
    else:
      with tf.io.gfile.GFile(self._manifest_path, 'rb') as f:
        return self._get_manifest_contents(f)

  def write(self, manifest):
    try:
      # First attempt to delete the manifest if it exists in case it can't be
      # edited in-place.
      self._delete()
    except tf.errors.NotFoundError:
      pass
    self._open()
    # Manifests are small, so writing in a semi-human readable form (protocol=0)
    # is preferred over the efficiency gains of higher protocols.
    self._file.write(pickle.dumps(manifest, protocol=0))


class _WriteToTFRecordGzip(beam.io.WriteToTFRecord):

  def __init__(self, file_path_prefix):
    super().__init__(file_path_prefix, file_name_suffix='.gz')


@beam.typehints.with_input_types(bytes)
class WriteAnalysisCacheToFS(beam.PTransform):
  """Writes a cache object that can be read by ReadAnalysisCacheFromFS.

  Given a cache collection, this writes it to the configured directory.
  If the configured directory already contains cache, this will merge the new
  cache with the old.
  NOTE: This merging of cache is determined at beam graph construction time,
  so the cache must already exist there when constructing this.
  """

  def __init__(self, pipeline, cache_base_dir, dataset_keys=None, sink=None):
    """Init method.

    Args:
      pipeline: A beam Pipeline.
      cache_base_dir: A str, the path that the cache should be stored in.
      dataset_keys: (Optional) An iterable of strings.
      sink: (Optional) A PTransform class that takes a path in its constructor,
        and is used to write the cache. If not provided this uses a GZipped
        TFRecord sink.
    """
    self.pipeline = pipeline
    self._cache_base_dir = cache_base_dir
    if dataset_keys is None:
      self._sorted_dataset_keys = None
    else:
      self._sorted_dataset_keys = sorted(dataset_keys)
    self._sink = sink
    if self._sink is None:
      # TODO(b/37788560): Possibly use Riegeli as a default file format once
      # possible.
      self._sink = _WriteToTFRecordGzip

  def _write_cache(self, manifest_file, dataset_key_index, dataset_key_dir,
                   cache_dict):
    manifest = manifest_file.read()
    start_cache_idx = max(manifest.values()) + 1 if manifest else 0

    cache_is_written = []
    for cache_key_idx, (cache_entry_key,
                        cache_pcoll) in enumerate(cache_dict.items(),
                                                  start_cache_idx):
      path = os.path.join(dataset_key_dir, str(cache_key_idx))
      manifest[cache_entry_key] = cache_key_idx
      cache_is_written.append(
          cache_pcoll
          | 'Write[AnalysisIndex{}][CacheKeyIndex{}]'.format(
              dataset_key_index, cache_key_idx) >> self._sink(path))

    manifest_file.write(manifest)
    return cache_is_written

  def expand(self, dataset_cache_dict):
    if self._sorted_dataset_keys is None:
      sorted_dataset_keys_list = sorted(dataset_cache_dict.keys())
    else:
      sorted_dataset_keys_list = self._sorted_dataset_keys
      missing_keys = set(dataset_cache_dict.keys()).difference(
          set(sorted_dataset_keys_list))
      if missing_keys:
        raise ValueError(
            'The dataset keys in the cache dictionary must be a subset of the '
            'keys in dataset_keys. Missing {}.'.format(missing_keys))
    if not all(isinstance(d, DatasetKey) for d in sorted_dataset_keys_list):
      raise ValueError('Expected dataset_keys to be of type DatasetKey')

    cache_is_written = []
    for dataset_key, cache_dict in dataset_cache_dict.items():
      dataset_key_idx = sorted_dataset_keys_list.index(dataset_key)
      dataset_key_dir = _get_dataset_cache_path(self._cache_base_dir,
                                                dataset_key)

      with _ManifestFile(dataset_key_dir) as manifest_file:
        cache_is_written.extend(
            self._write_cache(manifest_file, dataset_key_idx, dataset_key_dir,
                              cache_dict))

    return cache_is_written


class ReadAnalysisCacheFromFS(beam.PTransform):
  """Reads cache from the FS written by WriteAnalysisCacheToFS."""

  def __init__(self,
               cache_base_dir,
               dataset_keys,
               cache_entry_keys=None,
               source=None):
    """Init method.

    Args:
      cache_base_dir: A string, the path that the cache should be stored in.
      dataset_keys: An iterable of `DatasetKey`s.
      cache_entry_keys: (Optional) An iterable of cache entry key strings. If
        provided, only cache entries that exist in `cache_entry_keys` will be
        read.
      source: (Optional) A PTransform class that takes a path argument in its
        constructor, and is used to read the cache.
    """
    self._cache_base_dir = cache_base_dir
    if not all(isinstance(d, DatasetKey) for d in dataset_keys):
      raise ValueError('Expected dataset_keys to be of type DatasetKey')
    self._sorted_dataset_keys = sorted(dataset_keys)
    self._filtered_cache_entry_keys = (None if cache_entry_keys is None else
                                       set(cache_entry_keys))
    # TODO(b/37788560): Possibly use Riegeli as a default file format once
    # possible.
    self._source = source if source is not None else beam.io.ReadFromTFRecord

  def _should_read_cache_entry_key(self, key):
    return (self._filtered_cache_entry_keys is None or
            key in self._filtered_cache_entry_keys)

  def expand(self, pvalue):
    result = {}

    for dataset_key_idx, dataset_key in enumerate(self._sorted_dataset_keys):

      dataset_cache_path = _get_dataset_cache_path(self._cache_base_dir,
                                                   dataset_key)
      manifest_file = _ManifestFile(dataset_cache_path)
      manifest = manifest_file.read()
      if not manifest:
        continue
      result[dataset_key] = {}
      for key, cache_key_idx in manifest.items():
        if self._should_read_cache_entry_key(key):
          result[dataset_key][key] = (
              pvalue.pipeline
              | 'Read[AnalysisIndex{}][CacheKeyIndex{}]'.format(
                  dataset_key_idx, cache_key_idx) >> self._source('{}{}'.format(
                      os.path.join(dataset_cache_path, str(cache_key_idx)),
                      '-*-of-*')))
    return result


def validate_dataset_keys(dataset_keys):
  regex = re.compile(r'^[a-zA-Z0-9\.\-_]+$')
  for dataset_key in dataset_keys:
    if not isinstance(dataset_key, DatasetKey):
      raise ValueError('Dataset key {} must be of type DatasetKey')
    if not regex.match(dataset_key.key):
      raise ValueError(
          'Dataset key {!r} does not match allowed pattern: {!r}'.format(
              dataset_key.key, regex.pattern))


def make_cache_entry_key(cache_key):
  return _CACHE_VERSION + tf.compat.as_bytes(cache_key)
