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
"""Transforms to read/write transform functions from disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import apache_beam as beam
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.tf_metadata import metadata_io


def _copy_tree(source, destination):
  """Recursively copies source to destination."""
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  if tf.gfile.IsDirectory(source):
    tf.gfile.MakeDirs(destination)
    for filename in tf.gfile.ListDirectory(source):
      _copy_tree(
          os.path.join(source, filename), os.path.join(destination, filename))
  else:
    tf.gfile.Copy(source, destination)


class WriteTransformFn(beam.PTransform):
  """Writes a TransformFn to disk."""

  def __init__(self, path):
    super(WriteTransformFn, self).__init__()
    self._path = path

  def _extract_input_pvalues(self, transform_fn):
    saved_model_dir, (_, property_pcoll) = transform_fn
    return transform_fn, [saved_model_dir, property_pcoll]

  def expand(self, transform_fn):
    saved_model_dir, properties = transform_fn

    metadata_path = os.path.join(self._path, 'transformed_metadata')
    pipeline = saved_model_dir.pipeline
    write_metadata_done = (
        properties
        | 'WriteMetadata'
        >> beam_metadata_io.WriteMetadata(metadata_path, pipeline))

    transform_fn_path = os.path.join(self._path, 'transform_fn')
    write_transform_fn_done = (
        saved_model_dir
        | 'WriteTransformFn' >> beam.Map(_copy_tree, transform_fn_path))

    return (
        write_transform_fn_done
        | 'WaitOnWriteMetadataDone' >> beam.Map(
            lambda x, dummy: x,
            dummy=beam.pvalue.AsSingleton(write_metadata_done)))


class ReadTransformFn(beam.PTransform):
  """Reads a TransformFn written by WriteTransformFn."""

  def __init__(self, path):
    super(ReadTransformFn, self).__init__()
    self._path = path

  def expand(self, pvalue):
    transform_fn_path = os.path.join(self._path, 'transform_fn')
    saved_model_dir_pcoll = (
        pvalue.pipeline
        | 'CreateTransformFnPath' >> beam.Create([transform_fn_path]))

    metadata = metadata_io.read_metadata(
        os.path.join(self._path, 'transformed_metadata'))
    deferred_metadata = (
        pvalue.pipeline | 'CreateEmptyDeferredMetadata' >> beam.Create([{}]))

    return saved_model_dir_pcoll, (metadata, deferred_metadata)
