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
import tensorflow_transform as tft
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.tf_metadata import metadata_io

# Users should avoid these aliases, they are provided for backwards
# compatibility only.
TRANSFORMED_METADATA_DIR = tft.TFTransformOutput.TRANSFORMED_METADATA_DIR
TRANSFORM_FN_DIR = tft.TFTransformOutput.TRANSFORM_FN_DIR


def _copy_tree(source, destination):
  """Recursively copies source to destination."""
  # TODO(b/35363519): Perhaps use Beam IO eventually (which also already
  # supports recursive copy)?
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  if tf.gfile.IsDirectory(source):
    tf.gfile.MakeDirs(destination)
    for filename in tf.gfile.ListDirectory(source):
      _copy_tree(
          os.path.join(source, filename), os.path.join(destination, filename))
  else:
    tf.gfile.Copy(source, destination)


class WriteTransformFn(beam.PTransform):
  """Writes a TransformFn to disk.

  The internal structure is a directory containing two subdirectories.  The
  first is 'transformed_metadata' and contains metadata of the transformed data.
  The second is 'transform_fn' and contains a SavedModel representing the
  transformed data.
  """

  def __init__(self, path):
    super(WriteTransformFn, self).__init__()
    self._path = path

  def _extract_input_pvalues(self, transform_fn):
    saved_model_dir, metadata = transform_fn
    pvalues = [saved_model_dir]
    if isinstance(metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
    return transform_fn, pvalues

  def expand(self, transform_fn):
    saved_model_dir, metadata = transform_fn

    metadata_path = os.path.join(self._path,
                                 tft.TFTransformOutput.TRANSFORMED_METADATA_DIR)
    pipeline = saved_model_dir.pipeline
    write_metadata_done = (
        metadata
        | 'WriteMetadata'
        >> beam_metadata_io.WriteMetadata(metadata_path, pipeline))

    transform_fn_path = os.path.join(self._path,
                                     tft.TFTransformOutput.TRANSFORM_FN_DIR)
    write_transform_fn_done = (
        saved_model_dir
        | 'WriteTransformFn' >> beam.Map(_copy_tree, transform_fn_path))

    # TODO(KesterTong): Move this "must follows" logic into a TFT wide helper
    # function or into Beam.
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
    transform_fn_path = os.path.join(self._path,
                                     tft.TFTransformOutput.TRANSFORM_FN_DIR)
    saved_model_dir_pcoll = (
        pvalue.pipeline
        | 'CreateTransformFnPath' >> beam.Create([transform_fn_path]))

    metadata = metadata_io.read_metadata(
        os.path.join(self._path,
                     tft.TFTransformOutput.TRANSFORMED_METADATA_DIR))

    return saved_model_dir_pcoll, metadata
