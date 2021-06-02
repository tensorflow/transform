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

import os

import apache_beam as beam
import tensorflow_transform as tft
from tensorflow_transform import impl_helper
from tensorflow_transform.beam import common
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.tf_metadata import metadata_io

# Users should avoid these aliases, they are provided for backwards
# compatibility only.
TRANSFORMED_METADATA_DIR = tft.TFTransformOutput.TRANSFORMED_METADATA_DIR
TRANSFORM_FN_DIR = tft.TFTransformOutput.TRANSFORM_FN_DIR


def _copy_tree_to_unique_temp_dir(source, base_temp_dir_path):
  """Copies from source to a unique sub directory under base_temp_dir_path."""
  destination = common.get_unique_temp_path(base_temp_dir_path)
  _copy_tree(source, destination)
  return destination


def _copy_tree(source, destination):
  """Recursively copies source to destination."""
  # TODO(b/35363519): Perhaps use Beam IO eventually (which also already
  # supports recursive copy)?
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  if tf.io.gfile.isdir(source):
    source_dir_name = os.path.basename(os.path.normpath(source))
    if source_dir_name == impl_helper.METADATA_DIR_NAME:
      return

    tf.io.gfile.makedirs(destination)
    for filename in tf.io.gfile.listdir(source):
      _copy_tree(
          os.path.join(source, filename), os.path.join(destination, filename))
  else:
    tf.io.gfile.copy(source, destination)


class WriteTransformFn(beam.PTransform):
  """Writes a TransformFn to disk.

  The internal structure is a directory containing two subdirectories.  The
  first is 'transformed_metadata' and contains metadata of the transformed data.
  The second is 'transform_fn' and contains a SavedModel representing the
  transformed data.
  """

  def __init__(self, path):
    super().__init__()
    self._path = path

  def _extract_input_pvalues(self, transform_fn):
    saved_model_dir, metadata = transform_fn
    pvalues = [saved_model_dir]
    if isinstance(metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
    return transform_fn, pvalues

  def expand(self, transform_fn):
    saved_model_dir, metadata = transform_fn
    pipeline = saved_model_dir.pipeline

    # Using a temp dir within `path` ensures that the source and dstination
    # paths for the rename below are in the same file system.
    base_temp_dir = os.path.join(self._path, 'transform_tmp')
    temp_metadata_path = (
        metadata
        | 'WriteMetadataToTemp' >> beam_metadata_io.WriteMetadata(
            base_temp_dir, pipeline, write_to_unique_subdirectory=True))

    temp_transform_fn_path = (
        saved_model_dir
        | 'WriteTransformFnToTemp' >> beam.Map(_copy_tree_to_unique_temp_dir,
                                               base_temp_dir))

    metadata_path = os.path.join(self._path,
                                 tft.TFTransformOutput.TRANSFORMED_METADATA_DIR)
    transform_fn_path = os.path.join(self._path,
                                     tft.TFTransformOutput.TRANSFORM_FN_DIR)

    def publish_outputs(unused_element, metadata_source_path,
                        transform_fn_source_path):
      import tensorflow as tf  # pylint: disable=g-import-not-at-top
      if not tf.io.gfile.exists(self._path):
        tf.io.gfile.makedirs(self._path)

      tf.io.gfile.rename(metadata_source_path, metadata_path, overwrite=True)
      tf.io.gfile.rename(
          transform_fn_source_path, transform_fn_path, overwrite=True)
      tf.io.gfile.rmtree(base_temp_dir)

    # TODO(KesterTong): Move this "must follows" logic into a tfx_bsl helper
    # function or into Beam.
    return (
        pipeline
        | 'CreateSole' >> beam.Create([None])
        | 'PublishMetadataAndTransformFn' >> beam.Map(
            publish_outputs,
            metadata_source_path=beam.pvalue.AsSingleton(temp_metadata_path),
            transform_fn_source_path=beam.pvalue.AsSingleton(
                temp_transform_fn_path)))


class ReadTransformFn(beam.PTransform):
  """Reads a TransformFn written by WriteTransformFn."""

  def __init__(self, path):
    super().__init__()
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
