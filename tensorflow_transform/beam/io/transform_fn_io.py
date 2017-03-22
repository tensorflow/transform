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
from apache_beam.io import fileio
from tensorflow_transform.beam.io import beam_metadata_io


class WriteTransformFn(beam.PTransform):
  """Writes a TransformFn to disk.

  The transform function will be written to the specified directory, with the
  SavedModel written to the transform_fn/ subdirectory and the output metadata
  written to the transformed_metadata/ directory.
  """

  def __init__(self, path):
    super(WriteTransformFn, self).__init__()
    self._path = path

  def _extract_input_pvalues(self, transform_fn):
    saved_model_dir_pcoll, _ = transform_fn
    return transform_fn, [saved_model_dir_pcoll]

  def expand(self, transform_fn):
    saved_model_dir_pcoll, metadata = transform_fn
    # Write metadata in non-deferred manner.  Once metadata contains deferred
    # components, the deferred components will be written in a deferred manner
    # while the non-deferred components will be written in a non-deferred
    # manner.
    def copy_tree(source, dest):
      fileio.ChannelFactory.copytree(source + '/', dest + '/')
    _ = metadata | 'WriteMetadata' >> beam_metadata_io.WriteMetadata(
        os.path.join(self._path, 'transformed_metadata'),
        pipeline=saved_model_dir_pcoll.pipeline)
    return saved_model_dir_pcoll | 'WriteTransformFn' >> beam.Map(
        copy_tree, os.path.join(self._path, 'transform_fn'))


class ReadTransformFn(beam.PTransform):
  """Reads a TransformFn written by WriteTransformFn.

  See WriteTransformFn for the directory layout.
  """

  def __init__(self, path):
    super(ReadTransformFn, self).__init__()
    self._path = path

  def expand(self, pvalue):
    metadata = (
        pvalue.pipeline | 'ReadMetadata' >> beam_metadata_io.ReadMetadata(
            os.path.join(self._path, 'transformed_metadata')))
    saved_model_dir_pcoll = pvalue | 'CreateDir' >> beam.Create(
        [os.path.join(self._path, 'transform_fn')])
    return (saved_model_dir_pcoll, metadata)
