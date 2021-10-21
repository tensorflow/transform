# Copyright 2021 Google Inc. All Rights Reserved.
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
"""Experimental functions that involve a full pass over the dataset.

This module contains functions that are used in the preprocessing function, to
define a full pass operation such as computing the sum, min, max or unique
values of a tensor over the entire dataset.  This is implemented by a reduction
operation in the Beam implementation.

From the user's point of view, an analyzer appears as a regular TensorFlow
function, i.e. it accepts and returns tensors.  However it is represented in
the graph as a `Analyzer` which is not a TensorFlow op, but a placeholder for
the computation that takes place outside of TensorFlow.
"""

from typing import Any, Collection, List, Optional, Tuple, Type

import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import nodes


def _apply_analyzer(analyzer_def_cls: Type[analyzer_nodes.AnalyzerDef],
                    *tensor_inputs: common_types.TensorType,
                    **analyzer_def_kwargs: Any) -> Tuple[tf.Tensor, ...]:
  """Applies the analyzer over the whole dataset.

  Args:
    analyzer_def_cls: A class inheriting from analyzer_nodes.AnalyzerDef that
      should be applied.
    *tensor_inputs: A list of input `Tensor`s or `CompositeTensor`s.
    **analyzer_def_kwargs: KW arguments to use when constructing
      analyzer_def_cls.

  Returns:
    A list of `Tensor`s representing the values of the analysis result.
  """
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)
  output_value_nodes = nodes.apply_multi_output_operation(
      analyzer_def_cls,
      input_values_node,
      **analyzer_def_kwargs)
  return tuple(map(analyzer_nodes.wrap_as_tensor, output_value_nodes))


@common.log_api_use(common.ANALYZER_COLLECTION)
def ptransform_analyzer(
    inputs: Collection[tf.Tensor],
    ptransform: Any,
    output_dtypes: Collection[tf.dtypes.DType],
    output_shapes: Collection[List[int]],
    output_asset_default_values: Optional[Collection[Optional[bytes]]] = None,
    name: Optional[str] = None):
  # pylint: disable=line-too-long
  """Applies a user-provided PTransform over the whole dataset.

  WARNING: This is experimental.

  Note that in order to have asset files copied correctly, any outputs that
  represent asset filenames must be added to the `tf.GraphKeys.ASSET_FILEPATHS`
  collection by the caller if using Transform's APIs in compat v1 mode.

  Example:

  >>> class MeanPerKey(beam.PTransform):
  ...   def expand(self, pcoll):
  ...     # Returning a single PCollection since this analyzer has 1 output.
  ...     return (pcoll
  ...             | 'TuplesOfArraysToTuples' >> beam.FlatMap(lambda kv: list(zip(*kv)))
  ...             | 'MeanPerKey' >> beam.CombinePerKey(beam.combiners.MeanCombineFn())
  ...             | 'ToList' >> beam.combiners.ToList()
  ...             | 'ExtractMeans' >>
  ...             beam.Map(lambda outputs: [v for _, v in sorted(outputs)]))
  >>> def preprocessing_fn(inputs):
  ...   outputs = tft.experimental.ptransform_analyzer(
  ...       inputs=[inputs['s'], inputs['x']],
  ...       ptransform=MeanPerKey(),
  ...       output_dtypes=[tf.float32],
  ...       output_shapes=[[2]])
  ...   (mean_per_key,) = outputs
  ...   return { 'x/mean_a': inputs['x'] / mean_per_key[0] }
  >>> raw_data = [dict(x=1, s='a'), dict(x=8, s='b'), dict(x=3, s='a')]
  >>> feature_spec = dict(
  ...     x=tf.io.FixedLenFeature([], tf.float32),
  ...     s=tf.io.FixedLenFeature([], tf.string))
  >>> raw_data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
  ...     tft.tf_metadata.schema_utils.schema_from_feature_spec(feature_spec))
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'x/mean_a': 0.5}, {'x/mean_a': 4.0}, {'x/mean_a': 1.5}]

  Args:
    inputs: An ordered collection of input `Tensor`s.
    ptransform: A Beam PTransform that accepts a Beam PCollection where each
      element is a list of `ndarray`s.  Each element in the list contains a
      batch of values for the corresponding input tensor of the analyzer.  It
      returns a tuple of `PCollection`, each containing a single element which
      is an `ndarray`. It may inherit from
      `tft_beam.experimental.PTransformAnalyzer` if access to a temp base
      directory is needed.
    output_dtypes: An ordered collection of TensorFlow dtypes of the output of
      the analyzer.
    output_shapes: An ordered collection of shapes of the output of the
      analyzer. Must have the same length as output_dtypes.
    output_asset_default_values: (Optional) An ordered collection of optional
      `bytes` aligned with output_dtypes/output_shapes. Every item in this
      collection which is not `None` indicates that the output is a TF asset
      path, and its value would be used as the default value of this asset file
      prior to analysis.
    name: (Optional) Similar to a TF op name.  Used to define a unique scope for
      this analyzer, which can be used for debugging info.

  Returns:
    A list of output `Tensor`s.  These will have `dtype` and `shape` as
      specified by `output_dtypes` and `output_shapes`.

  Raises:
    ValueError: If output_dtypes and output_shapes have different lengths.
  """
  # pylint: enable=line-too-long
  if len(output_dtypes) != len(output_shapes):
    raise ValueError('output_dtypes ({}) and output_shapes ({}) had different'
                     ' lengths'.format(output_dtypes, output_shapes))
  if output_asset_default_values is not None:
    if len(output_asset_default_values) != len(output_dtypes):
      raise ValueError(
          'output_dtypes ({}) and output_asset_default_values ({}) had '
          'different lengths'.format(output_dtypes,
                                     output_asset_default_values))
  else:
    output_asset_default_values = [None] * len(output_dtypes)
  with tf.compat.v1.name_scope(name, 'ptransform'):
    output_tensor_infos = [
        analyzer_nodes.TensorInfo(dtype, shape, default_asset_content)
        for dtype, shape, default_asset_content in zip(
            output_dtypes, output_shapes, output_asset_default_values)
    ]
    return _apply_analyzer(
        analyzer_nodes.PTransform,
        *inputs,
        ptransform=ptransform,
        output_tensor_info_list=output_tensor_infos)
