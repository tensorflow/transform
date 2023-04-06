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
"""Library for testing Tensorflow Transform."""

import os
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple
from absl import logging

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform import test_case
from tensorflow_transform.beam import test_helpers
from tensorflow_transform.tf_metadata import dataset_metadata
from tfx_bsl.coders import example_coder

import unittest
from tensorflow.python.util.protobuf import compare  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


parameters = test_case.parameters
cross_parameters = test_case.cross_parameters
named_parameters = test_case.named_parameters
cross_named_parameters = test_case.cross_named_parameters
is_external_environment = test_case.is_external_environment
skip_if_not_tf2 = test_case.skip_if_not_tf2
SkipTest = test_case.SkipTest

main = test_case.main


mock = tf.compat.v1.test.mock


def canonical_numeric_dtype(dtype):
  """Returns int64 for int dtypes and float32 for float dtypes."""
  if dtype.is_floating:
    return tf.float32
  elif dtype.is_integer:
    return tf.int64
  else:
    raise ValueError('Bad dtype {}'.format(dtype))


def make_feature_spec_wrapper(make_feature_spec, *args):
  """Skips test cases with RaggedFeature in TF 1.x."""
  try:
    return make_feature_spec(*args)
  except AttributeError as e:
    if 'no attribute \'RaggedFeature\'' in repr(e):
      raise unittest.SkipTest('RaggedFeature is not available in TF 1.x.')
    else:
      raise e


def _format_example_as_numpy_dict(example, feature_shape_dict):
  result = example_coder.ExampleToNumpyDict(example)
  for key, value in result.items():
    shape = feature_shape_dict[key]
    value = value.reshape(shape)
    if not shape:
      value = value.squeeze(0)
    result[key] = value
  return result


def _encode_transformed_data_batch(
    data: Tuple[pa.RecordBatch, Dict[str, pa.Array]],
    coder: example_coder.RecordBatchToExamplesEncoder) -> List[bytes]:
  """Produces a list of serialized tf.Examples from transformed data."""
  # Drop unary pass-through features that are not relevant for this testing
  # framework.
  record_batch, _ = data
  return coder.encode(record_batch)


class TransformTestCase(test_case.TransformTestCase):
  """Base test class for testing tf-transform preprocessing functions."""

  class _TestPipeline(beam.Pipeline):
    """Test pipeline class that retains pipeline metrics."""

    @property
    def has_ran(self):
      return hasattr(self, '_run_result')

    @property
    def metrics(self):
      if not self.has_ran:
        raise RuntimeError('Pipeline has to run before accessing its metrics')
      return self._run_result.metrics()

    def __exit__(self, exc_type, exc_val, exc_tb):
      if not exc_type:
        assert not self.has_ran
        self._run_result = self.run()
        self._run_result.wait_until_finish()

  def _makeTestPipeline(self):
    return self._TestPipeline(**test_helpers.make_test_beam_pipeline_kwargs())

  def _getMetricsCounter(self, metrics: beam.metrics.Metrics, name: str,
                         namespaces_list: Iterable[str]) -> int:
    metrics_filter = beam.metrics.MetricsFilter().with_name(name)
    if namespaces_list:
      metrics_filter = metrics_filter.with_namespaces(namespaces_list)
    metric = metrics.query(
        metrics_filter)['counters']
    committed = sum([r.committed for r in metric])
    attempted = sum([r.attempted for r in metric])
    self.assertEqual(
        committed,
        attempted,
        msg=f'Attempted counter {name} from namespace {namespaces_list}')
    return committed

  def assertMetricsCounterEqual(
      self,
      metrics: beam.metrics.Metrics,
      name: str,
      expected_count: int,
      namespaces_list: Optional[Iterable[str]] = None):
    counter_value = self._getMetricsCounter(metrics, name, namespaces_list)
    self.assertEqual(
        counter_value,
        expected_count,
        msg=f'Expected counter {name} from namespace {namespaces_list}')

  def assertMetricsCounterGreater(
      self,
      metrics: beam.metrics.Metrics,
      name: str,
      than: int,
      namespaces_list: Optional[Iterable[str]] = None):
    counter_value = self._getMetricsCounter(metrics, name, namespaces_list)
    self.assertGreater(
        counter_value,
        than,
        msg=f'Expected counter {name} from namespace {namespaces_list}')

  def assertAnalyzerOutputs(self,
                            input_data,
                            input_metadata,
                            analyzer_fn,
                            expected_outputs,
                            test_data=None,
                            desired_batch_size=None,
                            beam_pipeline=None,
                            force_tf_compat_v1=False,
                            output_record_batches=False):
    """Assert that input data and metadata is transformed as expected.

    This methods asserts transformed data and transformed metadata match
    with expected_data and expected_metadata.

    Args:
      input_data: Input data formatted in one of two ways:
        * A sequence of dicts whose values are one of:
          strings, lists of strings, numeric types or a pair of those.
          Must have at least one key so that we can infer the batch size, or
        * A sequence of pa.RecordBatch.
      input_metadata: One of -
        * DatasetMetadata describing input_data if `input_data` are dicts.
        * TensorAdapterConfig otherwise.
      analyzer_fn: A function taking a dict of tensors and returning a dict of
        tensors.  Unlike a preprocessing_fn, this should emit the results of a
        call to an analyzer, while a preprocessing_fn must typically add a batch
        dimension and broadcast across this batch dimension.
      expected_outputs: A dict whose keys are the same as those of the output of
        `analyzer_fn` and whose values are convertible to an ndarrays.
      test_data: (optional) If this is provided then instead of calling
        AnalyzeAndTransformDataset with input_data, this function will call
        AnalyzeDataset with input_data and TransformDataset with test_data.
        Must be provided if the input_data is empty. test_data should also
        conform to input_metadata.
      desired_batch_size: (Optional) A batch size to batch elements by. If not
        provided, a batch size will be computed automatically.
      beam_pipeline: (optional) A Beam Pipeline to use in this test.
      force_tf_compat_v1: A bool. If `True`, TFT's public APIs use
        Tensorflow in compat.v1 mode.
      output_record_batches: (optional) A bool. If `True`, `TransformDataset`
        and `AnalyzeAndTransformDataset` output `pyarrow.RecordBatch`es;
        otherwise, they output instance dicts.

    Raises:
      AssertionError: If the expected output does not match the results of
          the analyzer_fn.
    """

    def preprocessing_fn(inputs):
      """A helper function for validating analyzer outputs."""
      # Get tensors representing the outputs of the analyzers
      analyzer_outputs = analyzer_fn(inputs)

      # Check that keys of analyzer_outputs match expected_output.
      self.assertCountEqual(analyzer_outputs.keys(), expected_outputs.keys())

      # Get batch size from any input tensor.
      an_input = next(iter(inputs.values()))
      if isinstance(an_input, tf.RaggedTensor):
        batch_size = an_input.bounding_shape(axis=0)
      else:
        batch_size = tf.shape(input=an_input)[0]

      # Add a batch dimension and broadcast the analyzer outputs.
      result = {}
      for key, output_tensor in analyzer_outputs.items():
        # Get the expected shape, and set it.
        expected_output_shape = list(expected_outputs[key].shape)
        try:
          output_tensor.set_shape(expected_output_shape)
        except ValueError as e:
          raise ValueError(
              f'Error for key {key}, shapes are incompatible. Got '
              f'{output_tensor.shape}, expected {expected_output_shape}.'
          ) from e
        # Add a batch dimension
        output_tensor = tf.expand_dims(output_tensor, 0)
        # Broadcast along the batch dimension
        result[key] = tf.tile(
            output_tensor,
            multiples=[batch_size] + [1] * len(expected_output_shape))

      return result

    if input_data and not test_data:
      # Create test dataset by repeating the first instance a number of times.
      num_test_instances = 3
      test_data = [input_data[0]] * num_test_instances
      expected_data = [expected_outputs] * num_test_instances
    else:
      # Ensure that the test dataset is specified and is not empty.
      assert test_data
      expected_data = [expected_outputs] * len(test_data)
    expected_metadata = dataset_metadata.DatasetMetadata.from_feature_spec({
        key: tf.io.FixedLenFeature(value.shape, tf.as_dtype(value.dtype))
        for key, value in expected_outputs.items()
    })

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        test_data=test_data,
        desired_batch_size=desired_batch_size,
        beam_pipeline=beam_pipeline,
        force_tf_compat_v1=force_tf_compat_v1,
        output_record_batches=output_record_batches)

  def assertAnalyzeAndTransformResults(self,
                                       input_data,
                                       input_metadata,
                                       preprocessing_fn,
                                       expected_data=None,
                                       expected_metadata=None,
                                       expected_vocab_file_contents=None,
                                       test_data=None,
                                       desired_batch_size=None,
                                       beam_pipeline=None,
                                       temp_dir=None,
                                       force_tf_compat_v1=False,
                                       output_record_batches=False):
    """Assert that input data and metadata is transformed as expected.

    This methods asserts transformed data and transformed metadata match
    with expected_data and expected_metadata.

    Args:
      input_data: Input data formatted in one of three ways:
        * A sequence of dicts whose values are one of:
          strings, lists of strings, numeric types or a pair of those.
          Must have at least one key so that we can infer the batch size, or
        * A sequence of pa.RecordBatch.
        * A Beam source PTransform that produces either of the above.
      input_metadata: One of -
        * DatasetMetadata describing input_data if `input_data` are dicts.
        * TensorAdapterConfig otherwise.
      preprocessing_fn: A function taking a dict of tensors and returning
          a dict of tensors.
      expected_data: (optional) A dataset with the same type constraints as
          input_data, but representing the output after transformation.
          If supplied, transformed data is asserted to be equal.
      expected_metadata: (optional) DatasetMetadata describing the transformed
          data. If supplied, transformed metadata is asserted to be equal.
      expected_vocab_file_contents: (optional) A dictionary from vocab filenames
          to their expected content as a list of text lines or a list of tuples
          of frequency and text. Values should be the expected result of calling
          f.readlines() on the given asset files.
      test_data: (optional) If this is provided then instead of calling
          AnalyzeAndTransformDataset with input_data, this function will call
          AnalyzeDataset with input_data and TransformDataset with test_data.
          Note that this is the case even if input_data and test_data are equal.
          test_data should also conform to input_metadata.
      desired_batch_size: (optional) A batch size to batch elements by. If not
          provided, a batch size will be computed automatically.
      beam_pipeline: (optional) A Beam Pipeline to use in this test.
      temp_dir: If set, it is used as output directory, else a new unique
          directory is created.
      force_tf_compat_v1: A bool. If `True`, TFT's public APIs use Tensorflow
          in compat.v1 mode.
      output_record_batches: (optional) A bool. If `True`, `TransformDataset`
          and `AnalyzeAndTransformDataset` output `pyarrow.RecordBatch`es;
          otherwise, they output instance dicts.
    Raises:
      AssertionError: if the expected data does not match the results of
          transforming input_data according to preprocessing_fn, or
          (if provided) if the expected metadata does not match.
    """

    expected_vocab_file_contents = expected_vocab_file_contents or {}

    # Note: we don't separately test AnalyzeDataset and TransformDataset as
    # AnalyzeAndTransformDataset currently simply composes these two
    # transforms.  If in future versions of the code, the implementation
    # differs, we should also run AnalyzeDataset and TransformDataset composed.
    temp_dir = temp_dir or tempfile.mkdtemp(
        prefix=self._testMethodName, dir=self.get_temp_dir())
    with beam_pipeline or self._makeTestPipeline() as pipeline:
      with tft_beam.Context(
          temp_dir=temp_dir,
          desired_batch_size=desired_batch_size,
          force_tf_compat_v1=force_tf_compat_v1,
      ):
        source_ptransform = (
            input_data if isinstance(input_data, beam.PTransform) else
            beam.Create(input_data, reshuffle=False))
        input_data = pipeline | 'CreateInput' >> source_ptransform
        if test_data is None:
          (transformed_data, transformed_metadata), transform_fn = (
              input_data,
              input_metadata,
          ) | tft_beam.AnalyzeAndTransformDataset(
              preprocessing_fn, output_record_batches=output_record_batches
          )
        else:
          transform_fn = (input_data, input_metadata) | tft_beam.AnalyzeDataset(
              preprocessing_fn
          )
          test_data = pipeline | 'CreateTest' >> beam.Create(test_data)
          transformed_data, transformed_metadata = (
              (test_data, input_metadata),
              transform_fn,
          ) | tft_beam.TransformDataset(
              output_record_batches=output_record_batches
          )

        # Write transform_fn so we can test its assets
        _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

        transformed_data_path = os.path.join(temp_dir, 'transformed_data')
        if expected_data is not None:
          _ = (
              (transformed_data, transformed_metadata)
              | 'Encode' >> tft_beam.EncodeTransformedDataset()
              | 'Write'
              >> beam.io.tfrecordio.WriteToTFRecord(
                  transformed_data_path, shard_name_template=''
              )
          )

    # TODO(ebreck) Log transformed_data somewhere.
    tf_transform_output = tft.TFTransformOutput(temp_dir)
    if expected_data is not None:
      examples = tf.compat.v1.python_io.tf_record_iterator(
          path=transformed_data_path)
      shapes = {
          f.name:
          [s.size for s in f.shape.dim] if f.HasField('shape') else [-1]
          for f in tf_transform_output.transformed_metadata.schema.feature
      }
      transformed_data = [
          _format_example_as_numpy_dict(e, shapes) for e in examples
      ]
      self.assertDataCloseOrEqual(expected_data, transformed_data)

    if expected_metadata:
      # Make a copy with no annotations.
      transformed_schema = schema_pb2.Schema()
      transformed_schema.CopyFrom(
          tf_transform_output.transformed_metadata.schema)
      transformed_schema.ClearField('annotation')
      for feature in transformed_schema.feature:
        feature.ClearField('annotation')

      # assertProtoEqual has a size limit on the length of the
      # serialized as text strings. Therefore, we first try to use
      # assertProtoEqual, if that fails we try to use assertEqual, if that fails
      # as well then we raise the exception from assertProtoEqual.
      try:
        compare.assertProtoEqual(self, expected_metadata.schema,
                                 transformed_schema)
      except AssertionError as compare_exception:
        try:
          self.assertEqual(expected_metadata.schema, transformed_schema)
        except AssertionError:
          raise compare_exception

    for filename, file_contents in expected_vocab_file_contents.items():
      full_filename = tf_transform_output.vocabulary_file_by_name(filename)
      self.AssertVocabularyContents(full_filename, file_contents)

  def DebugPublishLatestsRenderedTFTGraph(
      self, output_file: Optional[str] = None
  ):
    """Outputs a rendered graph which may be used for debugging.

    Requires adding the binary resource to the test target:
    data = ["//third_party/graphviz:dot_binary"]

    Args:
      output_file: Path to output the rendered graph file.
    """
    logging.info(
        'DebugPublishLatestsRenderedTFTGraph is not currently supported.'
    )
