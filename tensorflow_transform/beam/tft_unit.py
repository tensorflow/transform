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

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform import test_case
from tensorflow_transform.beam import test_helpers
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.coders import example_coder
from tensorflow.python.util.protobuf import compare  # pylint: disable=g-direct-tensorflow-import
import unittest


from tensorflow_metadata.proto.v0 import schema_pb2

parameters = test_case.parameters
cross_parameters = test_case.cross_parameters
named_parameters = test_case.named_parameters
cross_named_parameters = test_case.cross_named_parameters
is_tf_api_version_1 = test_case.is_tf_api_version_1
is_external_environment = test_case.is_external_environment
skip_if_not_tf2 = test_case.skip_if_not_tf2
SkipTest = test_case.SkipTest

main = test_case.main


mock = tf.compat.v1.test.mock


def metadata_from_feature_spec(feature_spec, domains=None):
  """Construct a DatasetMetadata from a feature spec.

  Args:
    feature_spec: A feature spec
    domains: A dict containing domains of features

  Returns:
    A `tft.tf_metadata.dataset_metadata.DatasetMetadata` object.
  """
  return dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(feature_spec, domains))


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

  def assertMetricsCounterEqual(self, metrics, name, expected_count,
                                namespaces_list=None):
    metrics_filter = beam.metrics.MetricsFilter().with_name(name)
    if namespaces_list:
      metrics_filter = metrics_filter.with_namespaces(namespaces_list)
    metric = metrics.query(
        metrics_filter)['counters']
    committed = sum([r.committed for r in metric])
    attempted = sum([r.attempted for r in metric])
    self.assertEqual(committed, attempted)
    self.assertEqual(committed, expected_count)

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
        output_shape = list(expected_outputs[key].shape)
        try:
          output_tensor.set_shape(output_shape)
        except ValueError as e:
          raise ValueError('Error for key {}: {}'.format(key, str(e)))
        # Add a batch dimension
        output_tensor = tf.expand_dims(output_tensor, 0)
        # Broadcast along the batch dimension
        result[key] = tf.tile(
            output_tensor, multiples=[batch_size] + [1] * len(output_shape))

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
    expected_metadata = metadata_from_feature_spec({
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
      input_data: Input data formatted in one of two ways:
        * A sequence of dicts whose values are one of:
          strings, lists of strings, numeric types or a pair of those.
          Must have at least one key so that we can infer the batch size, or
        * A sequence of pa.RecordBatch.
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
      with beam_impl.Context(
          temp_dir=temp_dir,
          desired_batch_size=desired_batch_size,
          force_tf_compat_v1=force_tf_compat_v1):
        input_data = pipeline | 'CreateInput' >> beam.Create(input_data,
                                                             reshuffle=False)
        if test_data is None:
          (transformed_data, transformed_metadata), transform_fn = (
              (input_data, input_metadata)
              | beam_impl.AnalyzeAndTransformDataset(
                  preprocessing_fn,
                  output_record_batches=output_record_batches))
        else:
          transform_fn = ((input_data, input_metadata)
                          | beam_impl.AnalyzeDataset(preprocessing_fn))
          test_data = pipeline | 'CreateTest' >> beam.Create(test_data)
          transformed_data, transformed_metadata = (
              ((test_data, input_metadata), transform_fn)
              | beam_impl.TransformDataset(
                  output_record_batches=output_record_batches))

        # Write transform_fn so we can test its assets
        _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

        transformed_data_path = os.path.join(temp_dir, 'transformed_data')
        if expected_data is not None:
          if isinstance(transformed_metadata,
                        beam_metadata_io.BeamDatasetMetadata):
            deferred_schema = (
                transformed_metadata.deferred_metadata
                | 'GetDeferredSchema' >> beam.Map(lambda m: m.schema))
          else:
            deferred_schema = (
                self.pipeline | 'CreateDeferredSchema' >> beam.Create(
                    [transformed_metadata.schema]))

          if output_record_batches:
            # Since we are using a deferred schema, obtain a pcollection
            # containing the data coder that will be created from it.
            transformed_data_coder_pcol = (
                deferred_schema | 'RecordBatchToExamplesEncoder' >> beam.Map(
                    example_coder.RecordBatchToExamplesEncoder))
            # Extract transformed RecordBatches and convert them to tf.Examples.
            encode_ptransform = 'EncodeRecordBatches' >> beam.FlatMapTuple(
                lambda batch, _, data_coder: data_coder.encode(batch),
                data_coder=beam.pvalue.AsSingleton(transformed_data_coder_pcol))
          else:
            # Since we are using a deferred schema, obtain a pcollection
            # containing the data coder that will be created from it.
            transformed_data_coder_pcol = (
                deferred_schema
                | 'ExampleProtoCoder' >> beam.Map(tft.coders.ExampleProtoCoder))
            encode_ptransform = 'EncodeExamples' >> beam.Map(
                lambda data, data_coder: data_coder.encode(data),
                data_coder=beam.pvalue.AsSingleton(transformed_data_coder_pcol))

          _ = (
              transformed_data
              | encode_ptransform
              | beam.io.tfrecordio.WriteToTFRecord(
                  transformed_data_path, shard_name_template=''))

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
