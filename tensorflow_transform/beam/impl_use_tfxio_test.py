# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Same as impl_test.py, except that the TFXIO APIs are exercised."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import pyarrow as pa
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import impl_test
from tensorflow_transform.beam import tft_unit
from tfx_bsl.tfxio import tensor_adapter
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class BeamImplUseTFXIOTest(impl_test.BeamImplTest):

  def setUp(self):
    super(BeamImplUseTFXIOTest, self).setUp()
    tf.compat.v1.logging.info('Starting test case: %s', self._testMethodName)
    self._use_tfxio_context = beam_impl.Context(use_tfxio=True)
    self._use_tfxio_context.__enter__()

  def tearDown(self):
    self._use_tfxio_context.__exit__()
    super(BeamImplUseTFXIOTest, self).tearDown()

  # This is an override that passes use_tfxio=True to the overridden method.
  def assertAnalyzeAndTransformResults(self, *args, **kwargs):
    if 'use_tfxio' not in kwargs:
      kwargs['use_tfxio'] = True
    return super(
        BeamImplUseTFXIOTest, self).assertAnalyzeAndTransformResults(
            *args, **kwargs)

  # This is an override that passes use_tfxio=True to the overridden method.
  def assertAnalyzerOutputs(self, *args, **kwargs):
    if 'use_tfxio' not in kwargs:
      kwargs['use_tfxio'] = True
    return super(
        BeamImplUseTFXIOTest, self).assertAnalyzerOutputs(*args, **kwargs)

  def _UseTFXIO(self):
    return True

  def _MaybeConvertInputsToTFXIO(
      self, input_data, input_metadata, label='input_data'):
    return self.convert_to_tfxio_api_inputs(input_data, input_metadata, label)

  def testPassthroughKeys(self):
    passthrough_key = '__passthrough__'

    def preprocessing_fn(inputs):
      self.assertNotIn(passthrough_key, inputs)
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    x_data = [0., 1., 2.]
    passthrough_data = [1, None, 3]
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[x] for x in x_data], type=pa.list_(pa.float32())),
        pa.array([None if p is None else [p] for p in passthrough_data],
                 type=pa.list_(pa.int64())),
    ], ['x', passthrough_key])
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        input_record_batch.schema,
        {'x': text_format.Parse(
            'dense_tensor { column_name: "x" shape {} }',
            schema_pb2.TensorRepresentation())})
    expected_data = [{'x_scaled': x / 2.0, passthrough_key: p}
                     for x, p in zip(x_data, passthrough_data)]

    with self._makeTestPipeline() as pipeline:
      input_data = (
          pipeline | beam.Create([input_record_batch]))
      with beam_impl.Context(
          temp_dir=self.get_temp_dir(),
          passthrough_keys=set([passthrough_key])):
        (transformed_data, _), _ = (
            (input_data, tensor_adapter_config)
            | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

        def _assert_fn(output_data):
          self.assertCountEqual(expected_data, output_data)

        beam_test_util.assert_that(transformed_data, _assert_fn)


if __name__ == '__main__':
  tft_unit.main()
