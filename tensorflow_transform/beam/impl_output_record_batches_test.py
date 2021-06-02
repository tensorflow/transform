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
"""Same as impl_test.py, except that impl produces `pa.RecordBatch`es."""

import collections

import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_transform import impl_helper
from tensorflow_transform.beam import impl
from tensorflow_transform.beam import impl_test
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.tfxio.tensor_adapter import TensorAdapterConfig


class BeamImplOutputRecordBatchesTest(impl_test.BeamImplTest):

  def _OutputRecordBatches(self):
    return True

  def _MakeTransformOutputAssertFn(self, expected, sort=False):
    # Merge expected instance dicts.
    merged_expected = collections.defaultdict(list)
    for instance_dict in expected:
      for key, value in instance_dict.items():
        # Scalars must be wrapped in a list.
        if (hasattr(value, '__iter__') and not isinstance(value,
                                                          (str, bytes)) or
            value is None):
          maybe_wrapped_value = value
        else:
          maybe_wrapped_value = [value]
        merged_expected[key].append(maybe_wrapped_value)

    def _assert_fn(actual):
      # Merge output RecordBatches.
      merged_actual = collections.defaultdict(list)
      for record_batch, _ in actual:
        for key, value in record_batch.to_pydict().items():
          merged_actual[key].extend(value)
      if sort:
        for value in merged_actual.values():
          value.sort()
        for value in merged_expected.values():
          value.sort()
      self.assertDictEqual(merged_expected, merged_actual)

    return _assert_fn

  def testConvertToRecordBatchPassthroughData(self):
    passthrough_key1 = '__passthrough_with_batch_length__'
    passthrough_key2 = '__passthrough_with_one_value__'
    passthrough_key3 = '__passthrough_with_one_distinct_value_none__'
    passthrough_key4 = '__passthrough_with_one_distinct_value_not_none__'
    batch_dict = {
        'a':
            np.array([100, 1, 10], np.int64),
        passthrough_key1:
            pa.array([[1], None, [0]], pa.large_list(pa.int64())),
        passthrough_key2:
            pa.array([None], pa.large_list(pa.float32())),
        passthrough_key3:
            pa.array([None, None], pa.large_list(pa.large_binary())),
        passthrough_key4:
            pa.array([[10], [10]], pa.large_list(pa.int64()))
    }
    schema = schema_utils.schema_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.int64)})
    converter = impl_helper.make_tensor_to_arrow_converter(schema)
    passthrough_keys = {
        passthrough_key1, passthrough_key2, passthrough_key3, passthrough_key4
    }
    arrow_schema = pa.schema([
        ('a', pa.large_list(pa.int64())),
        (passthrough_key1, batch_dict[passthrough_key1].type),
        (passthrough_key2, batch_dict[passthrough_key2].type),
        (passthrough_key3, batch_dict[passthrough_key3].type),
        (passthrough_key4, batch_dict[passthrough_key4].type)
    ])
    # Note that we only need `input_metadata.arrow_schema`.
    input_metadata = TensorAdapterConfig(arrow_schema, {})
    record_batch, unary_features = impl._convert_to_record_batch(
        batch_dict, schema, converter, passthrough_keys, input_metadata)
    expected_record_batch = {
        'a': [[100], [1], [10]],
        passthrough_key1: [[1], None, [0]]
    }
    self.assertDictEqual(expected_record_batch, record_batch.to_pydict())
    expected_unary_features = {
        passthrough_key2: [None],
        passthrough_key3: [None],
        passthrough_key4: [[10]]
    }
    unary_features = {k: v.to_pylist() for k, v in unary_features.items()}
    self.assertDictEqual(expected_unary_features, unary_features)

    # Test pass-through data when input and output batch sizes are different and
    # the number of its unique values is >1.
    passthrough_key5 = '__passthrough_with_wrong_batch_size__'
    passthrough_keys.add(passthrough_key5)
    batch_dict[passthrough_key5] = pa.array([[1], [2]],
                                            pa.large_list(pa.int64()))
    input_metadata.arrow_schema = input_metadata.arrow_schema.append(
        pa.field(passthrough_key5, batch_dict[passthrough_key5].type))
    with self.assertRaisesRegexp(
        ValueError, 'Cannot pass-through data when '
        'input and output batch sizes are different'):
      _ = impl._convert_to_record_batch(batch_dict, schema, converter,
                                        passthrough_keys, input_metadata)


if __name__ == '__main__':
  tft_unit.main()
