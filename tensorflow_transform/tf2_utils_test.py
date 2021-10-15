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
"""Tests for tensorflow_transform.tf2_utils."""

import tensorflow as tf
from tensorflow_transform import tf2_utils
from tensorflow_transform import test_case

_TEST_BATCH_SIZES = [1, 10]
_TEST_DTYPES = [
    tf.int16,
    tf.int32,
    tf.int64,
    tf.float32,
    tf.float64,
    tf.string,
]

_TEST_TENSORS_TYPES = [
    (lambda dtype: tf.TensorSpec([None], dtype=dtype), tf.Tensor, []),
    (lambda dtype: tf.TensorSpec([None, 2], dtype=dtype), tf.Tensor, [2]),
    (lambda dtype: tf.RaggedTensorSpec([None, None], dtype=dtype),
     tf.RaggedTensor, [None]),
    (
        lambda dtype: tf.RaggedTensorSpec(  # pylint: disable=g-long-lambda
            [None, None, 2],
            dtype=dtype,
            ragged_rank=1),
        tf.RaggedTensor,
        [None, 2]),
]


class TF2UtilsTest(test_case.TransformTestCase):

  @test_case.parameters(*test_case.cross_parameters(
      [(x,) for x in _TEST_BATCH_SIZES],
      [(x,) for x in _TEST_DTYPES],
      _TEST_TENSORS_TYPES,
  ))
  def test_supply_missing_tensor_inputs(self, batch_size, dtype,
                                        type_spec_getter, tensor_type,
                                        inner_shape):
    test_case.skip_if_not_tf2('Tensorflow 2.x required.')

    @tf.function(input_signature=[{
        'x_1': tf.TensorSpec([None], dtype=tf.int32),
        'x_2': type_spec_getter(dtype),
    }])
    def foo(inputs):
      return inputs

    conc_fn = foo.get_concrete_function()
    # structured_input_signature is a tuple of (args, kwargs). [0][0] retrieves
    # the structure of the first arg, which for `foo` is `inputs`.
    structured_inputs = tf.nest.pack_sequence_as(
        conc_fn.structured_input_signature[0][0],
        conc_fn.inputs,
        expand_composites=True)
    missing_keys = ['x_2']
    result = tf2_utils.supply_missing_inputs(structured_inputs, batch_size,
                                             missing_keys)

    self.assertCountEqual(missing_keys, result.keys())
    self.assertIsInstance(result['x_2'], tensor_type)
    self.assertEqual(result['x_2'].shape.as_list(), [batch_size] + inner_shape)
    self.assertEqual(result['x_2'].dtype, dtype)


if __name__ == '__main__':
  test_case.main()
