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
"""Tensorflow-transform ExampleCoder tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_transform.beam.coders import example_proto_coder

from google.protobuf import text_format
import unittest


class ExampleProtoCoderTest(unittest.TestCase):

  _INPUT_SCHEMA = {
      'numeric1': tf.FixedLenFeature(shape=[], dtype=tf.int64),
      'numeric2': tf.VarLenFeature(dtype=tf.float32),
      'text1': tf.FixedLenFeature(shape=[0], dtype=tf.string),
      'category1': tf.VarLenFeature(dtype=tf.string),
      'y': tf.SparseFeature('idx', 'value', tf.float32, 10),
  }

  def _assert_encode_decode(self, coder, data, expected_proto,
                            expected_decoded):
    # Assert the data is decoded into the expected format.
    decoded = coder.decode(data)
    np.testing.assert_equal(expected_decoded, decoded)

    # Assert the decoded data can be encoded back into the original proto.
    encoded = coder.encode(decoded)
    parsed_proto = tf.train.Example()
    parsed_proto.ParseFromString(encoded)
    self.assertEqual(expected_proto, parsed_proto)

    # Assert the data can be decoded from the encoded string.
    decoded_again = coder.decode(encoded)
    np.testing.assert_equal(expected_decoded, decoded_again)

  def test_example_proto_coder(self):
    example_proto_text = """
    features {
      feature { key: "numeric1" value { int64_list { value: [ 12 ] } } }
      feature { key: "numeric2" value { float_list { value: [ 89.0 ] } } }
      feature { key: "text1"
                value { bytes_list { value: [ 'this is a ,text' ] } } }
      feature { key: "category1" value { bytes_list { value: [ 'female' ] } } }
      feature { key: "value" value { float_list { value: [ 12.0, 20.0 ] } } }
      feature { key: "idx" value { int64_list { value: [ 1, 4 ] } } }
    }
    """
    example_proto = tf.train.Example()
    text_format.Merge(example_proto_text, example_proto)
    data = example_proto.SerializeToString()

    coder = example_proto_coder.ExampleProtoCoder(self._INPUT_SCHEMA)
    expected_decoded = {
        'numeric1': np.array(12),
        'numeric2': np.array([89.0]),
        'text1': np.array(['this is a ,text']),
        'category1': np.array(['female']),
        'y': (np.array([12.0, 20.0]), np.array([1, 4]))
    }
    self._assert_encode_decode(coder, data, example_proto, expected_decoded)

if __name__ == '__main__':
  unittest.main()
