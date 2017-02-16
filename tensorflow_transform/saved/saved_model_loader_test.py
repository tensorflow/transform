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
"""Tests for saved_model_loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf

from tensorflow_transform.saved import saved_transform_io

import unittest


def _create_test_saved_model_dir():
  export_path = os.path.join(tempfile.mkdtemp(), 'export')

  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      input_float = tf.placeholder(tf.float32, shape=[1])
      output = (input_float - 2.0) / 5.0
      inputs = {'x': input_float}
      outputs = {'x_scaled': output}
      saved_transform_io.write_saved_transform_from_session(
          session, inputs, outputs, export_path)

  return export_path


class SavedModelLoaderTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._test_saved_model_dir = _create_test_saved_model_dir()

  # This class has no tests at the moment.

if __name__ == '__main__':
  unittest.main()
