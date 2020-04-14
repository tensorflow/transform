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

import tensorflow as tf
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import impl_test
from tensorflow_transform.beam import tft_unit


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


if __name__ == '__main__':
  tft_unit.main()
