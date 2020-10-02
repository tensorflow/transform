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
"""Same as tukey_hh_params_integration_test.py, except that the TF2 Beam APIs are exercised."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam import tukey_hh_params_integration_test


class TukeyHHParamsIntegrationV2Test(
    tukey_hh_params_integration_test.TukeyHHParamsIntegrationTest):

  def setUp(self):
    super(TukeyHHParamsIntegrationV2Test, self).setUp()
    tft_unit.skip_if_not_tf2('Tensorflow 2.x required')
    tf.compat.v1.logging.info('Starting test case: %s', self._testMethodName)
    self._force_tf_compat_v1_context = beam_impl.Context(
        force_tf_compat_v1=False)
    self._force_tf_compat_v1_context.__enter__()

  # This is an override that passes force_tf_compat_v1=False to the overridden
  # method.
  def assertAnalyzeAndTransformResults(self, *args, **kwargs):
    kwargs['force_tf_compat_v1'] = False
    return super(TukeyHHParamsIntegrationV2Test,
                 self).assertAnalyzeAndTransformResults(*args, **kwargs)

  # This is an override that passes force_tf_compat_v1=False to the overridden
  # method.
  def assertAnalyzerOutputs(self, *args, **kwargs):
    kwargs['force_tf_compat_v1'] = False
    return super(TukeyHHParamsIntegrationV2Test,
                 self).assertAnalyzerOutputs(*args, **kwargs)


if __name__ == '__main__':
  tft_unit.main()
