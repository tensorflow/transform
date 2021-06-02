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
"""Same as impl_test.py, except that the TF2 Beam APIs are exercised."""

import tensorflow as tf
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import impl_test
from tensorflow_transform.beam import tft_unit


class BeamImplV2Test(impl_test.BeamImplTest):

  def setUp(self):
    super().setUp()
    tft_unit.skip_if_not_tf2('Tensorflow 2.x required')
    tf.compat.v1.logging.info('Starting test case: %s', self._testMethodName)
    self._force_tf_compat_v1_context = beam_impl.Context(
        force_tf_compat_v1=False)
    self._force_tf_compat_v1_context.__enter__()

  def _UseTFCompatV1(self):
    return False


if __name__ == '__main__':
  tft_unit.main()
