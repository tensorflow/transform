# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Tests for tensorflow_transform.common."""

import tensorflow as tf
from tensorflow_transform import common
from tensorflow_transform import test_case


class CommonTest(test_case.TransformTestCase):

  def testLogAPIUse(self):

    @common.log_api_use("test_collection")
    def fn0():
      return None

    @common.log_api_use("test_collection")
    def fn1():
      return None

    @common.log_api_use("another_collection")
    def fn2():
      return None

    with tf.compat.v1.Graph().as_default() as graph:
      fn0()
      fn1()
      fn2()
      fn0()
      fn0()

    self.assertAllEqual([{"fn0": 3, "fn1": 1}],
                        graph.get_collection("test_collection"))
    self.assertAllEqual([{"fn2": 1}],
                        graph.get_collection("another_collection"))

  def testLogAPIUseWithNestedFunction(self):
    """Tests that API call is not logged when called from another logged API."""

    @common.log_api_use("test_collection")
    def fn0():
      fn1()
      return fn2()

    @common.log_api_use("test_collection")
    def fn1():
      return None

    @common.log_api_use("another_collection")
    def fn2():
      return None

    with tf.compat.v1.Graph().as_default() as graph:
      fn0()

    self.assertEqual([{"fn0": 1}], graph.get_collection("test_collection"))
    self.assertAllEqual([], graph.get_collection("another_collection"))


if __name__ == "__main__":
  test_case.main()
