# Copyright 2021 Google Inc. All Rights Reserved.
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
"""Tests for tensorflow_transform.annotators."""

import tensorflow as tf
from tensorflow_transform import annotators
from tensorflow_transform import test_case

from tensorflow.python.training.tracking import base  # pylint: disable=g-direct-tensorflow-import


class AnnotatorsTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      dict(testcase_name='tf_compat_v1', use_tf_compat_v1=True),
      dict(testcase_name='tf2', use_tf_compat_v1=False))
  def test_annotate_asset(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')

    def foo():
      annotators.annotate_asset('scope/my_key', 'scope/my_value')
      annotators.annotate_asset('my_key2', 'should_be_replaced')
      annotators.annotate_asset('my_key2', 'my_value2')

    if use_tf_compat_v1:
      with tf.Graph().as_default() as graph:
        foo()
    else:
      graph = tf.function(foo).get_concrete_function().graph

    self.assertDictEqual(
        annotators.get_asset_annotations(graph), {
            'my_key': 'my_value',
            'my_key2': 'my_value2'
        })

    annotators.clear_asset_annotations(graph)
    self.assertDictEqual(annotators.get_asset_annotations(graph), {})

  def test_object_tracker(self):
    test_case.skip_if_not_tf2('Tensorflow 2.x required')

    trackable_object = base.Trackable()

    @tf.function
    def preprocessing_fn():
      _ = annotators.make_and_track_object(lambda: trackable_object)
      return 1

    object_tracker = annotators.ObjectTracker()
    with annotators.object_tracker_scope(object_tracker):
      _ = preprocessing_fn()

    self.assertLen(object_tracker.trackable_objects, 1)
    self.assertEqual(trackable_object, object_tracker.trackable_objects[0])


if __name__ == '__main__':
  test_case.main()
