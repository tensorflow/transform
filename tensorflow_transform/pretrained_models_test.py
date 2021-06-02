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
"""Tests for tensorflow_transform.pretrained_models."""

import os

import tensorflow as tf
from tensorflow_transform import pretrained_models


class PretrainedModelsTest(tf.test.TestCase):

  def save_model_with_single_input(self, export_dir):
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.compat.v1.Graph().as_default() as graph:
      with self.test_session(graph=graph) as sess:
        input1 = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='myinput')
        initializer = tf.compat.v1.initializers.constant([1, 2, 3, 4, 5])
        with tf.compat.v1.variable_scope(
            'Model', reuse=None, initializer=initializer):
          v1 = tf.compat.v1.get_variable('v1', [5], dtype=tf.int32)
        output1 = tf.add(v1, input1, name='myadd')
        inputs = {'single_input': input1}
        outputs = {'single_output': output1}
        signature_def_map = {
            'my_signature_single_input':
                tf.compat.v1.saved_model.signature_def_utils
                .predict_signature_def(inputs, outputs)
        }
        sess.run(tf.compat.v1.global_variables_initializer())
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.SERVING], signature_def_map=signature_def_map)
        builder.save(False)

  def save_model_with_multi_inputs(self, export_dir):
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.compat.v1.Graph().as_default() as graph:
      with self.test_session(graph=graph) as sess:
        input1 = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='myinput1')
        input2 = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='myinput2')
        input3 = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='myinput3')
        initializer = tf.compat.v1.initializers.constant([1, 2, 3, 4, 5])
        with tf.compat.v1.variable_scope(
            'Model', reuse=None, initializer=initializer):
          v1 = tf.compat.v1.get_variable('v1', [5], dtype=tf.int32)
        o1 = tf.add(v1, input1, name='myadd1')
        o2 = tf.add(o1, input2, name='myadd2')
        output1 = tf.add(o2, input3, name='myadd3')
        inputs = {'input_name1': input1, 'input_name2': input2,
                  'input_name3': input3}
        outputs = {'single_output': output1}
        signature_def_map = {
            'my_signature_multi_input':
                tf.compat.v1.saved_model.signature_def_utils
                .predict_signature_def(inputs, outputs)
        }
        sess.run(tf.compat.v1.global_variables_initializer())
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.SERVING], signature_def_map=signature_def_map)
        builder.save(False)

  def make_tensor_fn_two_inputs(self):
    def tensor_fn(input1, input2):
      initializer = tf.compat.v1.initializers.constant([1, 2, 3])
      with tf.compat.v1.variable_scope(
          'Model', reuse=None, initializer=initializer):
        v1 = tf.compat.v1.get_variable('v1', [3], dtype=tf.int64)
        o1 = tf.add(v1, input1, name='myadda1')
        o = tf.subtract(o1, input2, name='myadda2')
        return o
    return tensor_fn

  def save_checkpoint_with_two_inputs(self, checkpoint_path):
    test_tensor_fn = self.make_tensor_fn_two_inputs()
    with tf.compat.v1.Graph().as_default() as graph:
      with self.test_session(graph=graph) as sess:
        input1 = tf.compat.v1.placeholder(
            dtype=tf.int64, shape=[3], name='myinputa')
        input2 = tf.compat.v1.placeholder(
            dtype=tf.int64, shape=[3], name='myinputb')
        test_tensor_fn(input1, input2)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, checkpoint_path)

  def testApplySavedModelSingleInput(self):
    export_dir = os.path.join(self.get_temp_dir(), 'single_input')
    self.save_model_with_single_input(export_dir)
    with tf.compat.v1.Graph().as_default() as graph:
      with self.test_session(graph=graph) as sess:
        input_tensor = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='input_tensor')
        output_tensor = pretrained_models.apply_saved_model(
            export_dir, input_tensor, [tf.saved_model.SERVING])
        feed_dict = {input_tensor: [2, 2, 2, 2, 2]}
        output_value = sess.run(output_tensor, feed_dict=feed_dict)
        self.assertAllEqual(output_value, [3, 4, 5, 6, 7])

  def testApplySavedModelMultiInputs(self):
    export_dir = os.path.join(self.get_temp_dir(), 'multi_inputs')
    self.save_model_with_multi_inputs(export_dir)
    with tf.compat.v1.Graph().as_default() as graph:
      with self.test_session(graph=graph) as sess:
        input_tensor_1 = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='input_tensor_1')
        input_tensor_2 = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='input_tensor_2')
        input_tensor_3 = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[5], name='input_tensor_3')
        inputs = {
            'input_name1': input_tensor_1,
            'input_name2': input_tensor_2,
            'input_name3': input_tensor_3
        }
        output_tensor = pretrained_models.apply_saved_model(
            export_dir,
            inputs, [tf.saved_model.SERVING],
            signature_name='my_signature_multi_input')
        feed_dict = {input_tensor_1: [2, 3, 4, 5, 6],
                     input_tensor_2: [1, 1, 1, 1, 1],
                     input_tensor_3: [1, 1, 1, 1, -1]}
        output_value = sess.run(output_tensor, feed_dict=feed_dict)
        self.assertAllEqual(output_value, [5, 7, 9, 11, 11])

  def testApplyFunctionWithCheckpointTwoInputs(self):
    checkpoint = os.path.join(self.get_temp_dir(), 'checkpoint_two')
    self.save_checkpoint_with_two_inputs(checkpoint)
    with tf.compat.v1.Graph().as_default() as graph:
      with self.test_session(graph=graph) as sess:
        input1 = tf.compat.v1.placeholder(
            dtype=tf.int64, shape=[3], name='input1')
        input2 = tf.compat.v1.placeholder(
            dtype=tf.int64, shape=[3], name='input2')
        output_tensor = pretrained_models.apply_function_with_checkpoint(
            self.make_tensor_fn_two_inputs(), [input1, input2], checkpoint)
        feed_dict = {input1: [1, 2, 3], input2: [3, 2, 1]}
        output_value = sess.run(output_tensor, feed_dict=feed_dict)
        # [1, 2, 3] + [1, 2, 3] - [3, 2, 1] = [-1, 2, 5]
        self.assertAllEqual(output_value, [-1, 2, 5])


if __name__ == '__main__':
  tf.test.main()
