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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import apache_beam as beam
from apache_beam.transforms import util as beam_test_util
import tensorflow as tf
from tensorflow_transform import api
from tensorflow_transform import base_tftransform_impl_test
from tensorflow_transform import impl_helper
from tensorflow_transform.beam import impl as beam_impl

import unittest


class AnalyzeAndTransformColumnsTest(
    base_tftransform_impl_test.BaseTFTransformImplTest):

  @property
  def impl(self):
    return beam_impl

  def testPipelineWithoutAutomaterialization(self):
    # The tests in BaseTFTransformImplTest, when run with the beam
    # implementation, pass lists instead of PCollections and thus invoke
    # automaterialization where each call to a beam PTransform will implicitly
    # run its own pipeline.
    #
    # In order to test the case where PCollections are not materialized in
    # between calls to the tf.Transform PTransforms, we include a test that is
    # not based on automaterialization.
    def preprocessing_fn(inputs):
      return {'x_scaled': api.scale_to_0_1(inputs['x'])}

    p = beam.Pipeline()
    schema = {'x': tf.FixedLenFeature((), tf.float32, 0)}
    columns = p | 'CreateTrainingData' >> beam.Create([{
        'x': v
    } for v in [4, 1, 5, 2]])
    _, transform_fn = (
        (columns, schema)
        | 'Analyze and Transform'
        >> self.impl.AnalyzeAndTransformDataset(preprocessing_fn))

    # Run transform_columns on some eval dataset.
    eval_data = p | 'CreateEvalData' >> beam.Create([{'x': v} for v in [6, 3]])
    transformed_eval_data, _ = (
        ((eval_data, schema), transform_fn)
        | 'Transform' >> self.impl.TransformDataset())
    p.run()
    expected_transformed_eval_data = [{'x_scaled': v} for v in [1.25, 0.5]]
    beam_test_util.assert_that(
        transformed_eval_data,
        beam_test_util.equal_to(expected_transformed_eval_data))

  def testRunExportedGraph(self):
    # Run analyze_and_transform_columns on some dataset.
    def preprocessing_fn(inputs):
      # TODO(abrao): Add an output column corresponding to SparseFeature when
      # that is supported.
      x_scaled = api.scale_to_0_1(inputs['x'])
      y_sum = api.transform(lambda y: tf.sparse_reduce_sum(y, axis=1),
                            inputs['y'])
      z_copy = api.transform(
          lambda z: tf.SparseTensor(z.indices, z.values, z.dense_shape),
          inputs['z'])
      return {'x_scaled': x_scaled, 'y_sum': y_sum, 'z_copy': z_copy}

    schema = {
        'x': tf.FixedLenFeature((), tf.float32, 0),
        'y': tf.SparseFeature('idx', 'val', tf.float32, 10),
        'z': tf.VarLenFeature(tf.float32)
    }
    columns = [
        {'x': 4, 'val': [0., 1.], 'idx': [0, 1], 'z': [2., 4., 6.]},
        {'x': 1, 'val': [2., 3.], 'idx': [2, 3], 'z': [8.]},
        {'x': 5, 'val': [4., 5.], 'idx': [4, 5], 'z': [1., 2., 3.]}
    ]
    _, (transform_fn, _) = (
        (columns, schema)
        | self.impl.AnalyzeAndTransformDataset(preprocessing_fn))

    # Import the function, and apply it to a batch of data.
    g = tf.Graph()
    with g.as_default():
      inputs, outputs = impl_helper.load_transform_fn_def(
          transform_fn[0])
      x, y, z = inputs['x'], inputs['y'], inputs['z']
      feed = {
          x: [6., 3., 0., 1.],
          y: tf.SparseTensorValue(
              indices=[[0, 6], [0, 7], [1, 8]],
              values=[6., 7., 8.],
              dense_shape=[2, 10]),
          z: tf.SparseTensorValue(
              indices=[[0, 1], [0, 2], [4, 10]],
              values=[1., 2., 3.],
              dense_shape=[4, 10])
      }

      sess = tf.Session()
      with sess.as_default():
        result = sess.run(outputs, feed_dict=feed)

      expected_transformed_data = {
          'x_scaled': [1.25, 0.5, -0.25, 0.0],
          'y_sum': [13.0, 8.0],
          'z_copy': tf.SparseTensorValue(
              indices=[[0, 1], [0, 2], [4, 10]],
              values=[1., 2., 3.],
              dense_shape=[4, 10])
      }
      self.assertDataEqual([expected_transformed_data], [result])

      # Verify that it breaks predictably if we feed unbatched data.
      with self.assertRaises(ValueError):
        feed = {
            x: 6.,
            y: tf.SparseTensorValue(indices=[[6], [7]], values=[6., 7.],
                                    dense_shape=[10])
        }
        sess = tf.Session()
        with sess.as_default():
          _ = sess.run(outputs, feed_dict=feed)


if __name__ == '__main__':
  unittest.main()
