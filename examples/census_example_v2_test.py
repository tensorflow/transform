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
"""Tests for census_example_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# GOOGLE-INITIALIZATION

import tensorflow.compat.v2 as tf
import census_example_v2
import local_model_server
from google.protobuf import text_format

# Use first row of test data set, which has high probability on label 1 (which
# corresponds to '<=50K').
_PREDICT_TF_EXAMPLE_TEXT_PB = """
    features {
      feature {
        key: "age"
        value { float_list: { value: 25 } }
      }
      feature {
        key: "workclass"
        value { bytes_list: { value: "Private" } }
      }
      feature {
        key: "education"
        value { bytes_list: { value: "11th" } }
      }
      feature {
        key: "education-num"
        value { float_list: { value: 7 } }
      }
      feature {
        key: "marital-status"
        value { bytes_list: { value: "Never-married" } }
      }
      feature {
        key: "occupation"
        value { bytes_list: { value: "Machine-op-inspct" } }
      }
      feature {
        key: "relationship"
        value { bytes_list: { value: "Own-child" } }
      }
      feature {
        key: "race"
        value { bytes_list: { value: "Black" } }
      }
      feature {
        key: "sex"
        value { bytes_list: { value: "Male" } }
      }
      feature {
        key: "capital-gain"
        value { float_list: { value: 0 } }
      }
      feature {
        key: "capital-loss"
        value { float_list: { value: 0 } }
      }
      feature {
        key: "hours-per-week"
        value { float_list: { value: 40 } }
      }
      feature {
        key: "native-country"
        value { bytes_list: { value: "United-States" } }
      }
    }
    """

_MODEL_NAME = 'my_model'

_CLASSIFICATION_REQUEST_TEXT_PB = """model_spec { name: "%s" }
    input {
      example_list {
        examples {
          %s
        }
      }
    }""" % (_MODEL_NAME, _PREDICT_TF_EXAMPLE_TEXT_PB)


class CensusExampleV2Test(tf.test.TestCase):

  def _get_data_dir(self):
    return os.path.join(os.path.dirname(__file__), 'testdata/census')

  def _get_working_dir(self):
    return os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  def _should_saved_model_load_work(self):
    # TODO(b/148082271): Rely only on the TF version here once it's consistent.
    from tensorflow.python.keras.utils import generic_utils  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    return tf.__version__ >= '2.2' or hasattr(generic_utils, 'validate_config')

  def testCensusExampleAccuracy(self):

    if not self._should_saved_model_load_work():
      self.skipTest('The generated SavedModel cannot be read with TF<2.2')
    raw_data_dir = self._get_data_dir()
    working_dir = self._get_working_dir()

    train_data_file = os.path.join(raw_data_dir, 'adult.data')
    test_data_file = os.path.join(raw_data_dir, 'adult.test')

    census_example_v2.transform_data(train_data_file, test_data_file,
                                     working_dir)

    train_pattern = os.path.join(
        working_dir, census_example_v2.TRANSFORMED_TRAIN_DATA_FILEBASE + '*')
    eval_pattern = os.path.join(
        working_dir, census_example_v2.TRANSFORMED_TEST_DATA_FILEBASE + '*')
    output_dir = os.path.join(working_dir, census_example_v2.EXPORTED_MODEL_DIR)
    results = census_example_v2.train_and_evaluate(
        train_pattern,
        eval_pattern,
        output_dir,
        working_dir,
        num_train_instances=1000,
        num_test_instances=1000)
    self.assertGreaterEqual(results[1], 0.7)

    # Removing the tf.Transform output directory in order to show that the
    # exported model is hermetic.
    shutil.rmtree(os.path.join(working_dir, 'transform_fn'))

    model_path = os.path.join(working_dir, census_example_v2.EXPORTED_MODEL_DIR)

    actual_model_path = os.path.join(model_path, '1')
    tf.keras.backend.clear_session()
    with tf.compat.v1.Graph().as_default():
      model = tf.keras.models.load_model(actual_model_path)
      model.summary()

      example = text_format.Parse(_PREDICT_TF_EXAMPLE_TEXT_PB,
                                  tf.train.Example())
      prediction = model.signatures['serving_default'](
          tf.constant([example.SerializeToString()], tf.string))
      with tf.compat.v1.keras.backend.get_session() as sess:
        prediction = sess.run(prediction)
      self.assertAllEqual([['0', '1']], prediction['classes'])
      self.assertAllClose([[0, 1]], prediction['scores'], atol=0.001)

    # This is required in order to support the classify API for this Keras
    # model.
    updater = tf.compat.v1.saved_model.signature_def_utils.MethodNameUpdater(
        actual_model_path)
    updater.replace_method_name(
        signature_key='serving_default',
        method_name='tensorflow/serving/classify',
        tags=['serve'])
    updater.save()

    if local_model_server.local_model_server_supported():
      with local_model_server.start_server(_MODEL_NAME, model_path) as address:
        ascii_classification_request = _CLASSIFICATION_REQUEST_TEXT_PB
        results = local_model_server.make_classification_request(
            address, ascii_classification_request)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].classes), 2)
        self.assertEqual(results[0].classes[0].label, '0')
        self.assertLess(results[0].classes[0].score, 0.01)
        self.assertEqual(results[0].classes[1].label, '1')
        self.assertGreater(results[0].classes[1].score, 0.99)

  def test_main_runs(self):
    census_example_v2.main(
        self._get_data_dir(),
        self._get_working_dir(),
        num_train_instances=10,
        num_test_instances=10)


if __name__ == '__main__':
  tf.test.main()
