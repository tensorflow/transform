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
"""Tests for tensorflow_transform.gaussianization."""

import numpy as np
from tensorflow_transform import gaussianization
from tensorflow_transform import test_case

_MEAN_SCALE_SCALAR_TEST = dict(
    testcase_name='_scalar',
    h_params=np.array([0.1, 0.2], np.float32),
    expected_outputs=[
        np.float32(0.05540865005575452),
        np.float32(0.6932738015273474)
    ],
)

_MEAN_SCALE_ND_TEST = dict(
    testcase_name='_nd',
    h_params=np.array(
        [[[[0.0], [0.1], [0.5]], [[0.7], [0.8], [0.9]]],
         [[[0.0], [0.7], [0.6]], [[0.3], [0.2], [0.0]]]], np.float32),
    expected_outputs=[
        np.array([[[0.], [0.8865384], [0.19947124]],
                  [[-0.75989], [-1.4960338], [-3.5904799]]], np.float32),
        np.array([[[0.5641896], [1.4878997], [1.4943897]],
                  [[1.6034254], [2.1926064], [4.085859]]], np.float32)
    ],
)

_L_SKEWNESS_KURTOSIS_SCALAR_TEST = dict(
    testcase_name='_scalar',
    h_params=np.array([0.1, 0.2], np.float32),
    expected_outputs=[
        np.float32(0.05989154619056726),
        np.float32(0.21460719619685548)
    ],
)

_L_SKEWNESS_KURTOSIS_ND_TEST = dict(
    testcase_name='_nd',
    h_params=np.array(
        [[[[0.0], [0.1], [0.5]], [[0.7], [0.8], [0.9]]],
         [[[0.0], [0.7], [0.6]], [[0.3], [0.2], [0.0]]]], np.float32),
    expected_outputs=[
        np.array([[[0.], [0.5209037], [0.11905935]],
                  [[-0.4226278], [-0.6249933], [-0.833552]]], np.float32),
        np.array([[[0.12260159], [0.54675657], [0.5140212]],
                  [[0.55600286], [0.66664696], [0.81815743]]], np.float32),
    ],
)

_COMPUTE_TUKEY_H_PARAMS_REGULAR_TESTS = [dict(
    testcase_name='_regular_1',
    l_skewness_and_kurtosis=np.array(
        [0.05989154619056726, 0.21460719619685548], np.float32),
    expected_output=np.array([0.1, 0.2], np.float32),
), dict(
    testcase_name='_regular_2',
    l_skewness_and_kurtosis=np.array([0.1, 0.2], np.float32),
    expected_output=np.array([0.03056329, 0.20497137], np.float32)
), dict(
    testcase_name='_regular_3',
    l_skewness_and_kurtosis=np.array([0.8, 0.99], np.float32),
    expected_output=np.array([0.9635793, 0.99589026], np.float32)
), dict(
    testcase_name='_regular_4',
    l_skewness_and_kurtosis=np.array([0.6, 0.7], np.float32),
    expected_output=np.array([0.3535486, 0.82437974], np.float32)
)]

_COMPUTE_TUKEY_H_PARAMS_NEG_SKEWNESS_TEST = dict(
    testcase_name='_neg_skewness',
    l_skewness_and_kurtosis=np.array(
        [-0.05989154619056726, 0.21460719619685548], np.float32),
    expected_output=np.array([0.2, 0.1], np.float32)
)

_COMPUTE_TUKEY_H_PARAMS_PATOLOGICAL_TESTS = [dict(
    # For this test, the values of skewness and kurtosis are valid, but not
    # achievable by a Tukey HH distribution. The solution is the closest
    # possible.
    testcase_name='_patological',
    l_skewness_and_kurtosis=np.array(
        [0.7, 0.5], np.float32),
    expected_output=np.array([0.0, 0.65736556], np.float32)
), dict(
    testcase_name='_pat_invalid_skewness',
    l_skewness_and_kurtosis=np.array(
        [1.0, 0.5], np.float32),
    expected_output=np.array([0.0, 0.65736556], np.float32)
), dict(
    testcase_name='_pat_invalid_kurtosis',
    l_skewness_and_kurtosis=np.array(
        [0.5, 1.5], np.float32),
    expected_output=np.array(
        [00.9999859847861059, 0.9999950120303265], np.float32)
)]

_LAMBERT_W_SCALAR_TESTS = [dict(
    testcase_name='lambert_w_scalar_0',
    samples=np.float32(0.0),
    expected_output=np.float32(0.0)
), dict(
    testcase_name='lambert_w_scalar_small',
    samples=np.float32(1.0e-4),
    expected_output=np.float32(9.999000e-05)
), dict(
    testcase_name='lambert_w_scalar_e',
    samples=np.float32(np.exp(1.0)),
    expected_output=np.float32(1.0)
), dict(
    testcase_name='lambert_w_scalar_large',
    samples=np.float32(10.0 * np.exp(10.0)),
    expected_output=np.float32(10.0)
)]

_LAMBERT_W_ND_TESTS = [dict(
    testcase_name='lambert_w_1D',
    samples=np.linspace(0.0, 10, 8, dtype=np.float32),
    expected_output=np.array(
        [0., 0.70550971, 1.02506557, 1.24009733, 1.40379211, 1.53656406,
         1.6485427, 1.745528], np.float32)
), dict(
    testcase_name='lambert_w_3D',
    samples=np.linspace(0.0, 10, 8, dtype=np.float32).reshape((2, 4, 1)),
    expected_output=np.array(
        [0., 0.70550971, 1.02506557, 1.24009733, 1.40379211, 1.53656406,
         1.6485427, 1.745528], np.float32).reshape((2, 4, 1))
)]

_INVERSE_TUKEY_HH_SCALAR_TESTS = [dict(
    testcase_name='inverse_tukey_scalar_0',
    samples=np.float32(0.0),
    hl=np.float32(1.0),
    hr=np.float32(2.0),
    expected_output=np.float32(0.0)
), dict(
    testcase_name='inverse_tukey_small_positive',
    samples=np.float32(1.0e-4),
    hl=np.float32(1.0),
    hr=np.float32(2.0),
    expected_output=np.float32(1.0e-4)
), dict(
    testcase_name='inverse_tukey_small_negative',
    samples=np.float32(-1.0e-4),
    hl=np.float32(1.0),
    hr=np.float32(2.0),
    expected_output=np.float32(-1.0e-4)
), dict(
    testcase_name='inverse_tukey_large_positive',
    samples=np.float32(5.0 * np.exp(25.0)),
    hl=np.float32(1.0),
    hr=np.float32(2.0),
    expected_output=np.float32(5.0)
), dict(
    testcase_name='inverse_tukey_large_negative',
    samples=np.float32(-5.0 * np.exp(0.5 * 25.0)),
    hl=np.float32(1.0),
    hr=np.float32(2.0),
    expected_output=np.float32(-5.0)
)]


def _tukey_hh(x, hl, hr):
  return np.where(
      x > 0.0,
      x * np.exp(0.5 * hr * np.square(x)),
      x * np.exp(0.5 * hl * np.square(x)))

_INVERSE_TUKEY_HH_ND_TESTS = [dict(
    testcase_name='inverse_tukey_1D',
    samples=np.array(
        _tukey_hh(np.linspace(-5.0, 5.0, 20), 1.0, 2.0), np.float32),
    hl=np.float32(1.0),
    hr=np.float32(2.0),
    expected_output=np.linspace(-5.0, 5.0, 20, dtype=np.float32)
), dict(
    testcase_name='inverse_tukey_3D',
    samples=np.array(
        _tukey_hh(np.linspace(-5.0, 5.0, 100).reshape((10, 5, 2)),
                  np.linspace(1.0, 1.5, 10).reshape((1, 5, 2)),
                  np.linspace(2.0, 2.5, 10).reshape((1, 5, 2))), np.float32),
    hl=np.linspace(1.0, 1.5, 10, dtype=np.float32).reshape((1, 5, 2)),
    hr=np.linspace(2.0, 2.5, 10, dtype=np.float32).reshape((1, 5, 2)),
    expected_output=np.linspace(
        -5.0, 5.0, 100, dtype=np.float32).reshape((10, 5, 2))
)]


class GaussianizationTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      _MEAN_SCALE_SCALAR_TEST,
      _MEAN_SCALE_ND_TEST
  )
  def test_tukey_hh_l_mean_and_scale(self, h_params, expected_outputs):
    outputs = gaussianization.tukey_hh_l_mean_and_scale(h_params)
    self.assertEqual(len(outputs), len(expected_outputs))
    for output, expected_output in zip(outputs, expected_outputs):
      self.assertEqual(output.dtype, expected_output.dtype)
      self.assertAllEqual(output.shape, expected_output.shape)
      self.assertAllClose(output, expected_output)

  @test_case.named_parameters(
      _L_SKEWNESS_KURTOSIS_SCALAR_TEST,
      _L_SKEWNESS_KURTOSIS_ND_TEST
  )
  def test_tukey_hh_l_skewness_and_kurtosis(self, h_params, expected_outputs):
    outputs = gaussianization._tukey_hh_l_skewness_and_kurtosis(h_params)
    self.assertEqual(len(outputs), len(expected_outputs))
    for output, expected_output in zip(outputs, expected_outputs):
      self.assertEqual(output.dtype, expected_output.dtype)
      self.assertAllEqual(output.shape, expected_output.shape)
      self.assertAllClose(output, expected_output)

  @test_case.named_parameters(*(
      [_COMPUTE_TUKEY_H_PARAMS_NEG_SKEWNESS_TEST] +
      _COMPUTE_TUKEY_H_PARAMS_REGULAR_TESTS +
      _COMPUTE_TUKEY_H_PARAMS_PATOLOGICAL_TESTS))
  def test_compute_tukey_hh_params(
      self, l_skewness_and_kurtosis, expected_output):
    output = gaussianization.compute_tukey_hh_params(l_skewness_and_kurtosis)
    self.assertEqual(output.dtype, expected_output.dtype)
    self.assertAllEqual(output.shape, expected_output.shape)
    self.assertAllClose(output, expected_output)

  @test_case.named_parameters(*_LAMBERT_W_SCALAR_TESTS + _LAMBERT_W_ND_TESTS)
  def test_lambert_w(self, samples, expected_output):
    output = gaussianization.lambert_w(samples)
    self.assertEqual(output.dtype, expected_output.dtype)
    self.assertAllEqual(output.shape, expected_output.shape)
    self.assertAllClose(output, expected_output)

  @test_case.named_parameters(
      *_INVERSE_TUKEY_HH_SCALAR_TESTS + _INVERSE_TUKEY_HH_ND_TESTS)
  def test_inverse_tukey_hh(self, samples, hl, hr, expected_output):
    output = gaussianization.inverse_tukey_hh(samples, hl, hr)
    self.assertEqual(output.dtype, expected_output.dtype)
    self.assertAllEqual(output.shape, expected_output.shape)
    self.assertAllClose(output, expected_output)


if __name__ == '__main__':
  test_case.main()
