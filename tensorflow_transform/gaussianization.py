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
"""Utilities used to compute parameters for gaussianization."""

import numpy as np
import tensorflow as tf

# The expressions to compute the first L-moments from the parameters of the
# Tukey HH distribution are taken from:
# Todd C. Headrick, and Mohan D. Pant. "Characterizing Tukey h and
# hh-Distributions through L-Moments and the L-Correlation," ISRN Applied
# Mathematics, vol. 2012, 2012. doi:10.5402/2012/980153


def tukey_hh_l_mean_and_scale(h_params):
  """Computes L-mean and L-scale for a Tukey HH distribution.

  Args:
    h_params: An np.array with dimension 2 on the first axis. The slice
    h_params[0, ...] contains the left parameter of the distribution and
    h_params[1, ...] the right parameter. Each entry h must in 0 <= h < 1.

  Returns:
    The tuple (L_mean, L_scale) containing the first two L-moments for the
    given parameters. Each entry has the same shape as h_params, except for
    the first axis, which is removed.
  """
  one_div_sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)
  hl = h_params[0, ...]
  hr = h_params[1, ...]
  dtype = h_params.dtype
  l_1 = one_div_sqrt2pi * (1.0 / (hl - 1.0) + 1.0 / (1.0 - hr))
  l_2 = one_div_sqrt2pi * (
      (np.sqrt(2.0 - hl) + np.sqrt(2.0 - hr) - hl * np.sqrt(2.0 - hl) -
       hr * np.sqrt(2 - hr)) /
      ((hl - 1.0) * (hr - 1.0) * np.sqrt((hl - 2.0) * (hr - 2.0))))
  return (l_1.astype(dtype), l_2.astype(dtype))


def _tukey_hh_l_skewness_and_kurtosis(h_params):
  """Computes L-skewness and L-kurtosis for a Tukey HH distribution.

  Args:
    h_params: An np.array with dimension 2 on the first axis. The slice
    h_params[0, ...] contains the left parameter of the distribution and
    h_params[1, ...] the right parameter.

  Returns:
    The tuple (L_skewness, L_kurtosis) for the given parameters. Each entry
    has the same shape as h_params, except for the first axis, which is
    removed.
  """
  def skewness_num(h1, h2):
    return (12 * np.sqrt(2.0 - h1) * (h2 - 2.0) * (h2 - 1.0) *
            np.arctan(1.0 / np.sqrt(2.0 - h1)))

  def skewness_den(h):
    return h * np.sqrt(2 - h) - np.sqrt(2 - h)

  def kurtosis_den_part(h):
    return h * np.sqrt(2.0 - h) - np.sqrt(2.0 - h)

  hl = h_params[0, ...]
  hr = h_params[1, ...]
  dtype = h_params.dtype
  skewness = (skewness_num(hl, hr) -
              np.pi * (hl - hr) * (hl - 2.0) * (hr - 2.0) -
              skewness_num(hr, hl)) / (
                  2 * np.pi * np.sqrt((hl - 2.0) * (hr - 2.0)) *
                  (skewness_den(hl) + skewness_den(hr)))
  kurtosis_num_1 = (
      hr * np.sqrt((hl - 4.0) * (hl - 2.0) * (hl - 1.0) * (hr - 2.0)) -
      2.0 * np.sqrt((hl - 4.0) * (hl - 1.0)))
  kurtosis_num_2 = (hl * (hl - 3.0) * np.sqrt((hl - 4.0) * (hl - 1.0)) +
                    np.sqrt((hl - 4.0) * (hl - 2.0) * (hl - 1.0) * (hr - 2.0)))
  kurtosis_num_3 = (30.0 * (hl - 1.0) *
                    np.sqrt((hl - 4.0) * (hl - 2.0) * (hr - 2.0) / (hl - 1.0)) *
                    (hr - 1.0) * np.arctan(np.sqrt(1.0 + 2.0 / (hl - 4.0))))
  kurtosis_num_4 = (30.0 * (hl - 2) *
                    np.sqrt((hl - 4.0) * (hl - 1.0)) * (hl - 1.0) *
                    np.arctan(np.sqrt(1.0 + 2.0 / (hr - 4.0))))
  kurtosis_den = (np.pi * np.sqrt((4.0 - hl) * (2.0 - hl) * (1.0 - hl)) *
                  (kurtosis_den_part(hl) + kurtosis_den_part(hr)))
  kurtosis = (6.0 * np.pi * (kurtosis_num_1 - kurtosis_num_2) +
              kurtosis_num_3 + kurtosis_num_4) / kurtosis_den
  return (skewness.astype(dtype), kurtosis.astype(dtype))


def _binary_search(error_fn, low_value, high_value):
  """Binary search for a function given start and end interval.

  This is a simple binary search over the values of the function error_fn given
  the interval [low_value, high_value]. We expect that the starting condition is
  error_fn(low_value) < 0 and error_fn(high_value) > 0 and we bisect the
  interval until the exit conditions are met. The result is the final interval
  [low_value, high_value] that is normally much smaller than the initial one,
  but still satisfying the starting condition.

  Args:
    error_fn: Function mapping values to errors.
    low_value: Lower interval endpoint. We expect f(low_value) < 0.
    high_value: Higher interval endpoint. We expect f(high_value) > 0.

  Returns:
    The final interval endpoints (low_value, high_value) after the sequence of
    bisections.
  """
  # Exit conditions.
  stop_iter_step = 10  # Max number of iterations.
  stop_error_step = 1e-6  # Minimum function variation.
  stop_value_step = 1e-6  # Minimum variable variation.

  current_iter = 0
  while True:
    current_value = (low_value + high_value) / 2.0
    current_error = error_fn(current_value)
    if current_error < 0.0:
      low_value = current_value
    else:
      high_value = current_value
    current_iter += 1
    if (current_iter > stop_iter_step or
        np.abs(current_error) < stop_error_step or
        high_value - low_value < stop_value_step):
      break
  return low_value, high_value


def _params_to_errors(h, delta_h, l_skewness_and_kurtosis):
  """Maps parameters to errors on L-skewness and L-kurtosis.

  Args:
    h: Value of right parameter of the Tukey HH distribution.
    delta_h: Different between right and left parameter of the Tukey HH
      distribution.
    l_skewness_and_kurtosis: np.array containing the target values of
      L-skewness and L-kurtosis.

  Returns:
    An np.array containing the difference between the values of L-skewness and
    L-kurtosis corresponding to the parameters hl = h - delta_h, hr =h and the
    target values.
  """
  dtype = l_skewness_and_kurtosis.dtype
  h_params = np.array([h - delta_h, h], dtype=dtype)
  current_l_skewness_and_kurtosis = np.array(
      _tukey_hh_l_skewness_and_kurtosis(h_params), dtype=dtype)
  return current_l_skewness_and_kurtosis - l_skewness_and_kurtosis


def compute_tukey_hh_params(l_skewness_and_kurtosis):
  """Computes the H paramesters of a Tukey HH distribution.

  Given the L-skewness and L-kurtosis of a Tukey HH distribution we compute
  the H parameters of the distribution.

  Args:
    l_skewness_and_kurtosis: A np.array with shape (2,) containing L-skewness
    and L-kurtosis.

  Returns:
    An np.array with the same type and shape of the argument containing the
    left and right H parameters of the distribution.
  """

  # Exit conditions for the search loop.
  stop_iter_step = 20  # Max number of iteration for the search loop.
  stop_error_step = 1e-6  # Minimum function variation.
  stop_value_step = 1e-6  # Minimum variable variation.

  dtype = l_skewness_and_kurtosis.dtype

  # Returns zero parameters (i.e. treat as gaussian) if L-kurtosis is smaller
  # than for a gaussian.

  result = np.zeros_like(l_skewness_and_kurtosis)
  if l_skewness_and_kurtosis[1] < 0.1226017:
    return result

  # If L-skewness is negative, swap the parameters.

  swap_params = False
  if l_skewness_and_kurtosis[0] < 0.0:
    l_skewness_and_kurtosis[0] = -l_skewness_and_kurtosis[0]
    swap_params = True

  l_skewness_and_kurtosis[1] = np.minimum(
      l_skewness_and_kurtosis[1], 1.0 - 1.0e-5)

  # If L-skewness is zero, left and right parameters are equal and there is a
  # a closed form to compute them from L-kurtosis. We start from this value
  # and then change them to match simultaneously L-skeweness and L-kurtosis.
  # For that, we parametrize the search space with the array
  # [h_rigth, h_right - h_left], i.e. the value of the right parameter and the
  # difference right minus left paramerters. In the search iteration, we
  # alternate between updates on the first and the second entry of the search
  # parameters.

  initial_h = 3.0 - 1.0 / np.cos(
      np.pi / 15.0 * (l_skewness_and_kurtosis[1] - 6.0))
  search_params = np.array([initial_h, 0.0], dtype=dtype)

  # Current lower and upper bounds for the search parameters.

  min_search_params = np.array([initial_h, 0.0], dtype=dtype)
  max_search_params = np.array([1.0 - 1.0e-7, initial_h], dtype=dtype)

  current_iter = 0
  previous_search_params = np.zeros_like(search_params)
  while current_iter < stop_iter_step:
    # Search for L-skewness at constant h. Increase delta_h.
    error_skewness = lambda x: _params_to_errors(  # pylint: disable=g-long-lambda
        search_params[0], x, l_skewness_and_kurtosis)[0]
    if error_skewness(max_search_params[1]) > 0.0:
      low_delta_h, high_delta_h = _binary_search(
          error_skewness, min_search_params[1], max_search_params[1])
      search_params[1] = high_delta_h
      max_search_params[1] = high_delta_h  # The new delta is an upperbound.
      upperbound_delta_found = True
    else:
      search_params[1] = max_search_params[1]
      min_search_params[1] = max_search_params[1]  # No solution: lowerbound.
      upperbound_delta_found = False

    # Search for L-kurtosis at constant possibly overestimated delta.
    error_kurtosis = lambda x: _params_to_errors(  # pylint: disable=g-long-lambda
        x, search_params[1], l_skewness_and_kurtosis)[1]
    low_h, high_h = _binary_search(
        error_kurtosis, min_search_params[0], max_search_params[0])
    if upperbound_delta_found:
      search_params[0] = high_h
      max_search_params[0] = high_h   # Delta overestimated: upperbound for h.
    else:
      search_params[0] = low_h
      min_search_params[0] = low_h   # Delta underestimated: lowerbound for h.
      max_search_params[1] = low_h  # Delta not found, search on full range.

    if upperbound_delta_found:  # If not found, we repeat the first 2 steps.
      # Otherwise, Search for delta at constant overestimated h.
      error_skewness = lambda x: _params_to_errors(  # pylint: disable=g-long-lambda
          search_params[0], x, l_skewness_and_kurtosis)[0]
      low_delta_h, high_delta_h = _binary_search(
          error_skewness, min_search_params[1], max_search_params[1])
      search_params[1] = low_delta_h
      min_search_params[1] = low_delta_h

      # Search for h at constant delta.
      error_kurtosis = lambda x: _params_to_errors(  # pylint: disable=g-long-lambda
          x, search_params[1], l_skewness_and_kurtosis)[1]
      low_h, high_h = _binary_search(
          error_kurtosis, min_search_params[0], max_search_params[0])
      search_params[0] = low_h
      min_search_params[0] = low_h

    current_error = _params_to_errors(
        search_params[0], search_params[1], l_skewness_and_kurtosis)
    delta_search_params = search_params - previous_search_params
    current_iter += 1
    previous_search_params = search_params.copy()
    if (np.all(np.abs(current_error) < stop_error_step) or
        np.all(np.abs(delta_search_params) < stop_value_step)):
      break

  result[0] = search_params[0] - search_params[1]
  result[1] = search_params[0]
  if swap_params:
    result = result[::-1]
  return result


def lambert_w(x):
  """Computes the Lambert W function of a `Tensor`.

  Computes the principal branch of the Lambert W function, i.e. the value w such
  that w * exp(w) = x for a a given x. For the principal branch, x must be real
  x >= -1 / e, and w >= -1.

  Args:
    x: A `Tensor` containing the values for which the principal branch of
      the Lambert W function is computed.

  Returns:
    A `Tensor` with the same shape and dtype as x containing the value of the
    Lambert W function.
  """
  dtype = x.dtype
  e = tf.constant(np.exp(1.0), dtype)
  inv_e = tf.constant(np.exp(-1.0), dtype)
  s = (np.exp(1) - 1.0) / (np.exp(2) - 1.0)
  slope = tf.constant(s, dtype)
  c = tf.constant(1 / np.exp(1) * (1 - s), dtype)
  log_s = tf.math.log(x)
  w_init = tf.where(
      x < inv_e,
      x,
      tf.where(x < e,
               slope * x + c,
               (log_s + (1.0 / log_s - 1.0) * tf.math.log(log_s))))

  def newton_update(count, w):
    expw = tf.math.exp(w)
    wexpw = w * expw
    return count + 1, w - (wexpw - x) / (expw + wexpw)

  count = tf.constant(0, tf.int32)
  num_iter = tf.constant(8)
  (unused_final_count, w) = tf.while_loop(
      lambda count, w: tf.less(count, num_iter),
      newton_update,
      [count, w_init])
  return w


def inverse_tukey_hh(x, hl, hr):
  """Compute the inverse of the Tukey HH function.

  The Tukey HH function transforms a standard Gaussian distribution into the
  Tukey HH distribution and it's defined as:

  x = u * exp(hl * u ^ 2) for u < 0 and x = u * exp(hr * u ^ 2) for u >= 0.

  Given the values of x, this function computes the corresponding values of u.

  Args:
    x: The input `Tensor`.
    hl: The "left" parameter of the distribution. It must have the same dtype
      and shape of x (or a broadcastable shape) or be a scalar.
    hr: The "right" parameter of the distribution. It must have the same dtype
      and shape of x (or a broadcastable shape) or be a scalar.

  Returns:
    The inverse of the Tukey HH function.
  """
  def one_side(x, h):
    h_x_square = tf.multiply(h, tf.square(x))
    return tf.where(
        # Prevents the 0 / 0 form for small values of x..
        tf.less(h_x_square, 1.0e-7),
        x,  # The error is < 1e-14 for this case.
        tf.sqrt(tf.divide(lambert_w(h_x_square), h)))

  return tf.where(tf.less(x, 0.0), -one_side(-x, hl), one_side(x, hr))
