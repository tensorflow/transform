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
"""Utilities for information-theoretic preprocessing algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np


def calculate_expected_mutual_information_per_label(n, x, y_j):
  """Calculates the expected mutual information (EMI) of a feature and a label.

    The EMI reflects the MI expected by chance, and is used to compute adjusted
    mutual information. See www.wikipedia.org/wiki/Adjusted_mutual_information.

    EMI(x, y) = sum_{n_ij = max(0, x_i + y_j - n) to min(x_i, y_j)} (
      n_ij / n * log2((n * n_ij / (x_i * y_j))
      * ((x_i! * y_j! * (n - x_i)! * (n - y_j)!) /
      (n! * n_ij! * (x_i - n_ij)! * (y_j - n_ij)! * (n - x_i - y_j + n_ij)!)))
    where n_ij is the joint count of feature and label, x_i is the count for
    feature x, y_j is the count for label y, and n represents total count.

    Note: In the paper, expected mutual information is calculated by summing
    over both i and j, but here we don't count the contribution of the case i=0
    (where the feature is not present), and this is consistent with how mutual
    information is computed.

  Args:
    n: The sum of weights for all features.
    x: The sum of weights for the feature whose expected mutual information is
      computed.
    y_j: The sum of weights for positive (or negative) labels for all features.

  Returns:
    Calculated expected mutual information.
  """
  coefficient = (-np.log2(x) - np.log2(y_j) + np.log2(n))
  sum_probability = 0.0
  partial_result = 0.0
  for n_j, p_j in _hypergeometric_pmf(n, x, y_j):
    if n_j != 0:
      partial_result += n_j * (coefficient + np.log2(n_j)) * p_j
    sum_probability += p_j
  # With approximate calculations for log2(x) and exp2(x) with large x, we need
  # a correction to the probablity approximation, so we divide by the sum of the
  # probabilities.
  return partial_result / sum_probability


def _hypergeometric_pmf(n, x, y_j):
  """Probablity for expectation computation under hypergeometric distribution.

  Args:
    n: The sum of weights for all features.
    x: The sum of weights for the feature whose expected mutual information is
      computed.
    y_j: The sum of weights for positive (or negative) labels for all features.

  Yields:
    Calculated coefficient, numerator and denominator for hypergeometric
    distribution.
  """
  start = int(max(0, n - (n - x) - (n - y_j)))
  end = int(min(x, y_j))
  # Use log factorial to preserve calculation precision.
  numerator = (
      _logfactorial(x) + _logfactorial(y_j) + _logfactorial(n - x) +
      _logfactorial(n - y_j))
  denominator = (
      _logfactorial(n) + _logfactorial(start) + _logfactorial(x - start) +
      _logfactorial(y_j - start) + _logfactorial(n - x - y_j + start))
  for n_j in range(start, end + 1):
    p_j = np.exp(numerator - denominator)
    denominator += (
        np.log(n_j + 1) - np.log(x - n_j) - np.log(y_j - n_j) +
        np.log(n - x - y_j + n_j + 1))
    yield n_j, p_j


def _logfactorial(n):
  """Calculate natural logarithm of n!."""
  return math.lgamma(n + 1)
