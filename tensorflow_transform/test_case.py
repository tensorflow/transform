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
"""Library for Tensorflow Transform test cases."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect


from absl.testing import parameterized

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import test_util

main = tf.test.main

named_parameters = parameterized.named_parameters


def parameters(*testcases):
  """like parameterized.parameters but tests show arg names.

  Only works for class methods without *args or **kwargs.

  Args:
    *testcases: The input to parameterized.parameters().

  Returns:
    A wrapper function which passes the arguments through as a dictionary.
  """

  def wrapper(fn):
    """Constructs and returns the arguments as a dictionary."""
    arg_names = inspect.getargspec(fn).args
    if arg_names[0] != 'self':
      raise ValueError(
          'First argument to test is expected to be "self", but is {}'.format(
              arg_names[0]))
    arg_names = arg_names[1:]

    def to_arg_dict(testcase):
      testcase = tuple(testcase)
      if len(testcase) != len(arg_names):
        raise ValueError(
            'The number of arguments to parameterized test do not match the '
            'number of expected arguments: {} != {}, arguments: {}, names: {}'.
            format(len(testcase), len(arg_names), testcase, arg_names))
      return dict(zip(arg_names, testcase))

    testcases_with_names = [to_arg_dict(testcase) for testcase in testcases]
    return parameterized.parameters(*testcases_with_names)(fn)

  return wrapper


class TransformTestCase(parameterized.TestCase, test_util.TensorFlowTestCase):
  """Base test class for testing tf-transform code."""

  # Display context for failing rows in data assertions.
  longMessage = True  # pylint: disable=invalid-name

  def assertDataCloseOrEqual(self, a_data, b_data):
    """Assert two datasets contain nearly equal values.

    Args:
      a_data: a sequence of dicts whose values are
              either strings, lists of strings, numeric types or a pair of
              those.
      b_data: same types as a_data

    Raises:
      AssertionError: if the two datasets are not the same.
    """
    a_data, b_data = self._sorted_data(a_data), self._sorted_data(b_data)
    self.assertEqual(
        len(a_data), len(b_data), 'len(%r) != len(%r)' % (a_data, b_data))
    for i, (a_row, b_row) in enumerate(zip(a_data, b_data)):
      self.assertCountEqual(a_row.keys(), b_row.keys(), msg='Row %d' % i)
      for key in a_row.keys():
        a_value = a_row[key]
        b_value = b_row[key]
        msg = 'Row %d, key %s' % (i, key)
        if isinstance(a_value, tuple):
          self._assertValuesCloseOrEqual(a_value[0], b_value[0], msg=msg)
          self._assertValuesCloseOrEqual(a_value[1], b_value[1], msg=msg)
        else:
          self._assertValuesCloseOrEqual(a_value, b_value, msg=msg)

  def _assertValuesCloseOrEqual(self, a_value, b_value, msg=None):
    try:
      if (isinstance(a_value, (six.binary_type, six.text_type)) or
          isinstance(a_value, list) and a_value and
          isinstance(a_value[0], (six.binary_type, six.text_type)) or
          isinstance(a_value, np.ndarray) and a_value.dtype == np.object):
        self.assertAllEqual(a_value, b_value)
      else:
        self.assertAllClose(a_value, b_value)
    except (AssertionError, TypeError) as e:
      if msg:
        e.args = ((e.args[0] + ' : ' + msg,) + e.args[1:])
      raise

  def WriteRenderedDotFile(self, dot_string, output_file=None):
    tf.logging.info('Writing a rendered dot file is not yet supported.')

  def _numpy_arrays_to_lists(self, maybe_arrays):
    return [
        x.tolist() if isinstance(x, np.ndarray) else x for x in maybe_arrays]

  def _sorted_dicts(self, list_of_dicts):
    # Sorts dicts by their unordered (key, value) pairs.
    return sorted(list_of_dicts, key=lambda d: sorted(d.items()))

  def _sorted_data(self, list_of_dicts_of_arrays):
    list_of_values = [
        self._numpy_arrays_to_lists(d.values()) for d in list_of_dicts_of_arrays
    ]
    list_of_keys = [d.keys() for d in list_of_dicts_of_arrays]
    unsorted_dict_list = [
        dict(zip(a, b)) for a, b in zip(list_of_keys, list_of_values)
    ]
    return self._sorted_dicts(unsorted_dict_list)
