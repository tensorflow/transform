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
"""Deprecated functionality required in previous version of tf.Transform."""

# GOOGLE-INITIALIZATION

from tensorflow.python.util import deprecation


@deprecation.deprecated(
    None,
    'apply_function is no longer needed.  `apply_function(fn, *args)` is now '
    'equvalent to `fn(*args)`')
def apply_function(fn, *args):
  """Deprecated function, equivalent to fn(*args).

  In previous versions of tf.Transform, it was necessary to wrap function
  application in `apply_function`, that is call apply_function(fn, *args)
  instead of calling fn(*args) directly.  This was necessary due to limitations
  in the ability of tf.Transform to inspect the TensorFlow graph.  These
  limitations no longer apply so apply_function is no longer needed.

  Args:
    fn: The function to apply.
    *args: The arguments to apply `fn` to.

  Returns:
    The results of applying fn.
  """
  return fn(*args)
