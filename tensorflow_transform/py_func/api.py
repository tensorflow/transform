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
"""Public API for using py_funcs in TFTransform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_transform.py_func import pyfunc_helper


def apply_pyfunc(func, Tout, stateful=True, name=None, *args):  # pylint: disable=invalid-name
  """Applies a python function to some `Tensor`s.

  Applies a python function to some `Tensor`s given by the argument list. The
  number of arguments should match the number of inputs to the function.

  This function is for using inside a preprocessing_fn.  It is a wrapper around
  `tf.py_func`.  A function added this way can run in Transform, and during
  training when the graph is imported using the `transform_raw_features` method
  of the `TFTransformOutput` class.  However if the resulting training graph is
  serialized and deserialized, then the `tf.py_func` op will not work and will
  cause an error.  This means that TensorFlow Serving will not be able to serve
  this graph.

  The underlying reason for this limited support is that `tf.py_func` ops were
  not designed to be serialized since they contain a reference to arbitrary
  Python functions. This function pickles those functions and including them in
  the graph, and `transform_raw_features` similarly unpickles the functions.
  But unpickling requires a Python environment, so there it's not possible to
  provide support in non-Python languages for loading such ops.  Therefore
  loading these ops in libraries such as TensorFlow Serving is not supported.

  Args:
    func: A Python function, which accepts a list of NumPy `ndarray` objects
      having element types that match the corresponding `tf.Tensor` objects
      in `*args`, and returns a list of `ndarray` objects (or a single
      `ndarray`) having element types that match the corresponding values
      in `Tout`.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) If True, the function should be considered stateful.
      If a function is stateless, when given the same input it will return the
      same output and have no observable side effects. Optimizations such as
      common subexpression elimination are only performed on stateless
      operations.
    name: A name for the operation (optional).
    *args: The list of `Tensor`s to apply the arguments to.
  Returns:
    A `Tensor` representing the application of the function.
  """
  return pyfunc_helper.insert_pyfunc(func, Tout, stateful, name, *args)
