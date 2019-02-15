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
"""Utility functions to use py_funcs in tf.transform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import dill
import tensorflow as tf

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops

_PYFUNC_COLLECTION_KEY = 'pyfuncs'


class _PyFuncDef(collections.namedtuple('_PyFuncDef', ['token', 'func'])):
  """An internal wrapper around tuple(token, func).

  The main purpose of this class is to provides the two methods:
  `from_proto` and `to_proto` that enable storing tuple objects in the graph's
  collections as proto objects.
  """

  @staticmethod
  def from_proto(attr_value, import_scope=None):
    del import_scope  # Unused
    return dill.loads(attr_value.s)

  @staticmethod
  def from_proto_string(proto_str, import_scope=None):
    del import_scope  # Unused
    attr_value = attr_value_pb2.AttrValue()
    attr_value.ParseFromString(proto_str)
    return _PyFuncDef.from_proto(attr_value)

  def to_proto(self, export_scope=None):
    del export_scope  # Unused
    result = attr_value_pb2.AttrValue()
    result.s = dill.dumps(self)
    return result

# Register the pyfuncs collection to use `AttrValue` proto type.
# The proto object stored in the graph collection will contain the pickled value
# of a `_PyFuncDef` object as a string in its `s` field.
# Note that `AttrValue` is used here only as a convenient placeholder for a
# string, and does not represent the actual attributes of an `op` as in the
# usual case.
ops.register_proto_function(_PYFUNC_COLLECTION_KEY,
                            proto_type=attr_value_pb2.AttrValue,
                            to_proto=_PyFuncDef.to_proto,
                            from_proto=_PyFuncDef.from_proto)


def insert_pyfunc(func, Tout, stateful, name, *args):  # pylint: disable=invalid-name
  """Calls tf.py_func and inserts the `func` in the internal registry."""
  result = tf.py_func(func, inp=list(args),
                      Tout=Tout, stateful=stateful, name=name)

  token = result.op.node_def.attr['token'].s
  tf.add_to_collection(_PYFUNC_COLLECTION_KEY, _PyFuncDef(token, func))
  return result


def register_pyfuncs_from_saved_transform(graph, meta_graph):
  """Registers `py_func`s in the MetaGraphDef.

  Takes the picked `py_func`s stored in the MetaGraphDef and adds them to the
  graph.  Registered `py_func`s are referred to internally by the token
  attribute of the `py_func` op.  We first create some arbitrary ops which
  are not used, but which result in the pickled functions stored in the
  MetaGraphDef being registered.  We then take the tokens of these newly
  registered functions, and remap the tokens in the MetaGraphDef to contain
  the new tokens for each function (this remapping is required since we cannot
  specify what token should be used to register a function).

  Args:
    graph: The tf.Graph into which the meta_graph_def will be imported.
    meta_graph: The MetaGraphDef containing the `py_func`s.  All the `py_func`
       ops in the graph will be modified in-place to have their token point to
       the newly regsitered function.
  """
  if _PYFUNC_COLLECTION_KEY not in meta_graph.collection_def:
    return

  # TODO(b/35929054) to enable it in TF itself. Once supported,
  # we should refactor this code to remove extra work for pickling and
  # re-registering of the py_funcs.
  pyfuncs_collection = meta_graph.collection_def[_PYFUNC_COLLECTION_KEY]

  new_tokens_by_old_token = {}
  with graph.as_default():
    for func_def_str in pyfuncs_collection.bytes_list.value:
      func_def = _PyFuncDef.from_proto_string(func_def_str)
      # Re-insert the original python function into the default graph.
      # The operation itself in the graph does not matter (hence the dummy
      # values for name, Tout, and stateful). This is done only to reinsert
      # the function body in the internal TF's function registry.
      # TODO(b/123241062): We should even remove this op from the graph if
      # possible.
      func_temp_name = func_def.token + b'_temp'
      output_tensor = insert_pyfunc(
          func_def.func, tf.float32, False, func_temp_name)
      # Store the token associated with the function associated with the call
      # to tf.py_func.
      token = output_tensor.op.get_attr('token')
      new_tokens_by_old_token[func_def.token] = token

  # Swap the old token stored for the function with the new one, if there are
  # any tokens to change.
  if new_tokens_by_old_token:
    for node in meta_graph.graph_def.node:
      if node.op == 'PyFunc' or node.op == 'PyFuncStateless':
        assert node.attr['token'].s in new_tokens_by_old_token, (
            'Function: %r was not registered' % node.name)
        node.attr['token'].s = new_tokens_by_old_token[node.attr['token'].s]
