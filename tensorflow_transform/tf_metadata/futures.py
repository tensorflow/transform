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
"""Allow setting metadata values as Futures, and filling them in later."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import six


class Future(object):

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name

  def __repr__(self):
    return "{}({})".format(self.__class__.__name__, repr(self.__dict__))


def _substitute_futures(obj, name_to_value, started=None):
  """Substitute `Future`s hierarchically within the given object or collection.

  Args:
    obj: an object, dict, list, or set potentially containing `Future`s.
    name_to_value: a dict of Future name to value.
    started: a set of objects that have already been visited, to avoid cycles.

  Returns:
    A list of remaining Futures that were not substituted.
  """
  if started is None:
    started = set()

  if isinstance(obj, dict):
    iterable = six.iteritems(obj)
    def subst_fn(key, name):
      obj[key] = name_to_value[name]
  elif isinstance(obj, list):
    iterable = enumerate(obj)
    def subst_fn(key, name):
      obj[key] = name_to_value[name]
  elif isinstance(obj, set):
    iterable = {entry: entry for entry in obj}
    def subst_fn(key, name):
      obj.remove(key)
      obj.add(name_to_value[name])
  else:
    if obj in started:
      return
    started.add(obj)
    iterable = six.iteritems(obj.__dict__)
    def subst_fn(key, name):
      obj.__setattr__(key, name_to_value[name])

  return [future
          for k, v in iterable
          for future in _maybe_subst(k, v, name_to_value, started, subst_fn)]


def _maybe_subst(k, v, name_to_value, started, subst_fn):
  if isinstance(v, Future):
    if v.name in name_to_value:
      subst_fn(k, v.name)
    else:
      return [v]
  if isinstance(v, (FutureContent, dict, list, set)):
    return _substitute_futures(v, name_to_value, started)
  return []


class FutureContent(object):
  """An object that may contain some fields that are Futures."""

  __metaclass__ = abc.ABCMeta

  def substitute_futures(self, name_to_value, started=None):
    return _substitute_futures(self, name_to_value, started)

  def all_futures_resolved(self, started=None):
    """Determine whether any Futures remain to be substituted."""
    return not _substitute_futures(self, {}, started)
