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
"""Constants and types shared by tf.Transform package."""

import collections
import contextlib
import functools
from typing import Any, Callable, Generator

import tensorflow as tf

from tensorflow.python.util import tf_decorator  # pylint: disable=g-direct-tensorflow-import

ANALYZER_COLLECTION = 'tft_analyzer_use'
MAPPER_COLLECTION = 'tft_mapper_use'

ANNOTATION_PREFIX_URL = 'type.googleapis.com'

# TODO(b/132098015): Schema annotations aren't yet supported in OSS builds.
try:
  from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top, unused-import
  IS_ANNOTATIONS_PB_AVAILABLE = True
except ImportError:
  IS_ANNOTATIONS_PB_AVAILABLE = False

_in_logging_context = False


@contextlib.contextmanager
def logging_context() -> Generator[None, None, None]:
  global _in_logging_context
  _in_logging_context = True
  try:
    yield
  finally:
    _in_logging_context = False


def log_api_use(
    collection_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Creates a decorator that logs function calls in the tensorflow graph."""

  def decorator(fn):
    """Logs function calls in a tensorflow graph collection."""

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
      if not _in_logging_context:
        with logging_context():
          graph = tf.compat.v1.get_default_graph()
          # Collection is a list that contains a single Counter of {name: count}
          # Note: We aggregate counts of function calls instead having one
          # collection item per call, since TFT users can use an arbitrarily
          # large number of analyzers and mappers and we don't want the graph
          # to get too big.
          # TODO(rachelim): Make this collection serializable so it can be added
          # to the SavedModel.
          collection = graph.get_collection_ref(collection_name)
          if not collection:
            collection.append(collections.Counter())
          collection[0][fn.__name__] += 1
          return fn(*args, **kwargs)
      else:
        return fn(*args, **kwargs)

    # We use tf_decorator here so that TF can correctly introspect into
    # functions for docstring generation.
    return tf_decorator.make_decorator(fn, wrapped_fn)

  return decorator
