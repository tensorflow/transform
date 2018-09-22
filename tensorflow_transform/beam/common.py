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
"""Constants and types shared by tf.Transform Beam package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid


import apache_beam as beam
from apache_beam.typehints import Union
from six import integer_types
from six import string_types
import tensorflow as tf

NUMERIC_TYPE = Union[float, Union[integer_types]]
PRIMITIVE_TYPE = Union[NUMERIC_TYPE, Union[string_types]]

METRICS_NAMESPACE = 'tfx.Transform'

_DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER = {
    # We rely on Beam to manage concurrency, i.e. we expect it to run one
    # session per CPU--so we don't want to proliferate TF threads.
    # Nonetheless we provide 4 threads per session for TF ops, 2 inter-
    # and 2 intra-thread.  In many cases only 2 of these will be runnable
    # at any given time.  This approach oversubscribes a bit to make sure
    # the CPUs are really saturated.
    #
    beam.runners.DataflowRunner:
        tf.ConfigProto(
            use_per_session_threads=True,
            inter_op_parallelism_threads=2,
            intra_op_parallelism_threads=2).SerializeToString(),

}


def _maybe_deserialize_tf_config(serialized_tf_config):
  if serialized_tf_config is None:
    return None

  result = tf.ConfigProto()
  result.ParseFromString(serialized_tf_config)
  return result


def make_unique_temp_dir(base_temp_dir):
  """Create path to a unique temp dir from given base temp dir."""
  return os.path.join(base_temp_dir, uuid.uuid4().hex)


PTRANSFORM_BY_ATTRIBUTES_CLASS = {}


def register_ptransform(attributes_class):
  """Decorator to register a PTransform as the implementation for an analyzer.

  Note that this PTransform may be called multiple times, but with unique
  attributes, so it should implement default_label to be unique given
  attributes.

  This function is used to define implementations of the analyzers defined in
  attributes_classes.py

  Args:
    attributes_class: The class of attributes that is being registered.

  Returns:
    A class decorator that registers a PTransform as an implementation of the
        generalized op for that type.
  """

  def register(ptransform_class):
    assert attributes_class not in PTRANSFORM_BY_ATTRIBUTES_CLASS
    PTRANSFORM_BY_ATTRIBUTES_CLASS[attributes_class] = ptransform_class
    return ptransform_class

  return register


def lookup_registered_ptransform(attributes):
  try:
    return PTRANSFORM_BY_ATTRIBUTES_CLASS[attributes.__class__]
  except KeyError:
    raise ValueError('No implementation registered for {}'.format(
        attributes.__class__))
