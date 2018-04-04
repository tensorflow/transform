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
