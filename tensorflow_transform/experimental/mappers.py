# Copyright 2022 Google Inc. All Rights Reserved.
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
"""Experimental functions that transform features based on full-pass analysis.

The core tf.Transform API requires a user to construct a
"preprocessing function" that accepts and returns `Tensor`s.  This function is
built by composing regular functions built from TensorFlow ops, as well as
special functions we refer to as `Analyzer`s.  `Analyzer`s behave similarly to
TensorFlow ops but require a full pass over the whole dataset to compute their
output value.  The analyzers are defined in analyzers.py, while this module
provides helper functions that call analyzers and then use the results of the
anaylzers to transform the original data.

The user-defined preprocessing function should accept and return `Tensor`s that
are batches from the dataset, whose batch size may vary.
"""

from typing import Any, Optional

import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import mappers
from tensorflow_transform.experimental import analyzers as experimental_analyzers


@common.log_api_use(common.MAPPER_COLLECTION)
def compute_and_apply_approximate_vocabulary(
    x: common_types.ConsistentTensorType,
    default_value: Any = -1,
    top_k: Optional[int] = None,
    num_oov_buckets: int = 0,
    vocab_filename: Optional[str] = None,
    weights: Optional[tf.Tensor] = None,
    file_format: common_types.VocabularyFileFormatType = analyzers
    .DEFAULT_VOCABULARY_FILE_FORMAT,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Generates an approximate vocabulary for `x` and maps it to an integer.

  Args:
    x: A `Tensor` or `CompositeTensor` of type tf.string or tf.int[8|16|32|64].
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.
    vocab_filename: The file name for the vocabulary file. If None, a name based
      on the scope name in the context of this graph will be used as the file
      name. If not None, should be unique within a given preprocessing function.
      NOTE in order to make your pipelines resilient to implementation details
      please set `vocab_filename` when you are using the vocab_filename on a
      downstream component.
    weights: (Optional) Weights `Tensor` for the vocabulary. It must have the
      same shape as x.
    file_format: (Optional) A str. The format of the resulting vocabulary file.
      Accepted formats are: 'tfrecord_gzip', 'text'. 'tfrecord_gzip' requires
        tensorflow>=2.4. The default value is 'text'.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` where each string value is mapped to an
    integer. Each unique string value that appears in the vocabulary
    is mapped to a different integer and integers are consecutive starting from
    zero. String value not in the vocabulary is assigned default_value.
    Alternatively, if num_oov_buckets is specified, out of vocabulary strings
    are hashed to values in [vocab_size, vocab_size + num_oov_buckets) for an
    overall range of [0, vocab_size + num_oov_buckets).

  Raises:
    ValueError: If `top_k` is negative.
      If `file_format` is not in the list of allowed formats.
      If x.dtype is not string or integral.
  """
  with tf.compat.v1.name_scope(name,
                               'compute_and_apply_approximate_vocabulary'):
    deferred_vocab_and_filename = experimental_analyzers.approximate_vocabulary(
        x=x,
        top_k=top_k,
        vocab_filename=vocab_filename,
        weights=weights,
        file_format=file_format,
        name=name)
    return mappers.apply_vocabulary(
        x,
        deferred_vocab_and_filename,
        default_value,
        num_oov_buckets,
        file_format=file_format)
