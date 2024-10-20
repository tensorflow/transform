# Copyright 2023 Google Inc. All Rights Reserved.
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
"""Imports keras 2."""
import os

from absl import logging
import tensorflow as tf

if 'TF_USE_LEGACY_KERAS' not in os.environ:
  # Make sure we are using Keras 2.
  os.environ['TF_USE_LEGACY_KERAS'] = '1'
elif os.environ['TF_USE_LEGACY_KERAS'] not in ('true', 'True', '1'):
  logging.warning(
      'TF_USE_LEGACY_KERAS is set to %s, which will not use Keras 2. Tensorflow'
      ' Transform is only compatible with Keras 2. Please set'
      ' TF_USE_LEGACY_KERAS=1.',
      os.environ['TF_USE_LEGACY_KERAS'],
  )

version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
  # `tf.keras` points to `keras 3`, so use `tf_keras` package
  try:
    import tf_keras  # pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    raise ImportError(  # pylint: disable=raise-missing-from
        'Keras 2 requires the `tf_keras` package.'
        'Please install it with `pip install tf_keras`.'
    ) from None
else:
  tf_keras = tf.keras  # Keras 2
