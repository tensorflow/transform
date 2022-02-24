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
"""Init module for TF.Transform."""

# pylint: disable=wildcard-import
from tensorflow_transform import coders
from tensorflow_transform import experimental
from tensorflow_transform.analyzers import *
from tensorflow_transform.annotators import *
from tensorflow_transform.inspect_preprocessing_fn import *
from tensorflow_transform.mappers import *
from tensorflow_transform.output_wrapper import TFTransformOutput
from tensorflow_transform.output_wrapper import TransformFeaturesLayer
from tensorflow_transform.py_func.api import apply_pyfunc
# pylint: enable=wildcard-import

# Import version string.
from tensorflow_transform.version import __version__

# TF 2.6 split support for filesystems such as Amazon S3 out to the
# `tensorflow_io` package. Hence, this import is needed wherever we touch the
# filesystem.
try:
  import tensorflow_io as _  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  pass
