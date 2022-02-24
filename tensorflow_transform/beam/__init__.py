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
"""Module level imports for tensorflow_transform.beam."""

# pylint: disable=wildcard-import
# The doc-generator's `explicit_package_contents_filter` requires that
# sub-modules you want documented are explicitly imported.
# Also: analyzer_impls registers implementation of analyzers.
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.beam import analyzer_impls
from tensorflow_transform.beam import experimental
from tensorflow_transform.beam.context import Context
from tensorflow_transform.beam.impl import AnalyzeAndTransformDataset
from tensorflow_transform.beam.impl import AnalyzeDataset
from tensorflow_transform.beam.impl import AnalyzeDatasetWithCache
from tensorflow_transform.beam.impl import TransformDataset
from tensorflow_transform.beam.tft_beam_io import *

# pylint: enable=wildcard-import

# TF 2.6 split support for filesystems such as Amazon S3 out to the
# `tensorflow_io` package. Hence, this import is needed wherever we touch the
# filesystem.
try:
  import tensorflow_io as _  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  pass
