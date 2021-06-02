# Copyright 2021 Google Inc. All Rights Reserved.
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
"""Beam implementations of experimental tf.Transform canonical analyzers."""
import apache_beam as beam


class PTransformAnalyzer(beam.PTransform):
  """A PTransform analyzer's base class which provides a temp dir if needed."""

  def __init__(self):
    self._base_temp_dir = None

  @property
  def base_temp_dir(self):
    return self._base_temp_dir

  @base_temp_dir.setter
  def base_temp_dir(self, val):
    self._base_temp_dir = val
