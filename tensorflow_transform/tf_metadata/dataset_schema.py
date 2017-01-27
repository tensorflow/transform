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
"""In-memory representation of the schema of a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Schema(object):
  """The schema of a dataset.

  This is an in-memory representation that may be serialized and deserialized to
  and from a variety of disk representations.
  """

  def __init__(self):
    self._features = {}

  def merge(self, other):
    # possible argument: resolution strategy (error or pick first and warn?)
    for key, value in other.features.items():
      if key in self.features:
        self.features[key].merge(value)
      else:
        self.features[key] = value

  # TODO(soergel): make this more immutable
  @property
  def features(self):
    # a dict of features
    return self._features
