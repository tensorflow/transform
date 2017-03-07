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
"""In-memory representation of the problem statements associated with a dataset.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class ProblemStatements(collections.namedtuple('ProblemStatements', [])):

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self._asdict() == other._asdict()
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def merge(self, other):
    pass


class ProblemStatement(collections.namedtuple(
    'ProblemStatement',
    ['raw_feature_keys',
     'raw_label_keys',
     'raw_weights_keys',
     'transformed_feature_keys',
     'transformed_label_keys',
     'transformed_weights_keys'])):
  # the main constraint that we could enforce is that a transformed feature or
  # weight cannot depend on a raw label.
  pass
