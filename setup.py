# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Package Setup script for the tf.Transform binary.
"""

from setuptools import find_packages
from setuptools import setup


def get_required_install_packages():
  return [
      'google-cloud-dataflow == 0.5.5',
  ]


def get_version():
  return '0.1'


setup(
    name='tensorflow-transform',
    version=get_version(),
    author='Google',
    author_email='tf-transform-feedback@google.com',
    namespace_packages=[],
    install_requires=get_required_install_packages(),
    packages=find_packages(),
    include_package_data=True,
    description='Tensorflow Transform',
    requires=[])
