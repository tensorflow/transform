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
"""Package Setup script for tf.Transform.
"""
from setuptools import find_packages
from setuptools import setup

# Tensorflow transform version.
__version__ = '0.12.0'


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF.
  return [
      'absl-py>=0.1.6',
      'apache-beam[gcp]>=2.10,<3',
      'numpy>=1.14.5,<2',
      # TODO(b/124072021): currently one test fails for py3 with protobuf 3.6
      # update this to a non-rc version.
      'protobuf==3.7.0rc2',
      'six>=1.10,<2',
      'tensorflow-metadata>=0.9,<0.13',

      # TODO(b/123240958): Uncomment this once TF can automatically select
      # between CPU and GPU installation.
      # 'tensorflow>=1.12,<2',
      'pydot>=1.2.0,<1.3',
  ]

# TODO(b/121329572): Remove the following comment after we can guarantee the
# required versions of packages through kokoro release workflow.
# Note: In order for the README to be rendered correctly, make sure to have the
# following minimum required versions of the respective packages when building
# and uploading the zip/wheel package to PyPI:
# setuptools >= 38.6.0, wheel >= 0.31.0, twine >= 1.11.0

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='tensorflow-transform',
    version=__version__,
    author='Google Inc.',
    author_email='tf-transform-feedback@google.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # TODO(b/34685282): Once we support Python 3, remove this line.
        'Programming Language :: Python :: 2 :: Only',
        # TODO(b/34685282): Once we support Python 3, uncomment these lines.
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    # TODO(b/34685282): Remove < 3 after Apache Beam 2.11 is released.
    python_requires='>=2.7,<3',
    packages=find_packages(),
    include_package_data=True,
    description='A library for data preprocessing with TensorFlow',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tensorflow transform tfx',
    url='https://www.tensorflow.org/tfx/transform',
    download_url='https://pypi.org/project/tensorflow-transform',
    requires=[])
