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
"""Package Setup script for tf.Transform."""
import os
from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def select_constraint(default, nightly=None, git_master=None):
  """Select dependency constraint based on TFX_DEPENDENCY_SELECTOR env var."""
  selector = os.environ.get('TFX_DEPENDENCY_SELECTOR')
  if selector == 'UNCONSTRAINED':
    return ''
  elif selector == 'NIGHTLY' and nightly is not None:
    return nightly
  elif selector == 'GIT_MASTER' and git_master is not None:
    return git_master
  else:
    return default


# Get version from version module.
with open('tensorflow_transform/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy, and
  # protobuf) with TF and pyarrow version with tfx-bsl.
  return [
      'absl-py>=0.9,<2.0.0',
      'apache-beam[gcp]>=2.53,<3;python_version>="3.11"',
      'apache-beam[gcp]>=2.50,<2.51;python_version<"3.11"',
      'numpy>=1.22.0',
      'protobuf>=4.25.2,<6.0.0;python_version>="3.11"',
      'protobuf>=4.21.6,<6.0.0;python_version<"3.11"',
      'pyarrow>=10,<11',
      'pydot>=1.2,<2',
      'tensorflow>=2.17,<2.18',
      'tensorflow-metadata'
      + select_constraint(
          default='>=1.17.1,<1.18.0',
          nightly='>=1.18.0.dev',
          git_master='@git+https://github.com/tensorflow/metadata@master',
      ),
      'tf_keras>=2',
      'tfx-bsl'
      + select_constraint(
          default='>=1.17.1,<1.18.0',
          nightly='>=1.18.0.dev',
          git_master='@git+https://github.com/tensorflow/tfx-bsl@master',
      ),
  ]

def _make_docs_packages():
  return [
      req for req in Path("./requirements-docs.txt")
      .expanduser()
      .resolve()
      .read_text()
      .splitlines()
      if req
  ]


# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='tensorflow-transform',
    version=__version__,
    author='Google Inc.',
    author_email='tensorflow-extended-dev@googlegroups.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    extras_require= {
      'test': ['pytest>=8.0'],
      'docs': _make_docs_packages(),
    },
    python_requires='>=3.9,<4',
    packages=find_packages(),
    include_package_data=True,
    package_data={'tensorflow_transform': ['py.typed', 'requirements-docs.txt']},
    description='A library for data preprocessing with TensorFlow',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tensorflow transform tfx',
    url='https://www.tensorflow.org/tfx/transform/get_started',
    download_url='https://github.com/tensorflow/transform/tags',
    requires=[])
