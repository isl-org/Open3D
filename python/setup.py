# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

from __future__ import print_function
from setuptools import setup, find_packages
import os

data_files_spec = [
    ('share/jupyter/nbextensions/open3d', 'open3d/nbextension', '*.*'),
    ('share/jupyter/labextensions/open3d', 'open3d/labextension', '**'),
    ('share/jupyter/labextensions/open3d', '.', 'install.json'),
    ('etc/jupyter/nbconfig/notebook.d', '.', 'open3d.json'),
]

if "@BUILD_JUPYTER_EXTENSION@" == "ON":
    try:
        from jupyter_packaging import (
            create_cmdclass,
            install_npm,
            ensure_targets,
            combine_commands,
        )

        # ipywidgets and jupyterlab are required to package JS code properly. They
        # are not used in setup.py.
        import ipywidgets
        import jupyterlab
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)
        print("Run `pip install jupyter_packaging ipywidgets jupyterlab`.")

    here = os.path.dirname(os.path.abspath(__file__))
    js_dir = os.path.join(here, 'js')

    # Representative files that should exist after a successful build.
    js_targets = [
        os.path.join(js_dir, 'dist', 'index.js'),
    ]

    cmdclass = create_cmdclass('jsdeps', data_files_spec=data_files_spec)
    cmdclass['jsdeps'] = combine_commands(
        install_npm(js_dir, npm=['yarn'], build_cmd='build:prod'),
        ensure_targets(js_targets),
    )
else:
    cmdclass = dict()

# Force platform specific wheel.
# https://stackoverflow.com/a/45150383/1255535
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

    cmdclass['bdist_wheel'] = bdist_wheel
except ImportError:
    print("Warning: cannot import `wheel` to build platform-specific wheel. "
          "Run `pip install wheel` to fix this warning.")

# Read requirements.
with open('requirements.txt', 'r') as f:
    install_requires = [line.strip() for line in f.readlines() if line]

# Read requirements for ML.
if '@BUNDLE_OPEN3D_ML@' == 'ON':
    with open('@OPEN3D_ML_ROOT@/requirements.txt', 'r') as f:
        install_requires += [line.strip() for line in f.readlines() if line]

setup_args = dict(
    name="@PYPI_PACKAGE_NAME@",
    version='@PROJECT_VERSION@',
    include_package_data=True,
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    cmdclass=cmdclass,
    author='Open3D Team',
    author_email='@PROJECT_EMAIL@',
    url="@PROJECT_HOME@",
    project_urls={
        'Documentation': '@PROJECT_DOCS@',
        'Source code': '@PROJECT_CODE@',
        'Issues': '@PROJECT_ISSUES@',
    },
    keywords="3D reconstruction point cloud mesh RGB-D visualization",
    license="MIT",
    classifiers=[
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        "Environment :: X11 Applications",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Education",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Multimedia :: Graphics :: Capture",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    description='Open3D: A Modern Library for 3D Data Processing.',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
)

setup(**setup_args)
