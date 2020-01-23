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

from setuptools import setup, find_packages
import glob

# Force platform specific wheel
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    # https://stackoverflow.com/a/45150383/1255535
    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    print(
        'Warning: cannot import "wheel" package to build platform-specific wheel'
    )
    print('Install the "wheel" package to fix this warning')
    bdist_wheel = None

cmdclass = {'bdist_wheel': bdist_wheel} if bdist_wheel is not None else dict()

# Read requirements.txt
with open('requirements.txt', 'r') as f:
    lines = f.readlines()
install_requires = [line.strip() for line in lines if line]

# Data files for packaging
data_files = [
    ('share/jupyter/nbextensions/open3d', glob.glob('open3d/static/*')),
    ('etc/jupyter/nbconfig/notebook.d', ['enable_jupyter_extension.json'])
]

setup(
    author='Open3D Team',
    author_email='@PROJECT_EMAIL@',
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
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
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
    description=[
        "Open3D is an open-source library that supports rapid development of software that deals with 3D data."
    ],
    cmdclass=cmdclass,
    install_requires=install_requires,
    include_package_data=True,
    data_files=data_files,
    keywords="3D reconstruction point cloud mesh RGB-D visualization",
    license="MIT",
    long_description=open('README.rst').read(),
    # Name of the package on PyPI
    name="@PYPI_PACKAGE_NAME@",
    packages=[
        'open3d',
    ],
    url="@PROJECT_HOME@",
    project_urls={
        'Documentation': '@PROJECT_DOCS@',
        'Source code': '@PROJECT_CODE@',
        'Issues': '@PROJECT_ISSUES@',
    },
    version='@PROJECT_VERSION@',
    zip_safe=False,
)
