# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys
import platform
import ctypes
from setuptools import setup, find_packages
from setuptools.command.install import install as _install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

data_files_spec = [
    ("share/jupyter/nbextensions/open3d", "open3d/nbextension", "*.*"),
    ("share/jupyter/labextensions/open3d", "open3d/labextension", "**"),
    ("share/jupyter/labextensions/open3d", ".", "install.json"),
    ("etc/jupyter/nbconfig/notebook.d", ".", "open3d.json"),
]

if "@BUILD_JUPYTER_EXTENSION@" == "ON":
    try:
        from jupyter_packaging import (
            create_cmdclass,
            install_npm,
            ensure_targets,
            combine_commands,
        )

        # ipywidgets and jupyterlab are required to package JS code properly.
        # They are not used in setup.py.
        import ipywidgets  # noqa # pylint: disable=unused-import
        import jupyterlab  # noqa # pylint: disable=unused-import
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)
        print("Run `pip install -r requirements-jupyter-build.txt`.")

    here = os.path.dirname(os.path.abspath(__file__))
    js_dir = os.path.join(here, "js")

    # Representative files that should exist after a successful build.
    js_targets = [
        os.path.join(js_dir, "dist", "index.js"),
    ]

    cmdclass = create_cmdclass("jsdeps", data_files_spec=data_files_spec)
    cmdclass["jsdeps"] = combine_commands(
        install_npm(js_dir, npm=["yarn"], build_cmd="build:prod"),
        ensure_targets(js_targets),
    )
else:
    cmdclass = {}


# Force platform specific wheel.
# https://stackoverflow.com/a/45150383/1255535
class bdist_wheel(_bdist_wheel):

    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    # https://github.com/Yelp/dumb-init/blob/57f7eebef694d780c1013acd410f2f0d3c79f6c6/setup.py#L25
    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        if plat[:5] == "linux":
            libc = ctypes.CDLL("libc.so.6")
            libc.gnu_get_libc_version.restype = ctypes.c_char_p
            GLIBC_VER = libc.gnu_get_libc_version().decode("utf8").split(".")
            plat = f"manylinux_{GLIBC_VER[0]}_{GLIBC_VER[1]}{plat[5:]}"
        return python, abi, plat


cmdclass["bdist_wheel"] = bdist_wheel


# Force use of "platlib" dir for auditwheel to recognize this is a non-pure
# build
# http://lxr.yanyahua.com/source/llvmlite/setup.py
class install(_install):

    def finalize_options(self):
        _install.finalize_options(self)
        self.install_libbase = self.install_platlib
        self.install_lib = self.install_platlib


cmdclass["install"] = install

# Read requirements.
with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f.readlines() if line]

# Read requirements for ML.
if "@BUNDLE_OPEN3D_ML@" == "ON":
    with open("@OPEN3D_ML_ROOT@/requirements.txt", "r") as f:
        install_requires += [line.strip() for line in f.readlines() if line]

entry_points = {
    "console_scripts": ["open3d = @PYPI_PACKAGE_NAME@.tools.cli:main",]
}
if sys.platform != "darwin":  # Remove check when off main thread GUI works
    entry_points.update({
        "tensorboard_plugins": [
            "Open3D = @PYPI_PACKAGE_NAME@.visualization.tensorboard_plugin"
            ".plugin:Open3DPlugin",
        ]
    })
classifiers = [
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    "Development Status :: 3 - Alpha",
    "Environment :: MacOS X",
    "Environment :: Win32 (MS Windows)",
    "Environment :: X11 Applications",
    "Environment :: GPU :: NVIDIA CUDA",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Other Audience",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: C",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
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
]
name = "@PYPI_PACKAGE_NAME@"
with open("README.rst") as readme:
    long_description = readme.read()
# open3d-cpu wheel for Linux x86_64
if sys.platform.startswith("linux") and platform.machine() in (
        'i386', 'x86_64', 'AMD64') and "@BUILD_CUDA_MODULE@" == "OFF":
    name += "-cpu"
    long_description += ("\n\nThis wheel only contains CPU functionality. "
                         "Use the open3d wheel for full functionality.")
    classifiers.remove("Environment :: GPU :: NVIDIA CUDA")

setup_args = dict(
    name=name,
    version="@PROJECT_VERSION@",
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    packages=find_packages(),
    entry_points=entry_points,
    zip_safe=False,
    cmdclass=cmdclass,
    author="Open3D Team",
    author_email="@PROJECT_EMAIL@",
    url="@PROJECT_HOMEPAGE_URL@",
    project_urls={
        "Documentation": "@PROJECT_DOCS@",
        "Source code": "@PROJECT_CODE@",
        "Issues": "@PROJECT_ISSUES@",
    },
    classifiers=classifiers,
    keywords="3D reconstruction point cloud mesh RGB-D visualization",
    license="MIT",
    description="@PROJECT_DESCRIPTION@",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    # Metadata below is valid but currently ignored by pip (<=v23)
    obsoletes=["open3d_python"],
    provides=["open3d", "open3d_cpu"],  # For open3d-cpu
)

setup(**setup_args)
