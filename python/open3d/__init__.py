# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# Workaround when multiple copies of the OpenMP runtime have been linked to
# the program, which happens when PyTorch loads OpenMP runtime first. Not that
# this method is "unsafe, unsupported, undocumented", but we found it to be
# generally safe to use. This should be deprecated once we found a way to
# "ensure that only a single OpenMP runtime is linked into the process".
#
# https://github.com/llvm-mirror/openmp/blob/8453ca8594e1a5dd8a250c39bf8fcfbfb1760e60/runtime/src/i18n/en_US.txt#L449
# https://github.com/dmlc/xgboost/issues/1715
import os
import sys
import site
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Enable thread composability manager to coordinate Intel OpenMP and TBB
# threads. Only works with Intel OpenMP.  TBB must not be already loaded.
os.environ["TCM_ENABLE"] = "1"
from pathlib import Path

from open3d._build_config import _build_config

if sys.platform == "win32":
    # Required for CPU wheel (bundled TBB) and SYCL wheel (SYCL runtime, which
    # intel-sycl-rt will install in:
    # - <sys.prefix>/Library/bin # (for standard/virtualenv/conda installs)
    # - <site.USER_BASE>/Library/bin # (for user-level --user installs)
    # CUDA runtime is linked dynamically on Windows (unlike Linux, where it is
    # statically linked) and is installed by the nvidia-*-cu* pip packages
    # (see requirements_win_cuda.txt) into <site-packages>/nvidia/<component>/bin.
    _win32_dll_dirs = [os.add_dll_directory(str(Path(__file__).parent))]
    _site_dirs = [*site.PREFIXES, *site.getsitepackages(), site.USER_BASE]
    for _site_dir in _site_dirs:
        if os.path.isdir(os.path.join(_site_dir, "Library", "bin")):
            _win32_dll_dirs.append(
                os.add_dll_directory(os.path.join(_site_dir, "Library", "bin")))
        _nvidia_dir = os.path.join(_site_dir, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _nvidia_pkg_dir in os.listdir(_nvidia_dir):
                _nvidia_bin_dir = os.path.join(_nvidia_dir, _nvidia_pkg_dir,
                                               "bin")
                if os.path.isdir(_nvidia_bin_dir):
                    _win32_dll_dirs.append(
                        os.add_dll_directory(_nvidia_bin_dir))

from open3d.pybind import (
    core,
    camera,
    data,
    geometry,
    io,
    pipelines,
    utility,
    t,
)
from open3d import pybind

__DEVICE_API__ = "cpu"
if core.cuda.is_available():
    __DEVICE_API__ = "cuda"
elif core.sycl.is_available():
    __DEVICE_API__ = "xpu"


def _insert_pybind_names(skip_names=()):
    """Introduce pybind names as open3d names. Skip names corresponding to
    python subpackages, since they have a different import mechanism."""
    submodules = {}
    for modname in sys.modules:
        if "open3d.pybind" in modname:
            if any("." + skip_name in modname for skip_name in skip_names):
                continue
            # Keep the leading "open3d." so submodules are registered under
            # e.g. "open3d.t" rather than a bare "t" (which is not importable
            # via `import open3d.t`).
            subname = modname.replace("pybind.", "")
            if subname not in sys.modules:
                submodules[subname] = sys.modules[modname]
    sys.modules.update(submodules)


import open3d.visualization

_insert_pybind_names(skip_names=("ml",))

__version__ = "@PROJECT_VERSION@"

if int(sys.version_info[0]) < 3:
    raise RuntimeError("Open3D only supports Python 3.")

if (_build_config["BUILD_JUPYTER_EXTENSION"] and os.environ.get(
        "OPEN3D_DISABLE_WEB_VISUALIZER", "False").lower() != "true"):
    import platform

    if not (platform.machine().startswith("arm") or
            platform.machine().startswith("aarch")):
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                print("Jupyter environment detected. "
                      "Enabling Open3D WebVisualizer.")
                # Set default window system.
                open3d.visualization.webrtc_server.enable_webrtc()
                # HTTP handshake server is needed when Open3D is serving the
                # visualizer webpage. Disable since Jupyter is serving.
                open3d.visualization.webrtc_server.disable_http_handshake()
        except NameError:
            pass
    else:
        warnings.warn("Open3D WebVisualizer is not supported on ARM for now.",
                      RuntimeWarning)

# OPEN3D_ML_ROOT points to the root of the Open3D-ML repo.
# If set this will override the integrated Open3D-ML.
if "OPEN3D_ML_ROOT" in os.environ:
    print("Using external Open3D-ML in {}".format(os.environ["OPEN3D_ML_ROOT"]))
    sys.path.append(os.environ["OPEN3D_ML_ROOT"])
import open3d.ml

# Finally insert pybind names corresponding to ml
_insert_pybind_names()


def _jupyter_labextension_paths():
    """Called by Jupyter Lab Server to detect if it is a valid labextension and
    to install the widget.

    Returns:
        src: Source directory name to copy files from. Webpack outputs generated
            files into this directory and Jupyter Lab copies from this directory
            during widget installation.
        dest: Destination directory name to install widget files to. Jupyter Lab
            copies from `src` directory into <jupyter path>/labextensions/<dest>
            directory during widget installation.
    """
    return [{
        "src": "labextension",
        "dest": "open3d",
    }]


def _jupyter_nbextension_paths():
    """Called by Jupyter Notebook Server to detect if it is a valid nbextension
    and to install the widget.

    Returns:
        section: The section of the Jupyter Notebook Server to change.
            Must be "notebook" for widget extensions.
        src: Source directory name to copy files from. Webpack outputs generated
            files into this directory and Jupyter Notebook copies from this
            directory during widget installation.
        dest: Destination directory name to install widget files to. Jupyter
            Notebook copies from `src` directory into
            <jupyter path>/nbextensions/<dest> directory during widget
            installation.
        require: Path to importable AMD Javascript module inside the
            <jupyter path>/nbextensions/<dest> directory.
    """
    return [{
        "section": "notebook",
        "src": "nbextension",
        "dest": "open3d",
        "require": "open3d/extension",
    }]


if sys.platform == "win32":
    for dll_dir in _win32_dll_dirs:
        dll_dir.close()
del os, sys, Path, warnings, _insert_pybind_names
