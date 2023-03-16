# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from ctypes import CDLL
from ctypes.util import find_library
from pathlib import Path
import warnings
from open3d._build_config import _build_config


def load_cdll(path):
    """
    Wrapper around ctypes.CDLL to take care of Windows compatibility.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Shared library file not found: {path}.")

    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        # https://stackoverflow.com/a/64472088/1255535
        return CDLL(str(path), winmode=0)
    else:
        return CDLL(str(path))


if _build_config["BUILD_GUI"] and not (find_library("c++abi") or
                                       find_library("c++")):
    try:  # Preload libc++.so and libc++abi.so (required by filament)
        load_cdll(str(next((Path(__file__).parent).glob("*c++abi.*"))))
        load_cdll(str(next((Path(__file__).parent).glob("*c++.*"))))
    except StopIteration:  # Not found: check system paths while loading
        pass

# Enable CPU rendering based on env vars
if _build_config["BUILD_GUI"] and sys.platform.startswith("linux") and (
        os.getenv("OPEN3D_CPU_RENDERING", default="") == "true"):
    os.environ["LIBGL_DRIVERS_PATH"] = str(Path(__file__).parent)
    load_cdll(Path(__file__).parent / "libEGL.so.1")
    load_cdll(Path(__file__).parent / "libGL.so.1")

__DEVICE_API__ = "cpu"
if _build_config["BUILD_CUDA_MODULE"]:
    # Load CPU pybind dll gracefully without introducing new python variable.
    # Do this before loading the CUDA pybind dll to correctly resolve symbols
    try:  # StopIteration if cpu version not available
        load_cdll(str(next((Path(__file__).parent / "cpu").glob("pybind*"))))
    except StopIteration:
        warnings.warn(
            "Open3D was built with CUDA support, but Open3D CPU Python "
            "bindings were not found. Open3D will not work on systems without"
            " CUDA devices.", ImportWarning)
    try:
        # Check CUDA availability without importing CUDA pybind symbols to
        # prevent "symbol already registered" errors if first import fails.
        _pybind_cuda = load_cdll(
            str(next((Path(__file__).parent / "cuda").glob("pybind*"))))
        if _pybind_cuda.open3d_core_cuda_device_count() > 0:
            from open3d.cuda.pybind import (core, camera, data, geometry, io,
                                            pipelines, utility, t)
            from open3d.cuda import pybind
            __DEVICE_API__ = "cuda"
        else:
            warnings.warn(
                "Open3D was built with CUDA support, but no suitable CUDA "
                "devices found. If your system has CUDA devices, check your "
                "CUDA drivers and runtime.", ImportWarning)
    except OSError as os_error:
        warnings.warn(
            f'Open3D was built with CUDA support, but an error ocurred while loading the Open3D CUDA Python bindings. This is usually because the CUDA libraries could not be found. Check your CUDA installation. Falling back to the CPU pybind library. Reported error: {os_error}.',
            ImportWarning)
    except StopIteration:
        warnings.warn(
            "Open3D was built with CUDA support, but Open3D CUDA Python "
            "binding library not found! Falling back to the CPU Python "
            "binding library.", ImportWarning)

if __DEVICE_API__ == "cpu":
    from open3d.cpu.pybind import (core, camera, data, geometry, io, pipelines,
                                   utility, t)
    from open3d.cpu import pybind


def _insert_pybind_names(skip_names=()):
    """Introduce pybind names as open3d names. Skip names corresponding to
    python subpackages, since they have a different import mechanism."""
    submodules = {}
    for modname in sys.modules:
        if "open3d." + __DEVICE_API__ + ".pybind" in modname:
            if any("." + skip_name in modname for skip_name in skip_names):
                continue
            subname = modname.replace(__DEVICE_API__ + ".pybind.", "")
            if subname not in sys.modules:
                submodules[subname] = sys.modules[modname]
    sys.modules.update(submodules)


import open3d.visualization
_insert_pybind_names(skip_names=("ml",))

__version__ = "@PROJECT_VERSION@"

if int(sys.version_info[0]) < 3:
    raise Exception("Open3D only supports Python 3.")

if _build_config["BUILD_JUPYTER_EXTENSION"]:
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
        "require": "open3d/extension"
    }]


del os, sys, CDLL, load_cdll, find_library, Path, warnings, _insert_pybind_names
