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
import re
import site

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Enable thread composability manager to coordinate Intel OpenMP and TBB threads. Only works with Intel OpenMP.
# TBB must not be already loaded.
os.environ["TCM_ENABLE"] = "1"
from ctypes import CDLL
from ctypes.util import find_library
from pathlib import Path
import warnings
from open3d._build_config import _build_config

_added_dll_dirs = []
if sys.platform == "win32":  # Unix: Use rpath to find libraries
    _win32_dll_dir = os.add_dll_directory(str(Path(__file__).parent))
    # SYCL wheels depend on Intel DPC++ runtime packages (dpcpp-cpp-rt and its
    # dependencies: intel-sycl-rt, intel-cmplr-lib-rt, intel-opencl-rt, ...)
    # installed in the same environment. These pip wheels ship their DLLs in a
    # "<name>.data/data/Library/bin" wheel data dir, which pip installs to the
    # environment's data scheme root (sys.prefix), i.e. "<prefix>/Library/bin".
    # The ".data" dir does not persist under site-packages after install, so we
    # add the prefix-relative "Library/bin" (and its subdirs) to the DLL search
    # path. We also scan site-packages for any residual ".data" layout as a
    # fallback for non-standard install schemes.
    if _build_config["BUILD_SYCL_MODULE"]:
        _intel_pip_dll_dirs = set()
        _library_bin_roots = [Path(sys.prefix) / "Library" / "bin"]
        for _lib_bin in _library_bin_roots:
            if not _lib_bin.is_dir():
                continue
            _intel_pip_dll_dirs.add(_lib_bin)
            for _child in _lib_bin.iterdir():
                if _child.is_dir():
                    _intel_pip_dll_dirs.add(_child)
        _site_package_roots = [Path(_p) for _p in site.getsitepackages()]
        _user_site = site.getusersitepackages()
        if _user_site:
            _site_package_roots.append(Path(_user_site))
        for _site_root in _site_package_roots:
            if not _site_root.is_dir():
                continue
            for _lib_bin in _site_root.glob("*.data/data/Library/bin"):
                if not _lib_bin.is_dir():
                    continue
                _intel_pip_dll_dirs.add(_lib_bin)
                for _child in _lib_bin.iterdir():
                    if _child.is_dir():
                        _intel_pip_dll_dirs.add(_child)
        for _p in sorted(_intel_pip_dll_dirs):
            try:
                _added_dll_dirs.append(os.add_dll_directory(str(_p)))
            except Exception:
                pass

__DEVICE_API__ = "cpu"
if _build_config["BUILD_CUDA_MODULE"]:
    # Load CPU pybind dll gracefully without introducing new python variable.
    # Do this before loading the CUDA pybind dll to correctly resolve symbols
    try:  # StopIteration if cpu version not available
        CDLL(str(next((Path(__file__).parent / "cpu").glob("pybind*"))))
    except StopIteration:
        warnings.warn(
            "Open3D was built with CUDA support, but Open3D CPU Python "
            "bindings were not found. Open3D will not work on systems without"
            " CUDA devices.",
            ImportWarning,
        )
    try:
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            # Since Python 3.8, the PATH environment variable is not used to find DLLs anymore.
            # To allow Windows users to use Open3D with CUDA without running into dependency-problems,
            # look for the CUDA bin directory in PATH and explicitly add it to the DLL search path.
            cuda_bin_path = None
            for path in os.environ['PATH'].split(';'):
                # search heuristic: look for a path containing "cuda" and "bin" in this order.
                if re.search(r'cuda.*bin', path, re.IGNORECASE):
                    cuda_bin_path = path
                    break

            if cuda_bin_path:
                os.add_dll_directory(cuda_bin_path)

        # Check CUDA availability without importing CUDA pybind symbols to
        # prevent "symbol already registered" errors if first import fails.
        _pybind_cuda = CDLL(
            str(next((Path(__file__).parent / "cuda").glob("pybind*"))))
        if _pybind_cuda.open3d_core_cuda_device_count() > 0:
            from open3d.cuda.pybind import (
                core,
                camera,
                data,
                geometry,
                io,
                pipelines,
                utility,
                t,
            )
            from open3d.cuda import pybind

            __DEVICE_API__ = "cuda"
        else:
            warnings.warn(
                "Open3D was built with CUDA support, but no suitable CUDA "
                "devices found. If your system has CUDA devices, check your "
                "CUDA drivers and runtime.",
                ImportWarning,
            )
    except OSError as os_error:
        warnings.warn(
            f"Open3D was built with CUDA support, but an error occurred while loading the Open3D CUDA Python bindings. This is usually because the CUDA libraries could not be found. Check your CUDA installation. Falling back to the CPU pybind library. Reported error: {os_error}.",
            ImportWarning,
        )
    except StopIteration:
        warnings.warn(
            "Open3D was built with CUDA support, but Open3D CUDA Python "
            "binding library not found! Falling back to the CPU Python "
            "binding library.",
            ImportWarning,
        )

if _build_config["BUILD_SYCL_MODULE"]:
    try:
        from open3d.xpu.pybind import (
            core,
            camera,
            data,
            geometry,
            io,
            pipelines,
            utility,
            t,
        )
        from open3d.xpu import pybind

        __DEVICE_API__ = "xpu"
    except OSError as os_error:
        warnings.warn(
            f"Open3D was built with SYCL support, but an error occurred while loading the Open3D SYCL Python bindings. Ensure the DPC++ runtime (dpcpp-cpp-rt) is installed in this Python environment. Falling back to the CPU pybind library. Reported error: {os_error}.",
            ImportWarning,
        )
    except StopIteration:
        warnings.warn(
            "Open3D was built with SYCL support, but Open3D SYCL Python "
            "binding library not found! Falling back to the CPU Python "
            "binding library.",
            ImportWarning,
        )

if __DEVICE_API__ == "cpu":
    from open3d.cpu.pybind import (
        core,
        camera,
        data,
        geometry,
        io,
        pipelines,
        utility,
        t,
    )
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
    _win32_dll_dir.close()
    for _d in _added_dll_dirs:
        _d.close()
del os, sys, CDLL, find_library, Path, warnings, _insert_pybind_names
