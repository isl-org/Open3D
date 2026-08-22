# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys
import site
import warnings

# Open3D uses oneAPI TBB (not OpenMP) for CPU parallelism. Other packages (e.g. SciPy, MKL) may
# still bring their own Intel OpenMP; enabling the thread composability manager
# lets it share a thread pool with TBB instead of oversubscribing the machine.
# Only works with Intel OpenMP, and TBB must not be already loaded.
os.environ["TCM_ENABLE"] = "1"
from pathlib import Path

from open3d._build_config import _build_config

if sys.platform == "win32":
    # Required for CPU wheel (bundled TBB) and SYCL wheel (SYCL runtime from
    # pip-installed dpcpp-cpp-rt / intel-sycl-rt). Runtimes may appear under
    # <sys.prefix>/Library/bin, <site>/Library/bin, or pip's
    # <site>/*.data/data/Library/bin layout (see 3rdparty/README_SYCL.md).
    # CUDA runtime is linked dynamically on Windows and is installed by the
    # nvidia-*-cu* pip packages into <site-packages>/nvidia/<component>/bin.
    def _maybe_add_dll_dir(path, handles, seen):
        path = os.path.abspath(path)
        if path in seen or not os.path.isdir(path):
            return
        seen.add(path)
        handles.append(os.add_dll_directory(path))

    _win32_dll_dirs = []
    _seen_dll_dirs = set()
    _path_candidates = [str(Path(__file__).parent)]
    _path_candidates.append(os.path.join(sys.prefix, "Library", "bin"))

    _site_dirs = set(site.PREFIXES) | set(site.getsitepackages())
    if site.USER_BASE:
        _site_dirs.add(site.USER_BASE)
    for _site_dir in _site_dirs:
        if not _site_dir:
            continue
        _path_candidates.append(os.path.join(_site_dir, "Library", "bin"))
        _site_path = Path(_site_dir)
        if _site_path.is_dir():
            for _data_root in _site_path.glob("*.data"):
                _path_candidates.append(
                    str(_data_root / "data" / "Library" / "bin"))
        _nvidia_dir = os.path.join(_site_dir, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _nvidia_pkg_dir in os.listdir(_nvidia_dir):
                _nvidia_bin_dir = os.path.join(_nvidia_dir, _nvidia_pkg_dir,
                                               "bin")
                _path_candidates.append(_nvidia_bin_dir)

    if _build_config.get("BUILD_SYCL_MODULE"):
        import importlib.util

        _torch_spec = importlib.util.find_spec("torch")
        if _torch_spec and _torch_spec.submodule_search_locations:
            _path_candidates.append(
                os.path.join(_torch_spec.submodule_search_locations[0], "lib"))

    for _path in _path_candidates:
        _maybe_add_dll_dir(_path, _win32_dll_dirs, _seen_dll_dirs)

    # Transitive SYCL/CUDA deps may still be resolved via PATH on Windows.
    _path_prefix = os.pathsep.join(
        p for p in _path_candidates if p and os.path.isdir(p))
    if _path_prefix:
        os.environ["PATH"] = _path_prefix + os.pathsep + os.environ.get(
            "PATH", "")

    del _maybe_add_dll_dir, _seen_dll_dirs, _path_candidates, _path_prefix

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


del os, sys, Path, warnings, _insert_pybind_names
# If this is removed, pybind11_stubgen adds an incomplete "open3d = " to the stub file
del open3d
