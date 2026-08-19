# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Tests GPU-accelerated offscreen rendering fallback (EGL) for the legacy
open3d.visualization.Visualizer, used when no display is available. See also
test_cpu_rendering.py, which covers the new Filament-based
open3d.visualization.rendering.OffscreenRenderer's CPU (software) rendering."""

import ctypes
import platform
import os
import multiprocessing
import numpy as np
import pytest

# Child process exit code reporting that this machine has no EGL driver able to
# render desktop OpenGL, e.g. a GPU-less container without a software rasterizer.
NO_EGL_EXIT_CODE = 77


def egl_desktop_gl_available():
    """Returns whether libEGL can bind the desktop OpenGL API, which the legacy
    Visualizer's offscreen context requires (see EGLOffscreenContext.cpp).
    Drivers exposing OpenGL ES only, or no driver at all, cannot provide it."""
    EGL_OPENGL_API = 0x30A2
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError:
        return False
    egl.eglGetDisplay.restype = ctypes.c_void_p
    egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
    egl.eglInitialize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)
    ]
    egl.eglTerminate.argtypes = [ctypes.c_void_p]

    display = egl.eglGetDisplay(None)  # EGL_DEFAULT_DISPLAY
    if not display:
        return False
    major, minor = ctypes.c_int(), ctypes.c_int()
    if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
        return False
    available = bool(egl.eglBindAPI(EGL_OPENGL_API))
    egl.eglTerminate(display)
    return available


def capture_headless():
    """Runs in a separate process with no DISPLAY/WAYLAND_DISPLAY, forcing
    the Visualizer to fall back to its offscreen EGL context."""
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    if not egl_desktop_gl_available():
        raise SystemExit(NO_EGL_EXIT_CODE)
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((1.0, 0.0, 0.0))

    vis = o3d.visualization.Visualizer()
    assert vis.create_window(visible=False, width=320, height=240)
    vis.add_geometry(mesh)
    image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    assert image.shape == (240, 320, 3)
    # Rendered sphere should not be a blank (all-background) image.
    assert image.std() > 0.0


@pytest.mark.skipif(platform.system() != "Linux",
                    reason="EGL offscreen rendering fallback is Linux-only")
def test_legacy_visualizer_headless_capture():
    """Test that the legacy Visualizer can render offscreen via EGL when no
    windowing system display is available, in the standard Open3D binary."""
    # Use "spawn" (not the default "fork") to start from a clean interpreter.
    # Other tests in this pytest session may have already imported
    # TensorFlow, which bundles its own LLVM inside libtensorflow_framework.
    # A forked child inherits that already-loaded, symbol-versioned LLVM; on
    # GPU-less machines Mesa's software rasterizer (llvmpipe) resolves its
    # LLVM JIT calls (e.g. LLVMBuildGEP2) against the wrong LLVM copy,
    # corrupting memory and segfaulting. "spawn" avoids inheriting any of
    # the parent's loaded shared libraries.
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=capture_headless)
    proc.start()
    proc.join(timeout=30)
    if proc.exitcode is None:
        proc.kill()
        proc.join()  # Reap the killed process to avoid leaving a zombie.
        pytest.fail(__name__ + " did not complete.")
    if proc.exitcode == NO_EGL_EXIT_CODE:
        pytest.skip(
            "No EGL driver with desktop OpenGL support on this machine.")
    assert proc.exitcode == 0
