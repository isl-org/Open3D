# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Tests GPU-accelerated offscreen rendering fallback (EGL) for the legacy
open3d.visualization.Visualizer, used when no display is available."""

import ctypes
import glob
import platform
import os
from multiprocessing import Process
import numpy as np
import pytest


def _try_enable_segfault_backtrace():
    """Best-effort: dlopen glibc's libSegFault.so so that a crash in this
    process prints a native (C/C++) backtrace to stderr. This is diagnostic
    only, e.g. for debugging crashes in the EGL/Mesa/GL stack below Python;
    it is a no-op (and never raises) if libSegFault.so isn't available."""
    os.environ.setdefault("SEGFAULT_SIGNALS", "all")
    candidates = glob.glob("/lib/*/libSegFault.so") + glob.glob(
        "/usr/lib/*/libSegFault.so")
    for path in candidates:
        try:
            ctypes.CDLL(path)
            return
        except OSError:
            continue


def capture_headless():
    """Runs in a separate process with no DISPLAY/WAYLAND_DISPLAY, forcing
    Visualizer to fall back to its offscreen EGL context."""
    _try_enable_segfault_backtrace()
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
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
    proc = Process(target=capture_headless)
    proc.start()
    proc.join(timeout=30)
    if proc.exitcode is None:
        proc.kill()
        proc.join()  # Reap the killed process to avoid leaving a zombie.
        pytest.fail(__name__ + " did not complete.")
    assert proc.exitcode == 0
