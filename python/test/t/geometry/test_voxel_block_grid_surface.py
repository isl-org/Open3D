# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize(
    "sx,sy", [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)],
    ids=["flat", "positive_x", "negative_x", "positive_y", "negative_y"])
def test_integrate_plane_surface(device, sx, sy, record_property):
    """Depth integration must reconstruct an unbiased analytic plane."""
    voxel_size = 0.01
    z0 = 2.0
    fx, fy, cx, cy = 100.0, 110.0, 60.25, 50.75
    u, v = np.meshgrid(np.arange(128), np.arange(96))
    # Intersect z = z0 + sx*x + sy*y at INTEGER pixel centers. Do not
    # compensate for the old truncated lookup by shifting the input rays.
    depth_values = (z0 / (1.0 - sx * (u - cx) / fx - sy * (v - cy) / fy))
    depth = o3d.t.geometry.Image(
        o3c.Tensor(depth_values.astype(np.float32), device=device))
    intrinsic = o3c.Tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                           dtype=o3c.float64)
    extrinsic = o3c.Tensor.eye(4, dtype=o3c.float64)

    # Fixed blocks cover every plane and the truncation band, independently
    # of depth lookup. Query rays below stay away from the block/image edges.
    keys = np.stack(np.meshgrid(np.arange(-8, 8),
                                np.arange(-6, 6),
                                np.arange(18, 32),
                                indexing="ij"),
                    axis=-1)
    blocks = o3c.Tensor(keys.reshape(-1, 3).astype(np.int32), device=device)
    grid = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
                                         attr_dtypes=(o3c.float32, o3c.float32,
                                                      o3c.float32),
                                         attr_channels=((1,), (1,), (3,)),
                                         voxel_size=voxel_size,
                                         block_resolution=8,
                                         block_count=blocks.shape[0],
                                         device=device)
    grid.hashmap().activate(blocks)
    for name in ("tsdf", "weight", "color"):
        grid.attribute(name)[:] = 0.0
    # Color storage is needed by the extractor; integration remains depth-only.
    grid.integrate(blocks,
                   depth,
                   intrinsic,
                   extrinsic,
                   depth_scale=1.0,
                   depth_max=4.0,
                   trunc_voxel_multiplier=4.0)
    mesh = grid.extract_triangle_mesh(weight_threshold=0.5).cpu()
    assert mesh.triangle.indices.shape[0] > 0

    # Compare the same world positions, not differently distributed vertices.
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 33), np.linspace(-0.35, 0.35, 25))
    rays = np.zeros((x.size, 6), dtype=np.float32)
    rays[:, 0], rays[:, 1] = x.ravel(), y.ravel()
    rays[:, 2], rays[:, 5] = 4.0, -1.0
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    t_hit = scene.cast_rays(o3c.Tensor(rays))["t_hit"].numpy()
    assert np.isfinite(t_hit).all(), "Every fixed query ray must hit the mesh"
    residual = ((4.0 - t_hit.astype(np.float64) - z0 - sx * rays[:, 0] -
                 sy * rays[:, 1]) / np.sqrt(1.0 + sx * sx + sy * sy))
    rms = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    record_property("rms_mm", rms * 1000.0)
    record_property("bias_mm", bias * 1000.0)
    record_property("covered_rays", int(t_hit.size))
    if sx == 0.0 and sy == 0.0:
        assert np.max(np.abs(residual)) < 1e-6
    else:
        # Permit nearest-pixel discretization, but not a directional half-pixel
        # shift: a quarter voxel for RMS and a tenth voxel for mean bias.
        assert rms < 0.25 * voxel_size, (rms, bias)
        assert abs(bias) < 0.1 * voxel_size, (rms, bias)
