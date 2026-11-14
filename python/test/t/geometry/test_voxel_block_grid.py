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

# A 3-row, 4-column image, with integer coordinates at pixel centers.
# Expected indices are written explicitly: the test must not obtain its
# answer by calling the same rounding code it is meant to check.
PIXEL_CASES = [
    ("first_center", 0.0, 0.0, (0, 0)),
    ("last_center", 3.0, 2.0, (3, 2)),
    ("interior_center", 1.0, 1.0, (1, 1)),
    ("below_half", 1.25, 1.25, (1, 1)),
    ("round_x", 1.75, 1.25, (2, 1)),
    ("round_y", 1.25, 1.75, (1, 2)),
    ("round_both", 1.75, 1.75, (2, 2)),
    ("positive_half_tie", 1.5, 1.0, (2, 1)),
    ("left_edge_nearest_valid", -0.25, 1.0, (0, 1)),
    ("right_edge_nearest_valid", 3.25, 1.0, (3, 1)),
    ("top_edge_nearest_valid", 1.0, -0.25, (1, 0)),
    ("bottom_edge_nearest_valid", 1.0, 2.25, (1, 2)),
    ("left_edge_half_tie", -0.5, 1.0, None),
    ("right_edge_half_tie", 3.5, 1.0, None),
    ("top_edge_half_tie", 1.0, -0.5, None),
    ("bottom_edge_half_tie", 1.0, 2.5, None),
    ("outside_left", -0.75, 1.0, None),
    ("outside_right", 3.75, 1.0, None),
    ("outside_top", 1.0, -0.75, None),
    ("outside_bottom", 1.0, 2.75, None),
]


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize("u,v,pixel", [case[1:] for case in PIXEL_CASES],
                         ids=[case[0] for case in PIXEL_CASES])
def test_integrate_nearest_depth_pixel(device, u, v, pixel):
    """Integrating one voxel must read the nearest valid depth pixel."""
    # Distinct, exactly representable values identify which pixel was read.
    # z = 1 and sdf_trunc = 1 keep every accepted value unsaturated.
    depth_values = (1.0 + np.arange(1, 13, dtype=np.float32) / 16.0).reshape(
        (3, 4, 1))
    depth = o3d.t.geometry.Image(o3c.Tensor(depth_values, device=device))

    # With unit voxels and resolution 1, this block contains only (0, 0, 1).
    # Identity extrinsics and cx=u, cy=v project that voxel to exactly (u,v).
    blocks = o3c.Tensor([[0, 0, 1]], dtype=o3c.int32, device=device)
    intrinsic = o3c.Tensor([[1.0, 0.0, u], [0.0, 1.0, v], [0.0, 0.0, 1.0]],
                           dtype=o3c.float64)
    extrinsic = o3c.Tensor.eye(4, dtype=o3c.float64)
    grid = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight"),
                                         attr_dtypes=(o3c.float32, o3c.float32),
                                         attr_channels=((1,), (1,)),
                                         voxel_size=1.0,
                                         block_resolution=1,
                                         block_count=8,
                                         device=device)

    # Do not depend on uninitialized or newly allocated hash-map storage.
    grid.hashmap().activate(blocks)
    grid.attribute("tsdf")[:] = 0.0
    grid.attribute("weight")[:] = 0.0
    grid.integrate(blocks,
                   depth,
                   intrinsic,
                   extrinsic,
                   depth_scale=1.0,
                   depth_max=3.0,
                   trunc_voxel_multiplier=1.0)

    coords, flat_indices = grid.voxel_coordinates_and_flattened_indices()
    np.testing.assert_array_equal(coords.cpu().numpy(), [[0.0, 0.0, 1.0]])
    indices = flat_indices.cpu().numpy().reshape(-1)
    assert indices.size == 1
    index = int(indices[0])
    weight = grid.attribute("weight").cpu().numpy().reshape(-1)[index]
    tsdf = grid.attribute("tsdf").cpu().numpy().reshape(-1)[index]

    if pixel is None:
        assert weight == 0.0
        assert tsdf == 0.0
    else:
        x, y = pixel
        assert weight == 1.0
        assert tsdf == pytest.approx(float(depth_values[y, x, 0]) - 1.0,
                                     rel=0.0,
                                     abs=1e-6)
