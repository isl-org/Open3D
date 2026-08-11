# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import pytest
import mltest

# Skip all tests if the ml ops were not built.
pytestmark = mltest.default_marks


def _trilinear_devoxelize_ref(coords, feat, r):
    """Numpy reference matching TrilinearDevoxelize.cu's algorithm.

    coords: [b, 3, n] grid-index-space point coordinates, each component
      strictly inside [0, r-1) (i.e. never exactly on an integer grid
      boundary), so the +1 neighbor is always in-bounds and the kernel's
      boundary-clamp trick (idx_hi wraps to idx_lo at the last slice) never
      triggers -- this keeps the reference a plain trilinear interpolation.
    feat: [b, c, r, r, r] voxel grid.

    Returns outs [b, c, n].
    """
    b, _, n = coords.shape
    c = feat.shape[1]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]  # each [b, n]
    x_lo = np.floor(x).astype(np.int64)
    y_lo = np.floor(y).astype(np.int64)
    z_lo = np.floor(z).astype(np.int64)
    xd, yd, zd = x - x_lo, y - y_lo, z - z_lo

    outs = np.zeros((b, c, n), dtype=np.float32)
    for dx in (0, 1):
        wx = xd if dx else (1 - xd)
        for dy in (0, 1):
            wy = yd if dy else (1 - yd)
            for dz in (0, 1):
                wz = zd if dz else (1 - zd)
                w = (wx * wy * wz).astype(np.float32)  # [b, n]
                gathered = feat[np.arange(b)[:, None], :, x_lo + dx, y_lo + dy,
                                z_lo + dz]  # [b, n, c]
                outs += w[:, None, :] * np.transpose(gathered, (0, 2, 1))
    return outs


@mltest.parametrize.ml_gpu_only
@pytest.mark.parametrize('is_training', [True, False])
def test_trilinear_devoxelize_forward(ml, is_training):
    rng = np.random.RandomState(0)
    b, c, n, r = 2, 3, 50, 8

    # keep coords strictly inside (0, r-1) so the +1 neighbor is always
    # in-bounds -- see _trilinear_devoxelize_ref's docstring.
    coords = rng.uniform(1e-3, r - 1 - 1e-3, size=(b, 3, n)).astype(np.float32)
    features = rng.uniform(-1, 1, size=(b, c, r, r, r)).astype(np.float32)

    coords_dev = mltest.to_torch(coords, ml.device)
    features_dev = mltest.to_torch(features, ml.device)

    outs, inds, wgts = ml.ops.trilinear_devoxelize_forward(
        r, is_training, coords_dev, features_dev)

    assert outs.shape == (b, c, n)
    if is_training:
        assert inds.shape == (b, 8, n)
        assert wgts.shape == (b, 8, n)
    else:
        assert inds.shape == (1,)
        assert wgts.shape == (1,)

    expected = _trilinear_devoxelize_ref(coords, features, r)
    np.testing.assert_allclose(mltest.to_numpy(outs),
                               expected,
                               rtol=1e-5,
                               atol=1e-5)


@mltest.parametrize.ml_gpu_only
def test_trilinear_devoxelize_backward(ml):
    rng = np.random.RandomState(0)
    b, c, n, r = 2, 3, 50, 8

    coords = rng.uniform(1e-3, r - 1 - 1e-3, size=(b, 3, n)).astype(np.float32)
    features = rng.uniform(-1, 1, size=(b, c, r, r, r)).astype(np.float32)
    grad_y = rng.uniform(-1, 1, size=(b, c, n)).astype(np.float32)

    coords_dev = mltest.to_torch(coords, ml.device)
    features_dev = mltest.to_torch(features, ml.device)
    grad_y_dev = mltest.to_torch(grad_y, ml.device)

    _, inds, wgts = ml.ops.trilinear_devoxelize_forward(r, True, coords_dev,
                                                        features_dev)

    grad_x = ml.ops.trilinear_devoxelize_backward(grad_y_dev, inds, wgts, r)

    r3 = r * r * r
    assert grad_x.shape == (b, c, r3)

    # Reference: scatter-add matching TrilinearDevoxelizeGradKernel, driven
    # by the *actual* inds/wgts the op produced (so this isolates the
    # backward scatter from the forward interpolation weights).
    inds_np = mltest.to_numpy(inds)  # [b, 8, n]
    wgts_np = mltest.to_numpy(wgts)  # [b, 8, n]
    expected = np.zeros((b, c, r3), dtype=np.float32)
    for bi in range(b):
        for k in range(8):
            np.add.at(expected[bi], (slice(None), inds_np[bi, k]),
                      wgts_np[bi, k][None, :] * grad_y[bi])

    np.testing.assert_allclose(mltest.to_numpy(grad_x),
                               expected,
                               rtol=1e-4,
                               atol=1e-4)
