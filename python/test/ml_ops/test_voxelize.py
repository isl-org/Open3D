# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest
import mltest
from check_gradients import check_gradients

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks

# the supported dtypes
point_dtypes = pytest.mark.parametrize('point_dtype', [np.float32, np.float64])

# the supported dimensions
ndims = pytest.mark.parametrize('ndim', range(1, 9))


def voxelize_python(points_batch,
                    row_splits,
                    voxel_size,
                    point_range_min,
                    point_range_max,
                    max_points_per_voxel=None,
                    max_voxels=None):
    if max_points_per_voxel is None:
        max_points_per_voxel = points_batch.shape[0]
    if max_voxels is None:
        max_voxels = points_batch.shape[0]

    ans = []
    for batch_id in range(row_splits.shape[0] - 1):
        points = points_batch[row_splits[batch_id]:row_splits[batch_id + 1]]
        valid = np.logical_and(
            np.all(points >= point_range_min[np.newaxis, :], axis=1),
            np.all(points <= point_range_max[np.newaxis, :], axis=1))
        inv_voxel_size = 1 / voxel_size[np.newaxis, :]
        voxel_coords = ((points - point_range_min[np.newaxis, :]) *
                        inv_voxel_size).astype(np.int32)

        voxels = {tuple(c): [] for i, c in enumerate(voxel_coords) if valid[i]}
        for i, c in enumerate(voxel_coords):
            if valid[i] and len(voxels[tuple(c)]) < max_points_per_voxel:
                voxels[tuple(c)].append(row_splits[batch_id] + i)

        voxel_keys = list(voxels.keys())
        voxel_keys.sort(key=lambda x: x[::-1])
        voxel_keys = voxel_keys[:min(max_voxels, len(voxel_keys))]
        ans.append({k: set(voxels[k]) for k in voxel_keys})

    return ans


def convert_output_to_voxel_dict(out):
    ans = []
    for batch_id in range(out.voxel_batch_splits.shape[0] - 1):
        start = out.voxel_batch_splits[batch_id]
        end = out.voxel_batch_splits[batch_id + 1]
        voxels = {}
        for i, c in enumerate(out.voxel_coords[start:end]):
            point_indices = out.voxel_point_indices[out.voxel_point_row_splits[
                start + i]:out.voxel_point_row_splits[start + i + 1]]
            voxels[tuple(c)] = set(point_indices)
        ans.append(voxels)
    return ans


def assert_equal_voxel_dicts(out_b, ref_b):
    for out, ref in zip(out_b, ref_b):
        assert sorted(out.keys()) == sorted(ref.keys())
        for k in out:
            assert out[k] == ref[k]


iteration = 0


@mltest.parametrize.ml
@point_dtypes
@pytest.mark.parametrize('point_range_max', ([11, 11, 11], [2, 2, 2]))
@pytest.mark.parametrize('max_voxels', [1000, 2, 1, 0])
@pytest.mark.parametrize('max_points_per_voxel', [1000, 2, 1, 0])
def test_voxelize_simple(ml, point_dtype, point_range_max, max_voxels,
                         max_points_per_voxel):
    # global iteration
    # if iteration > 0:
    # return
    # iteration += 1
    # if 'GPU' in ml.device:
    # return
    # yapf: disable

    points = np.array([
        # 2 points in voxel
        [0.5, 0.5, 0.5],
        [0.7, 0.2, 0.3],
        # new batch
        # 1 point in two voxels
        [0.7, 0.5, 0.9],
        [10.7, 10.2, 10.3],
        # 2 points in another voxel
        [1.4, 1.5, 1.4],
        [1.7, 1.2, 1.3],
        ], dtype=point_dtype)
    row_splits = np.array([0, 2, 4, 6], dtype=np.int64)
    # row_splits = np.array([0,6], dtype=np.int64)

    # yapf: enable

    voxel_size = np.array([1.0, 1.1, 1.2], dtype=point_dtype)
    point_range_min = np.zeros((3,), dtype=point_dtype)
    point_range_max = np.array(point_range_max, dtype=point_dtype)
    ans = mltest.run_op(ml,
                        ml.device,
                        True,
                        ml.ops.voxelize,
                        points,
                        row_splits,
                        voxel_size,
                        point_range_min,
                        point_range_max,
                        max_points_per_voxel=max_points_per_voxel,
                        max_voxels=max_voxels)

    voxels = convert_output_to_voxel_dict(ans)
    voxels_reference = voxelize_python(
        points,
        row_splits,
        voxel_size,
        point_range_min,
        point_range_max,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels)
    # print(voxels)
    # print(voxels_reference)
    # print('max_voxels', max_voxels, 'max_points_per_voxel',
    # max_points_per_voxel)
    assert_equal_voxel_dicts(voxels, ref_b=voxels_reference)


@mltest.parametrize.ml
@point_dtypes
@ndims
@pytest.mark.parametrize('batch_size', [1, 2, 3, 8, 16])
@pytest.mark.parametrize('max_voxels', [10000, 16, 1, 0])
@pytest.mark.parametrize('max_points_per_voxel', [10000, 16, 1, 0])
def test_voxelize_random(ml, point_dtype, ndim, batch_size, max_voxels,
                         max_points_per_voxel):
    rng = np.random.RandomState(123)
    points = rng.rand(rng.randint(0, 10000), ndim).astype(point_dtype)

    row_splits = np.zeros(shape=(batch_size + 1,), dtype=np.int64)
    for i in range(batch_size):
        row_splits[i + 1] = rng.randint(
            points.shape[0] // batch_size) + row_splits[i]
    points = points[:row_splits[batch_size]]

    voxel_size = rng.uniform(0.01, 0.1, size=(ndim,)).astype(point_dtype)
    point_range_min = rng.uniform(0.0, 0.3, size=(ndim,)).astype(point_dtype)
    point_range_max = rng.uniform(0.7, 1.0, size=(ndim,)).astype(point_dtype)

    ans = mltest.run_op(ml,
                        ml.device,
                        True,
                        ml.ops.voxelize,
                        points,
                        row_splits,
                        voxel_size,
                        point_range_min,
                        point_range_max,
                        max_points_per_voxel=max_points_per_voxel,
                        max_voxels=max_voxels)

    voxels = convert_output_to_voxel_dict(ans)
    voxels_reference = voxelize_python(
        points,
        row_splits,
        voxel_size,
        point_range_min,
        point_range_max,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels)
    assert_equal_voxel_dicts(voxels, ref_b=voxels_reference)
