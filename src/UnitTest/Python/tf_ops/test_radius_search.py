# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import pytest
import mark_helper

# skip all tests if the tf ops were not built and disable warnings caused by
# tensorflow
pytestmark = mark_helper.tf_marks

# the supported dtypes for the point coordinates
dtypes = pytest.mark.parametrize('dtype', [np.float32, np.float64])

# the GPU only supports single precision float
gpu_dtypes = [np.float32]


@dtypes
@pytest.mark.parametrize('num_points_queries', [(10, 5), (31, 33), (33, 31),
                                                (123, 345)])
@pytest.mark.parametrize('metric', ['L1', 'L2'])
@pytest.mark.parametrize('ignore_query_point', [False, True])
@pytest.mark.parametrize('return_distances', [False, True])
@pytest.mark.parametrize('normalize_distances', [False, True])
def test_radius_search(dtype, num_points_queries, metric, ignore_query_point,
                       return_distances, normalize_distances):
    import tensorflow as tf
    import open3d.ml.tf as ml3d

    rng = np.random.RandomState(123)

    num_points, num_queries = num_points_queries

    points = rng.random(size=(num_points, 3)).astype(dtype)
    if ignore_query_point:
        queries = points
    else:
        queries = rng.random(size=(num_queries, 3)).astype(dtype)

    radii = rng.uniform(0.1, 0.3, size=queries.shape[:1]).astype(dtype)

    # kd tree for computing the ground truth
    tree = cKDTree(points, copy_data=True)
    p_norm = {'L1': 1, 'L2': 2, 'Linf': np.inf}[metric]
    gt_neighbors_index = [
        tree.query_ball_point(q, r, p=p_norm) for q, r in zip(queries, radii)
    ]

    ans = ml3d.ops.radius_search(points,
                                 queries,
                                 radii,
                                 metric,
                                 ignore_query_point=ignore_query_point,
                                 return_distances=return_distances,
                                 normalize_distances=normalize_distances)

    # convert to numpy for convenience
    ans_neighbors_index = ans.neighbors_index.numpy()
    ans_neighbors_row_splits = ans.neighbors_row_splits.numpy()
    if return_distances:
        ans_neighbors_distance = ans.neighbors_distance.numpy()

    for i, q in enumerate(queries):
        # check neighbors
        start = ans_neighbors_row_splits[i]
        end = ans_neighbors_row_splits[i + 1]
        q_neighbors_index = ans_neighbors_index[start:end]

        gt_set = set(gt_neighbors_index[i])
        if ignore_query_point:
            gt_set.remove(i)
        assert gt_set == set(q_neighbors_index)

        # check distances
        if return_distances:
            q_neighbors_dist = ans_neighbors_distance[start:end]
            for j, dist in zip(q_neighbors_index, q_neighbors_dist):
                if metric == 'L2':
                    gt_dist = np.sum((q - points[j])**2)
                    if normalize_distances:
                        gt_dist /= radii[i]**2
                else:
                    gt_dist = np.linalg.norm(q - points[j], ord=p_norm)
                    if normalize_distances:
                        gt_dist /= radii[i]

                np.testing.assert_allclose(dist, gt_dist, rtol=1e-7, atol=1e-8)


def test_radius_search_empty_point_sets():
    import tensorflow as tf
    import open3d.ml.tf as ml3d
    rng = np.random.RandomState(123)

    dtype = np.float32

    # no query points
    points = rng.random(size=(100, 3)).astype(dtype)
    queries = rng.random(size=(0, 3)).astype(dtype)
    radii = rng.uniform(0.1, 0.3, size=(0,)).astype(dtype)

    ans = ml3d.ops.radius_search(points, queries, radii, return_distances=True)

    assert ans.neighbors_index.shape.as_list() == [0]
    assert ans.neighbors_row_splits.shape.as_list() == [1]
    assert ans.neighbors_distance.shape.as_list() == [0]

    # no input points
    points = rng.random(size=(0, 3)).astype(dtype)
    queries = rng.random(size=(100, 3)).astype(dtype)
    radii = rng.uniform(0.1, 0.3, size=(100,)).astype(dtype)

    ans = ml3d.ops.radius_search(points, queries, radii, return_distances=True)

    assert ans.neighbors_index.shape.as_list() == [0]
    assert ans.neighbors_row_splits.shape.as_list() == [101]
    np.testing.assert_array_equal(np.zeros_like(ans.neighbors_row_splits),
                                  ans.neighbors_row_splits)
    assert ans.neighbors_distance.shape.as_list() == [0]
