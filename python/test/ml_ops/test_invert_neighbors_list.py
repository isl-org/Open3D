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
import pytest
import mltest

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks

# the supported dtypes for the attributes
value_dtypes = pytest.mark.parametrize(
    'dtype', [np.int32, np.int64, np.float32, np.float64])

attributes = pytest.mark.parametrize('attributes',
                                     ['scalar', 'none', 'multidim'])


@value_dtypes
@attributes
@mltest.parametrize.ml
def test_invert_neighbors_list(dtype, attributes, ml):

    # yapf: disable

    # define connectivity for 3 query points and 3 input points
    num_points = 3
    edges = np.array(
        [
            [0, 0], [0, 1], [0, 2],  # 3 neighbors
            [1, 2],                  # 1 neighbors
            [2, 1], [2, 2],          # 2 neighbors
        ],
        dtype=np.int32)

    # the neighbors_index is the second column
    neighbors_index = edges[:, 1]

    # exclusive prefix sum of the number of neighbors
    neighbors_row_splits = np.array([0, 3, 4, edges.shape[0]], dtype=np.int64)

    if attributes == 'scalar':
        neighbors_attributes = np.array([
            10, 20, 30,
            40,
            50, 60,
        ], dtype=dtype)
    elif attributes == 'none':
        neighbors_attributes = np.array([], dtype=dtype)
    elif attributes == 'multidim':
        neighbors_attributes = np.array([
            [10, 1], [20, 2], [30, 3],
            [40, 4],
            [50, 5], [60, 6],
        ], dtype=dtype)

# yapf: enable

    ans = mltest.run_op(ml,
                        ml.device,
                        True,
                        ml.ops.invert_neighbors_list,
                        num_points=num_points,
                        inp_neighbors_index=neighbors_index,
                        inp_neighbors_row_splits=neighbors_row_splits,
                        inp_neighbors_attributes=neighbors_attributes)

    expected_neighbors_row_splits = [0, 1, 3, edges.shape[0]]
    np.testing.assert_equal(ans.neighbors_row_splits,
                            expected_neighbors_row_splits)

    # checking the neighbors_index is more complicated because the order
    # of the neighbors for each query point is not defined.
    expected_neighbors_index = [
        set([0]),
        set([0, 2]),
        set([0, 1, 2]),
    ]
    for i, expected_neighbors_i in enumerate(expected_neighbors_index):
        start = ans.neighbors_row_splits[i]
        end = ans.neighbors_row_splits[i + 1]
        neighbors_i = set(ans.neighbors_index[start:end])
        assert neighbors_i == expected_neighbors_i

    if neighbors_attributes.shape == (0,):
        # if the input is a zero length vector then the returned attributes
        # vector also must be a zero length vector
        assert ans.neighbors_attributes.shape == (0,)
    else:
        # check if the attributes are still associated with the same edge
        edge_attr_map = {
            tuple(k): v for k, v in zip(edges, neighbors_attributes)
        }
        for i, _ in enumerate(expected_neighbors_index):
            start = ans.neighbors_row_splits[i]
            end = ans.neighbors_row_splits[i + 1]

            # neighbors and attributes for point i
            neighbors_i = ans.neighbors_index[start:end]
            attributes_i = ans.neighbors_attributes[start:end]
            for j, attr in zip(neighbors_i, attributes_i):
                key = (j, i)
                np.testing.assert_equal(attr, edge_attr_map[key])


@mltest.parametrize.ml
def test_invert_neighbors_list_shape_checking(ml):

    num_points = 3
    inp_neighbors_index = np.array([0, 1, 2, 2, 1, 2], dtype=np.int32)
    inp_neighbors_row_splits = np.array([0, 3, 4, 6], dtype=np.int64)
    inp_neighbors_attributes = np.array([10, 20, 30, 40, 50, 60],
                                        dtype=np.float32)

    # test the shape checking by passing arrays with wrong rank and/or size
    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(ml,
                          ml.cpu_device,
                          False,
                          ml.ops.invert_neighbors_list,
                          num_points=num_points,
                          inp_neighbors_index=inp_neighbors_index[1:],
                          inp_neighbors_row_splits=inp_neighbors_row_splits,
                          inp_neighbors_attributes=inp_neighbors_attributes)
    assert 'invalid shape' in str(einfo.value)

    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(ml,
                          ml.cpu_device,
                          False,
                          ml.ops.invert_neighbors_list,
                          num_points=num_points,
                          inp_neighbors_index=inp_neighbors_index[:,
                                                                  np.newaxis],
                          inp_neighbors_row_splits=inp_neighbors_row_splits,
                          inp_neighbors_attributes=inp_neighbors_attributes)
    assert 'invalid shape' in str(einfo.value)

    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(
            ml,
            ml.cpu_device,
            False,
            ml.ops.invert_neighbors_list,
            num_points=num_points,
            inp_neighbors_index=inp_neighbors_index,
            inp_neighbors_row_splits=inp_neighbors_row_splits[:, np.newaxis],
            inp_neighbors_attributes=inp_neighbors_attributes)
    assert 'invalid shape' in str(einfo.value)

    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(ml,
                          ml.cpu_device,
                          False,
                          ml.ops.invert_neighbors_list,
                          num_points=num_points,
                          inp_neighbors_index=inp_neighbors_index,
                          inp_neighbors_row_splits=inp_neighbors_row_splits,
                          inp_neighbors_attributes=inp_neighbors_attributes[1:])
    assert 'invalid shape' in str(einfo.value)
