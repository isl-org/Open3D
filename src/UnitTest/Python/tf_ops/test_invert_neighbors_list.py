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
import mark_helper

# skip all tests if the tf ops were not built and disable warnings caused by
# tensorflow
pytestmark = mark_helper.tf_marks

# the supported dtypes for the attributes
value_dtypes = pytest.mark.parametrize(
    'dtype', [np.int32, np.int64, np.float32, np.float64])

attributes = pytest.mark.parametrize(
    'attributes', ['scalar', 'none', 'multidim'])

@value_dtypes
@attributes
@mark_helper.devices
def test_invert_neighbors_list(dtype, attributes, device_name):
    import tensorflow as tf
    import open3d.ml.tf as ml3d

    # define connectivity for 3 query points and 3 input points
    num_points = 3
    edges = np.array(
            [ 
                [0, 0], [0, 1], [0, 2], # 3 neighbors
                [1, 2],                 # 1 neighbors
                [2, 1], [2, 2],         # 2 neighbors
            ],
            dtype = np.int32 )

    # the neighbors_index is the second column
    neighbors_index = edges[:,1]

    # exclusive prefix sum of the number of neighbors
    neighbors_prefix_sum = np.array([ 0, 3, 4 ], dtype=np.int64)

    if attributes == 'scalar':
        neighbors_attributes = np.array([ 
            10, 20, 30,
            40,
            50, 60,
            ],
            dtype=dtype)
    elif attributes == 'none':
        neighbors_attributes = np.array([], dtype=dtype)
    elif attributes == 'multidim':
        neighbors_attributes = np.array([ 
            [10,1], [20,2], [30,3],
            [40,4],
            [50,5], [60,6],
            ],
            dtype=dtype)


    with tf.device(device_name):
        ans = ml3d.ops.invert_neighbors_list(
                num_points, 
                neighbors_index, 
                neighbors_prefix_sum, 
                neighbors_attributes )
        assert device_name in ans.neighbors_index.device

    expected_neighbors_prefix_sum = [0, 1, 3]
    np.testing.assert_equal(ans.neighbors_prefix_sum.numpy(), expected_neighbors_prefix_sum)


    # checking the neighbors_index is more complicated because the order
    # of the neighbors for each query point is not defined. 
    expected_neighbors_index = [
            set([0]),       
            set([0, 2]),
            set([0, 1, 2]),
            ]
    for i, expected_neighbors_i in enumerate(expected_neighbors_index):
        start = ans.neighbors_prefix_sum[i]
        end = ans.neighbors_prefix_sum[i+1] if i+1 < len(ans.neighbors_prefix_sum) else len(neighbors_index)
        neighbors_i = set(ans.neighbors_index[start:end].numpy())
        assert neighbors_i == expected_neighbors_i

    if neighbors_attributes.shape == (0,):
        # if the input is a zero length vector then the returned attributes
        # vector also must be a zero length vector
        assert ans.neighbors_attributes.numpy().shape == (0,)
    else:
        # check if the attributes are still associated with the same edge
        edge_attr_map = { tuple(k): v for k,v in zip(edges, neighbors_attributes) }
        for i, _ in enumerate(expected_neighbors_index):
            start = ans.neighbors_prefix_sum[i]
            end = ans.neighbors_prefix_sum[i+1] if i+1 < len(ans.neighbors_prefix_sum) else len(neighbors_index)

            # neighbors and attributes for point i
            neighbors_i = ans.neighbors_index[start:end].numpy()
            attributes_i = ans.neighbors_attributes[start:end].numpy()
            for j, attr in zip(neighbors_i, attributes_i):
                key = (j,i)
                np.testing.assert_equal(attr, edge_attr_map[key])

