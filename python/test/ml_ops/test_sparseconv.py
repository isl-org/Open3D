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
"""Tests the reference python implementation of the sparse conv"""

import open3d as o3d
import numpy as np
np.set_printoptions(linewidth=600)
np.set_printoptions(threshold=np.inf)
import pytest
import mltest

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks


# yapf: disable
@pytest.mark.parametrize("kernel_size, out_channels, in_channels, with_inp_importance, with_normalization",[
                             ([1,1,1],            2,           7,                True,              False),
                             ([2,2,2],            1,           1,               False,              False),
                             ([3,3,3],            4,           2,                True,               True),
                             ([4,4,4],            3,           4,                True,               True),
                             ([5,5,5],            5,           3,               False,               True),
                        ])
# yapf: enable
@mltest.parametrize.ml
@pytest.mark.parametrize('dtype', [np.float32])
def test_compare_to_conv3d(ml, dtype, kernel_size, out_channels, in_channels,
                           with_inp_importance, with_normalization):
    """Compares to the 3D convolution in tensorflow"""

    # This test requires tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        return

    np.random.seed(0)

    filters = np.random.random(size=(*kernel_size, in_channels,
                                     out_channels)).astype(dtype)
    bias = np.random.random(size=(out_channels,)).astype(dtype)

    max_grid_extent = 10
    inp_positions = np.unique(np.random.randint(0, max_grid_extent,
                                                (256, 3)).astype(dtype),
                              axis=0)
    inp_positions_int = inp_positions.astype(np.int32)
    if with_inp_importance:
        inp_importance = np.random.rand(
            inp_positions.shape[0]).astype(dtype) - 0.5
    else:
        inp_importance = np.empty((0,), dtype=dtype)
    out_positions = np.unique(np.random.randint(
        np.max(kernel_size) // 2, max_grid_extent - np.max(kernel_size) // 2,
        (5, 3)).astype(dtype),
                              axis=0)
    out_positions_int = out_positions.astype(np.int32)

    voxel_size = 0.2

    inp_features = np.random.uniform(size=inp_positions.shape[0:1] +
                                     (in_channels,)).astype(dtype)

    if ml.module.__name__ == 'tensorflow':
        kernel_initializer = tf.constant_initializer(filters)
        bias_initializer = tf.constant_initializer(bias)
    elif ml.module.__name__ == 'torch':
        torch = ml.module

        def kernel_initializer(a):
            a.data = torch.from_numpy(filters)

        def bias_initializer(a):
            a.data = torch.from_numpy(bias)
    else:
        raise Exception('Unsupported ml framework {}'.format(
            ml.module.__name__))

    sparse_conv = ml.layers.SparseConv(in_channels=in_channels,
                                       filters=out_channels,
                                       kernel_size=kernel_size,
                                       normalize=with_normalization,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer)
    if ml.module.__name__ == 'torch':
        sparse_conv.to(ml.device)

    y = mltest.run_op(ml, ml.device, True, sparse_conv, inp_features,
                      inp_positions * voxel_size, out_positions * voxel_size,
                      voxel_size, inp_importance)

    # Compare the output to a standard 3d conv
    # store features in a volume to use standard 3d convs
    inp_volume = np.zeros(
        (1, max_grid_extent, max_grid_extent, max_grid_extent, in_channels),
        dtype=dtype)

    if with_inp_importance:
        inp_features *= inp_importance[:, np.newaxis]
    inp_volume[0, inp_positions_int[:, 2], inp_positions_int[:, 1],
               inp_positions_int[:, 0], :] = inp_features

    conv3d = tf.keras.layers.Conv3D(
        out_channels,
        kernel_size,
        kernel_initializer=tf.constant_initializer(filters),
        use_bias=False,
        padding='same')
    y_conv3d = conv3d(inp_volume).numpy()

    # extract result at output positions
    y_conv3d = np.ascontiguousarray(y_conv3d[0, out_positions_int[:, 2],
                                             out_positions_int[:, 1],
                                             out_positions_int[:, 0], :])

    if with_normalization:
        for i, v in enumerate(y_conv3d):
            num_neighbors = mltest.to_numpy(
                sparse_conv.nns.neighbors_row_splits[i + 1] -
                sparse_conv.nns.neighbors_row_splits[i])
            v /= dtype(num_neighbors)

    y_conv3d += bias

    np.testing.assert_allclose(y, y_conv3d, rtol=1e-5, atol=1e-8)


# yapf: disable
@pytest.mark.parametrize("kernel_size, out_channels, in_channels, with_out_importance, with_normalization",[
                             ([1,1,1],            2,           7,                True,              False),
                             ([2,2,2],            1,           1,               False,              False),
                             ([3,3,3],            4,           2,                True,               True),
                             ([4,4,4],            3,           4,                True,               True),
                             ([5,5,5],            5,           3,               False,               True),
                        ])
# yapf: enable
@mltest.parametrize.ml
@pytest.mark.parametrize('dtype', [np.float32])
def test_compare_to_conv3dtranspose(ml, dtype, kernel_size, out_channels,
                                    in_channels, with_out_importance,
                                    with_normalization):
    """Compares to the 3D transposed convolution in tensorflow"""
    # This test requires tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        return

    np.random.seed(0)

    filters = np.random.random(size=(*kernel_size, in_channels,
                                     out_channels)).astype(dtype)
    bias = np.random.random(size=(out_channels,)).astype(dtype)

    max_grid_extent = 10
    inp_positions = np.unique(np.random.randint(0, max_grid_extent,
                                                (512, 3)).astype(dtype),
                              axis=0)
    inp_positions_int = inp_positions.astype(np.int32)
    out_positions = np.unique(np.random.randint(
        np.max(kernel_size) // 2, max_grid_extent - np.max(kernel_size) // 2,
        (5, 3)).astype(dtype),
                              axis=0)

    if with_out_importance:
        out_importance = np.random.rand(
            out_positions.shape[0]).astype(dtype) - 0.5
    else:
        out_importance = np.empty((0,), dtype=dtype)
    out_positions_int = out_positions.astype(np.int32)

    voxel_size = 0.2

    inp_features = np.random.uniform(size=inp_positions.shape[0:1] +
                                     (in_channels,)).astype(dtype)

    if ml.module.__name__ == 'tensorflow':
        kernel_initializer = tf.constant_initializer(filters)
        bias_initializer = tf.constant_initializer(bias)
    elif ml.module.__name__ == 'torch':
        torch = ml.module

        def kernel_initializer(a):
            a.data = torch.from_numpy(filters)

        def bias_initializer(a):
            a.data = torch.from_numpy(bias)
    else:
        raise Exception('Unsupported ml framework {}'.format(
            ml.module.__name__))

    sparse_conv_transpose = ml.layers.SparseConvTranspose(
        in_channels=in_channels,
        filters=out_channels,
        kernel_size=kernel_size,
        normalize=with_normalization,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)

    if ml.module.__name__ == 'torch':
        sparse_conv_transpose.to(ml.device)

    y = mltest.run_op(ml, ml.device, True, sparse_conv_transpose, inp_features,
                      inp_positions * voxel_size, out_positions * voxel_size,
                      voxel_size, out_importance)

    # Compare the output to a standard 3d conv
    # store features in a volume to use standard 3d convs
    inp_volume = np.zeros(
        (1, max_grid_extent, max_grid_extent, max_grid_extent, in_channels),
        dtype=dtype)

    if with_normalization:
        for i, v in enumerate(inp_features):
            num_neighbors = mltest.to_numpy(
                sparse_conv_transpose.nns_inp.neighbors_row_splits[i + 1] -
                sparse_conv_transpose.nns_inp.neighbors_row_splits[i])
            if num_neighbors:
                v /= dtype(num_neighbors)

    inp_volume[0, inp_positions_int[:, 2], inp_positions_int[:, 1],
               inp_positions_int[:, 0], :] = inp_features

    conv3d = tf.keras.layers.Conv3DTranspose(
        out_channels,
        kernel_size,
        kernel_initializer=tf.constant_initializer(
            filters.transpose([0, 1, 2, 4, 3])),
        use_bias=False,
        padding='same')
    y_conv3d = conv3d(inp_volume).numpy()

    # extract result at output positions
    y_conv3d = np.ascontiguousarray(y_conv3d[0, out_positions_int[:, 2],
                                             out_positions_int[:, 1],
                                             out_positions_int[:, 0], :])

    if with_out_importance:
        y_conv3d *= out_importance[:, np.newaxis]

    y_conv3d += bias

    np.testing.assert_allclose(y, y_conv3d, rtol=1e-5, atol=1e-8)
