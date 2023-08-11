# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
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

    np.testing.assert_allclose(y, y_conv3d, rtol=1e-3, atol=1e-5)


# yapf: disable
@pytest.mark.parametrize("kernel_size, out_channels, in_channels, with_inp_importance, with_normalization, batch_size",[
                             ([1,1,1],            2,           7,                True,              False,          2),
                             ([2,2,2],            1,           1,               False,              False,          3),
                             ([3,3,3],            4,           2,                True,               True,          3),
                             ([4,4,4],            3,           4,                True,               True,          8),
                             ([5,5,5],            5,           3,               False,               True,          8),
                        ])
# yapf: enable
@mltest.parametrize.ml
@pytest.mark.parametrize('dtype', [np.float32])
def test_compare_to_conv3d_batches(ml, dtype, kernel_size, out_channels,
                                   in_channels, with_inp_importance,
                                   with_normalization, batch_size):
    """Compares to the 3D convolution in tensorflow"""
    # the problem is specific to tensorflow
    if ml.module.__name__ != 'tensorflow':
        return

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

    # create array defining start and end of each batch
    inp_positions_row_splits = np.zeros(shape=(batch_size + 1,), dtype=np.int64)
    out_positions_row_splits = np.zeros(shape=(batch_size + 1,), dtype=np.int64)
    for i in range(batch_size - 1):
        inp_positions_row_splits[
            i + 1] = np.random.randint(15) + inp_positions_row_splits[i]
        out_positions_row_splits[
            i + 1] = np.random.randint(15) + out_positions_row_splits[i]

    inp_positions = np.unique(np.random.randint(0, max_grid_extent,
                                                (256, 3)).astype(dtype),
                              axis=0)

    inp_positions_row_splits[-1] = inp_positions.shape[0]

    if with_inp_importance:
        inp_importance = np.random.rand(
            inp_positions.shape[0]).astype(dtype) - 0.5
    else:
        inp_importance = np.empty((0,), dtype=dtype)
    out_positions = np.unique(np.random.randint(
        np.max(kernel_size) // 2, max_grid_extent - np.max(kernel_size) // 2,
        (256, 3)).astype(dtype),
                              axis=0)
    out_positions_row_splits[-1] = out_positions.shape[0]

    voxel_size = 0.2

    inp_features = np.random.uniform(size=inp_positions.shape[0:1] +
                                     (in_channels,)).astype(dtype)

    kernel_initializer = tf.constant_initializer(filters)
    bias_initializer = tf.constant_initializer(bias)

    sparse_conv = ml.layers.SparseConv(in_channels=in_channels,
                                       filters=out_channels,
                                       kernel_size=kernel_size,
                                       normalize=with_normalization,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer)

    inp_positions = tf.RaggedTensor.from_row_splits(
        values=inp_positions, row_splits=inp_positions_row_splits)
    out_positions = tf.RaggedTensor.from_row_splits(
        values=out_positions, row_splits=out_positions_row_splits)
    inp_features = tf.RaggedTensor.from_row_splits(
        values=inp_features, row_splits=inp_positions_row_splits)

    y = mltest.run_op(ml, ml.device, True, sparse_conv, inp_features,
                      inp_positions * voxel_size, out_positions * voxel_size,
                      voxel_size, inp_importance)
    for idx in range(batch_size):
        inp_pos = inp_positions[idx].numpy()
        inp_feat = inp_features[idx].numpy()
        out_pos = out_positions[idx].numpy()
        inp_pos_int = inp_pos.astype(np.int32)
        out_pos_int = out_pos.astype(np.int32)
        y_out = y[idx]

        # Compare the output to a standard 3d conv
        # store features in a volume to use standard 3d convs
        inp_volume = np.zeros(
            (1, max_grid_extent, max_grid_extent, max_grid_extent, in_channels),
            dtype=dtype)

        if with_inp_importance:
            inp_feat *= inp_importance[
                inp_positions_row_splits[idx]:inp_positions_row_splits[idx + 1],
                np.newaxis]
        inp_volume[0, inp_pos_int[:, 2], inp_pos_int[:, 1],
                   inp_pos_int[:, 0], :] = inp_feat

        conv3d = tf.keras.layers.Conv3D(
            out_channels,
            kernel_size,
            kernel_initializer=tf.constant_initializer(filters),
            use_bias=False,
            padding='same')
        y_conv3d = conv3d(inp_volume).numpy()

        # extract result at output positions
        y_conv3d = np.ascontiguousarray(y_conv3d[0, out_pos_int[:, 2],
                                                 out_pos_int[:, 1],
                                                 out_pos_int[:, 0], :])
        if with_normalization:
            for i, v in enumerate(y_conv3d):
                num_neighbors = mltest.to_numpy(
                    sparse_conv.nns.neighbors_row_splits[
                        out_positions_row_splits[idx] + i + 1] -
                    sparse_conv.nns.neighbors_row_splits[
                        out_positions_row_splits[idx] + i])
                if num_neighbors > 0:
                    v /= dtype(num_neighbors)

        y_conv3d += bias

        np.testing.assert_allclose(y_out, y_conv3d, rtol=1e-3, atol=1e-5)


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

    np.testing.assert_allclose(y, y_conv3d, rtol=1e-3, atol=1e-8)


# yapf: disable
@pytest.mark.parametrize("kernel_size, out_channels, in_channels, with_out_importance, with_normalization, batch_size",[
                             ([1,1,1],            2,           7,                True,              False,          2),
                             ([2,2,2],            1,           1,               False,              False,          3),
                             ([3,3,3],            4,           2,                True,               True,          3),
                             ([4,4,4],            3,           4,                True,               True,          8),
                             ([5,5,5],            5,           3,               False,               True,          8),
                        ])
# yapf: enable
@mltest.parametrize.ml
@pytest.mark.parametrize('dtype', [np.float32])
def test_compare_to_conv3dtranspose_batches(ml, dtype, kernel_size,
                                            out_channels, in_channels,
                                            with_out_importance,
                                            with_normalization, batch_size):
    """Compares to the 3D convolution in tensorflow"""
    # the problem is specific to tensorflow
    if ml.module.__name__ != 'tensorflow':
        return

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

    # create array defining start and end of each batch
    inp_positions_row_splits = np.zeros(shape=(batch_size + 1,), dtype=np.int64)
    out_positions_row_splits = np.zeros(shape=(batch_size + 1,), dtype=np.int64)
    for i in range(batch_size - 1):
        inp_positions_row_splits[
            i + 1] = np.random.randint(15) + inp_positions_row_splits[i]
        out_positions_row_splits[
            i + 1] = np.random.randint(15) + out_positions_row_splits[i]

    inp_positions = np.unique(np.random.randint(0, max_grid_extent,
                                                (512, 3)).astype(dtype),
                              axis=0)

    inp_positions_row_splits[-1] = inp_positions.shape[0]

    out_positions = np.unique(np.random.randint(
        np.max(kernel_size) // 2, max_grid_extent - np.max(kernel_size) // 2,
        (256, 3)).astype(dtype),
                              axis=0)
    out_positions_row_splits[-1] = out_positions.shape[0]

    if with_out_importance:
        out_importance = np.random.rand(
            out_positions.shape[0]).astype(dtype) - 0.5
    else:
        out_importance = np.empty((0,), dtype=dtype)

    voxel_size = 0.2

    inp_features = np.random.uniform(size=inp_positions.shape[0:1] +
                                     (in_channels,)).astype(dtype)

    kernel_initializer = tf.constant_initializer(filters)
    bias_initializer = tf.constant_initializer(bias)

    sparse_conv_transpose = ml.layers.SparseConvTranspose(
        in_channels=in_channels,
        filters=out_channels,
        kernel_size=kernel_size,
        normalize=with_normalization,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)

    inp_positions = tf.RaggedTensor.from_row_splits(
        values=inp_positions, row_splits=inp_positions_row_splits)
    out_positions = tf.RaggedTensor.from_row_splits(
        values=out_positions, row_splits=out_positions_row_splits)
    inp_features = tf.RaggedTensor.from_row_splits(
        values=inp_features, row_splits=inp_positions_row_splits)

    y = mltest.run_op(ml, ml.device, True, sparse_conv_transpose, inp_features,
                      inp_positions * voxel_size, out_positions * voxel_size,
                      voxel_size, out_importance)
    for idx in range(batch_size):
        inp_pos = inp_positions[idx].numpy()
        inp_feat = inp_features[idx].numpy()
        out_pos = out_positions[idx].numpy()
        inp_pos_int = inp_pos.astype(np.int32)
        out_pos_int = out_pos.astype(np.int32)
        y_out = y[idx]

        # Compare the output to a standard 3d conv
        # store features in a volume to use standard 3d convs
        inp_volume = np.zeros(
            (1, max_grid_extent, max_grid_extent, max_grid_extent, in_channels),
            dtype=dtype)

        if with_normalization:
            for i, v in enumerate(inp_feat):
                num_neighbors = mltest.to_numpy(
                    sparse_conv_transpose.nns_inp.neighbors_row_splits[
                        inp_positions_row_splits[idx] + i + 1] -
                    sparse_conv_transpose.nns_inp.neighbors_row_splits[
                        inp_positions_row_splits[idx] + i])
                if num_neighbors:
                    v /= dtype(num_neighbors)

        inp_volume[0, inp_pos_int[:, 2], inp_pos_int[:, 1],
                   inp_pos_int[:, 0], :] = inp_feat

        conv3d = tf.keras.layers.Conv3DTranspose(
            out_channels,
            kernel_size,
            kernel_initializer=tf.constant_initializer(
                filters.transpose([0, 1, 2, 4, 3])),
            use_bias=False,
            padding='same')
        y_conv3d = conv3d(inp_volume).numpy()

        # extract result at output positions
        y_conv3d = np.ascontiguousarray(y_conv3d[0, out_pos_int[:, 2],
                                                 out_pos_int[:, 1],
                                                 out_pos_int[:, 0], :])

        if with_out_importance:
            y_conv3d *= out_importance[
                out_positions_row_splits[idx]:out_positions_row_splits[idx + 1],
                np.newaxis]

        y_conv3d += bias

        np.testing.assert_allclose(y_out, y_conv3d, rtol=1e-3, atol=1e-5)
