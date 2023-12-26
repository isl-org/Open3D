# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Tests the implementation of the general sparse convolution ops"""

import open3d as o3d
import numpy as np
np.set_printoptions(linewidth=600)
np.set_printoptions(threshold=np.inf)
import pytest
import mltest
from check_gradients import check_gradients

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks


# yapf: disable
@pytest.mark.parametrize("kernel_size, out_channels, in_channels, with_inp_importance, with_neighbors_importance, with_normalization", [
                             (1,            2,           7,                True,                     False,              False),
                             (2,            1,           1,               False,                     False,              False),
                             (3,            5,           3,               False,                      True,               True),
                             (33,           3,           4,               False,                      True,              False),
                        ])
# yapf: enable
@mltest.parametrize.ml
@pytest.mark.parametrize('dtype', [np.float32])
def test_sparseconv_gradient(ml, dtype, kernel_size, out_channels, in_channels,
                             with_inp_importance, with_neighbors_importance,
                             with_normalization):

    if dtype == np.float64:
        tolerance = {'atol': 1e-5, 'rtol': 1e-2, 'epsilon': 1e-6}
    elif dtype == np.float32:
        tolerance = {'atol': 1e-2, 'rtol': 1e-1, 'epsilon': 1e-3}

    rng = np.random.RandomState(123)

    conv_attrs = {
        'normalize': with_normalization,
    }

    filters = rng.random(size=(kernel_size, in_channels,
                               out_channels)).astype(dtype)

    num_inp = 33
    num_out = 16

    inp_features = rng.uniform(size=(num_inp, in_channels)).astype(dtype)
    if (with_inp_importance):
        inp_importance = rng.random(num_inp).astype(dtype)
    else:
        inp_importance = np.empty((0,)).astype(dtype)

    neighbors_row_splits = np.zeros((num_out + 1,), dtype=np.int64)

    for i in range(num_out):
        neighbors_row_splits[i + 1] = rng.randint(kernel_size +
                                                  1) + neighbors_row_splits[i]

    neighbors_index = np.zeros((neighbors_row_splits[-1],), dtype=np.int32)
    neighbors_kernel_index = np.zeros((neighbors_row_splits[-1],),
                                      dtype=np.uint8)
    for i in range(num_out):
        start = neighbors_row_splits[i]
        end = neighbors_row_splits[i + 1]
        neighbors_kernel_index[start:end] = rng.choice(kernel_size,
                                                       [end - start],
                                                       replace=False)
        neighbors_index[start:end] = rng.choice(num_inp, [end - start],
                                                replace=False)

    arange = np.arange(neighbors_index.shape[0])
    inv_neighbors_index, inv_neighbors_row_splits, inv_arange = mltest.run_op(
        ml, ml.device, False, ml.ops.invert_neighbors_list, num_inp,
        neighbors_index, neighbors_row_splits, arange)

    inv_neighbors_kernel_index = neighbors_kernel_index[inv_arange]
    if with_neighbors_importance:
        neighbors_importance = rng.random(
            neighbors_index.shape[0]).astype(dtype) - 0.5

        neighbors_importance_sum = mltest.run_op(ml, ml.device, False,
                                                 ml.ops.reduce_subarrays_sum,
                                                 neighbors_importance,
                                                 neighbors_row_splits)

        inv_neighbors_importance = neighbors_importance[inv_arange]
    else:
        neighbors_importance = np.empty((0,), dtype=dtype)
        neighbors_importance_sum = np.empty((0,), dtype=dtype)
        inv_neighbors_importance = np.empty((0,), dtype=dtype)

    # define functions for the gradient checker
    def sparse_conv_infeats(inp_features):
        return mltest.run_op(ml, ml.device, True, ml.ops.sparse_conv, filters,
                             inp_features, inp_importance, neighbors_index,
                             neighbors_kernel_index, neighbors_importance,
                             neighbors_row_splits, **conv_attrs)

    def sparse_conv_filter(filters):
        return mltest.run_op(ml, ml.device, True, ml.ops.sparse_conv, filters,
                             inp_features, inp_importance, neighbors_index,
                             neighbors_kernel_index, neighbors_importance,
                             neighbors_row_splits, **conv_attrs)

    def sparse_conv_filter_backprop(out_features_gradient, filters):
        return mltest.run_op_grad(ml,
                                  ml.device,
                                  True,
                                  ml.ops.sparse_conv,
                                  filters,
                                  '',
                                  out_features_gradient,
                                  filters=filters,
                                  inp_features=inp_features,
                                  inp_importance=inp_importance,
                                  neighbors_index=neighbors_index,
                                  neighbors_kernel_index=neighbors_kernel_index,
                                  neighbors_importance=neighbors_importance,
                                  neighbors_row_splits=neighbors_row_splits,
                                  **conv_attrs)

    def sparse_conv_infeat_backprop(out_features_gradient, inp_features):
        return mltest.run_op_grad(ml,
                                  ml.device,
                                  True,
                                  ml.ops.sparse_conv,
                                  inp_features,
                                  '',
                                  out_features_gradient,
                                  filters=filters,
                                  inp_features=inp_features,
                                  inp_importance=inp_importance,
                                  neighbors_index=neighbors_index,
                                  neighbors_kernel_index=neighbors_kernel_index,
                                  neighbors_importance=neighbors_importance,
                                  neighbors_row_splits=neighbors_row_splits,
                                  **conv_attrs)

    def sparse_conv_transpose_filter(filters):
        return mltest.run_op(ml, ml.device, True, ml.ops.sparse_conv_transpose,
                             filters, inp_importance, y_arr, neighbors_index,
                             neighbors_importance_sum, neighbors_row_splits,
                             inv_neighbors_index, inv_neighbors_kernel_index,
                             inv_neighbors_importance, inv_neighbors_row_splits,
                             **conv_attrs)

    def sparse_conv_transpose_infeats(inp_features):
        return mltest.run_op(ml, ml.device, True, ml.ops.sparse_conv_transpose,
                             filters.transpose([0, 2, 1]), inp_importance,
                             inp_features, neighbors_index,
                             neighbors_importance_sum, neighbors_row_splits,
                             inv_neighbors_index, inv_neighbors_kernel_index,
                             inv_neighbors_importance, inv_neighbors_row_splits,
                             **conv_attrs)

    def sparse_conv_transpose_filter_backprop(out_features_gradient, filters):
        return mltest.run_op_grad(
            ml,
            ml.device,
            True,
            ml.ops.sparse_conv_transpose,
            filters,
            '',
            out_features_gradient,
            filters=filters,
            out_importance=inp_importance,
            inp_features=y_arr,
            inp_neighbors_index=neighbors_index,
            inp_neighbors_importance_sum=neighbors_importance_sum,
            inp_neighbors_row_splits=neighbors_row_splits,
            neighbors_index=inv_neighbors_index,
            neighbors_kernel_index=inv_neighbors_kernel_index,
            neighbors_importance=inv_neighbors_importance,
            neighbors_row_splits=inv_neighbors_row_splits,
            **conv_attrs)

    def sparse_conv_transpose_infeat_backprop(out_features_gradient,
                                              inp_features):
        return mltest.run_op_grad(
            ml,
            ml.device,
            True,
            ml.ops.sparse_conv_transpose,
            inp_features,
            '',
            out_features_gradient,
            filters=filters.transpose([0, 2, 1]),
            out_importance=inp_importance,
            inp_features=inp_features,
            inp_neighbors_index=neighbors_index,
            inp_neighbors_importance_sum=neighbors_importance_sum,
            inp_neighbors_row_splits=neighbors_row_splits,
            neighbors_index=inv_neighbors_index,
            neighbors_kernel_index=inv_neighbors_kernel_index,
            neighbors_importance=inv_neighbors_importance,
            neighbors_row_splits=inv_neighbors_row_splits,
            **conv_attrs)

    y_arr = sparse_conv_infeats(inp_features)

    dbg = {}
    filter_gradient_OK = check_gradients(filters,
                                         sparse_conv_filter,
                                         sparse_conv_filter_backprop,
                                         debug_outputs=dbg,
                                         **tolerance)
    assert filter_gradient_OK

    feature_gradient_OK = check_gradients(inp_features,
                                          sparse_conv_infeats,
                                          sparse_conv_infeat_backprop,
                                          debug_outputs=dbg,
                                          **tolerance)

    assert feature_gradient_OK

    transpose_filter_gradient_OK = check_gradients(
        filters.transpose([0, 2, 1]),
        sparse_conv_transpose_filter,
        sparse_conv_transpose_filter_backprop,
        debug_outputs=dbg,
        **tolerance)
    assert transpose_filter_gradient_OK

    transpose_feature_gradient_OK = check_gradients(
        y_arr,
        sparse_conv_transpose_infeats,
        sparse_conv_transpose_infeat_backprop,
        debug_outputs=dbg,
        **tolerance)
    assert transpose_feature_gradient_OK
