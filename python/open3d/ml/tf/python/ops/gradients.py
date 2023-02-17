# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
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

from .lib import _lib
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops


@_ops.RegisterGradient("Open3DVoxelPooling")
def _voxel_pooling_grad(op, grad_pos, grad_feat):
    features_grad = _lib.open3d_voxel_pooling_grad(
        positions=op.inputs[0],
        features=op.inputs[1],
        voxel_size=op.inputs[2],
        pooled_positions=op.outputs[0],
        pooled_features_gradient=grad_feat,
        position_fn=op.get_attr('position_fn'),
        feature_fn=op.get_attr('feature_fn'),
    )
    return [None, features_grad, None]


@_ops.RegisterGradient("Open3DContinuousConv")
def _continuous_conv_grad(op, grad):

    filters = op.inputs[0]
    out_positions = op.inputs[1]
    extents = op.inputs[2]
    offset = op.inputs[3]
    inp_positions = op.inputs[4]
    inp_features = op.inputs[5]
    inp_importance = op.inputs[6]
    neighbors_index = op.inputs[7]
    neighbors_importance = op.inputs[8]
    neighbors_row_splits = op.inputs[9]

    filter_grad = _lib.open3d_continuous_conv_backprop_filter(
        align_corners=op.get_attr('align_corners'),
        interpolation=op.get_attr('interpolation'),
        coordinate_mapping=op.get_attr('coordinate_mapping'),
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=filters,
        out_positions=out_positions,
        extents=extents,
        offset=offset,
        inp_positions=inp_positions,
        inp_features=inp_features,
        inp_importance=inp_importance,
        neighbors_index=neighbors_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        out_features_gradient=grad,
    )

    # invert the neighbors list
    num_points = _tf.shape(inp_positions, out_type=_tf.int64)[0]
    inv_neighbors_index, inv_neighbors_row_splits, inv_neighbors_importance = _lib.open3d_invert_neighbors_list(
        num_points, neighbors_index, neighbors_row_splits, neighbors_importance)

    neighbors_importance_sum = _lib.open3d_reduce_subarrays_sum(
        neighbors_importance, neighbors_row_splits)

    inp_features_grad = _lib.open3d_continuous_conv_transpose(
        align_corners=op.get_attr('align_corners'),
        interpolation=op.get_attr('interpolation'),
        coordinate_mapping=op.get_attr('coordinate_mapping'),
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=_tf.transpose(filters, [0, 1, 2, 4, 3]),
        out_positions=inp_positions,
        out_importance=inp_importance,
        extents=extents,
        offset=offset,
        inp_positions=out_positions,
        inp_features=grad,
        inp_neighbors_importance_sum=neighbors_importance_sum,
        inp_neighbors_index=neighbors_index,
        inp_neighbors_row_splits=neighbors_row_splits,
        neighbors_index=inv_neighbors_index,
        neighbors_importance=inv_neighbors_importance,
        neighbors_row_splits=inv_neighbors_row_splits,
    )

    return [filter_grad] + [None] * 4 + [inp_features_grad] + [None] * 4


@_ops.RegisterGradient("Open3DContinuousConvTranspose")
def _continuous_conv_transpose_grad(op, grad):

    filters = op.inputs[0]
    out_positions = op.inputs[1]
    out_importance = op.inputs[2]
    extents = op.inputs[3]
    offset = op.inputs[4]
    inp_positions = op.inputs[5]
    inp_features = op.inputs[6]
    # unused inp_neighbors_index = op.inputs[7]
    inp_neighbors_importance_sum = op.inputs[8]
    inp_neighbors_row_splits = op.inputs[9]
    neighbors_index = op.inputs[10]
    neighbors_importance = op.inputs[11]
    neighbors_row_splits = op.inputs[12]

    filter_grad = _lib.open3d_continuous_conv_transpose_backprop_filter(
        align_corners=op.get_attr('align_corners'),
        interpolation=op.get_attr('interpolation'),
        coordinate_mapping=op.get_attr('coordinate_mapping'),
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=filters,
        out_positions=out_positions,
        out_importance=out_importance,
        extents=extents,
        offset=offset,
        inp_positions=inp_positions,
        inp_features=inp_features,
        inp_neighbors_importance_sum=inp_neighbors_importance_sum,
        inp_neighbors_row_splits=inp_neighbors_row_splits,
        neighbors_index=neighbors_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        out_features_gradient=grad,
    )

    # invert the neighbors list
    num_points = _tf.shape(inp_positions, out_type=_tf.int64)[0]
    inv_neighbors_index, _, inv_neighbors_importance = _lib.open3d_invert_neighbors_list(
        num_points, neighbors_index, neighbors_row_splits, neighbors_importance)

    inp_features_grad = _lib.open3d_continuous_conv(
        align_corners=op.get_attr('align_corners'),
        interpolation=op.get_attr('interpolation'),
        coordinate_mapping=op.get_attr('coordinate_mapping'),
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=_tf.transpose(filters, [0, 1, 2, 4, 3]),
        out_positions=inp_positions,
        extents=extents,
        offset=offset,
        inp_positions=out_positions,
        inp_features=grad,
        inp_importance=out_importance,
        neighbors_index=inv_neighbors_index,
        neighbors_importance=inv_neighbors_importance,
        neighbors_row_splits=inp_neighbors_row_splits,
    )

    return [filter_grad] + [None] * 5 + [inp_features_grad] + [None] * 6


@_ops.RegisterGradient("Open3DSparseConv")
def _sparse_conv_grad(op, grad):

    filters = op.inputs[0]
    inp_features = op.inputs[1]
    inp_importance = op.inputs[2]
    neighbors_index = op.inputs[3]
    neighbors_kernel_index = op.inputs[4]
    neighbors_importance = op.inputs[5]
    neighbors_row_splits = op.inputs[6]

    filter_grad = _lib.open3d_sparse_conv_backprop_filter(
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=filters,
        inp_features=inp_features,
        inp_importance=inp_importance,
        neighbors_index=neighbors_index,
        neighbors_kernel_index=neighbors_kernel_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        out_features_gradient=grad,
    )

    # invert the neighbors list
    num_points = _tf.shape(inp_features, out_type=_tf.int64)[0]
    arange = _tf.range(0, _tf.shape(neighbors_index)[0])
    inv_neighbors_index, inv_neighbors_row_splits, inv_arange = _lib.open3d_invert_neighbors_list(
        num_points, neighbors_index, neighbors_row_splits, arange)

    inv_neighbors_kernel_index = _tf.gather(neighbors_kernel_index, inv_arange)
    inv_neighbors_importance = _tf.cond(
        _tf.shape(neighbors_importance)[0] > 0,
        true_fn=lambda: _tf.gather(neighbors_importance, inv_arange),
        false_fn=lambda: _tf.ones((0,), dtype=_tf.float32))

    neighbors_importance_sum = _lib.open3d_reduce_subarrays_sum(
        neighbors_importance, neighbors_row_splits)

    inp_features_grad = _lib.open3d_sparse_conv_transpose(
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=_tf.transpose(filters, [0, 2, 1]),
        out_importance=inp_importance,
        inp_features=grad,
        inp_neighbors_importance_sum=neighbors_importance_sum,
        inp_neighbors_index=neighbors_index,
        inp_neighbors_row_splits=neighbors_row_splits,
        neighbors_index=inv_neighbors_index,
        neighbors_kernel_index=inv_neighbors_kernel_index,
        neighbors_importance=inv_neighbors_importance,
        neighbors_row_splits=inv_neighbors_row_splits,
    )

    return [filter_grad, inp_features_grad] + [None] * 5


@_ops.RegisterGradient("Open3DSparseConvTranspose")
def _sparse_conv_transpose_grad(op, grad):

    filters = op.inputs[0]
    out_importance = op.inputs[1]
    inp_features = op.inputs[2]
    inp_neighbors_importance_sum = op.inputs[4]
    inp_neighbors_row_splits = op.inputs[5]
    neighbors_index = op.inputs[6]
    neighbors_kernel_index = op.inputs[7]
    neighbors_importance = op.inputs[8]
    neighbors_row_splits = op.inputs[9]

    filter_grad = _lib.open3d_sparse_conv_transpose_backprop_filter(
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=filters,
        out_importance=out_importance,
        inp_features=inp_features,
        inp_neighbors_importance_sum=inp_neighbors_importance_sum,
        inp_neighbors_row_splits=inp_neighbors_row_splits,
        neighbors_index=neighbors_index,
        neighbors_kernel_index=neighbors_kernel_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        out_features_gradient=grad,
    )

    # invert the neighbors list
    num_points = _tf.shape(inp_features, out_type=_tf.int64)[0]
    arange = _tf.range(0, _tf.shape(neighbors_index)[0])
    inv_neighbors_index, _, inv_arange = _lib.open3d_invert_neighbors_list(
        num_points, neighbors_index, neighbors_row_splits, arange)

    inv_neighbors_kernel_index = _tf.gather(neighbors_kernel_index, inv_arange)
    if _tf.shape(neighbors_importance)[0] > 0:
        inv_neighbors_importance = _tf.gather(neighbors_importance, inv_arange)
    else:
        inv_neighbors_importance = _tf.ones((0,), dtype=_tf.float32)

    inp_features_grad = _lib.open3d_sparse_conv(
        normalize=op.get_attr('normalize'),
        max_temp_mem_MB=op.get_attr('max_temp_mem_MB'),
        filters=_tf.transpose(filters, [0, 2, 1]),
        inp_features=grad,
        inp_importance=out_importance,
        neighbors_index=inv_neighbors_index,
        neighbors_kernel_index=inv_neighbors_kernel_index,
        neighbors_importance=inv_neighbors_importance,
        neighbors_row_splits=inp_neighbors_row_splits,
    )

    return [filter_grad] + [None] + [inp_features_grad] + [None] * 7


@_ops.RegisterGradient("Open3DTrilinearDevoxelize")
def _trilinear_devoxelize_gradient(op, grad_out, grad_inds, grad_wgts):

    inds = op.outputs[1]
    wgts = op.outputs[2]
    r = op.attrs[1]

    grad_input = _lib.trilinear_devoxelize_grad(grad_out, inds, wgts, r)

    return None, grad_input
