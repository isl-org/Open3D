# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from ...python import ops
from ....paddle import classes
from .neighbor_search import FixedRadiusSearch, RadiusSearch
import paddle
from paddle import create_parameter
import numpy as np

__all__ = ['ContinuousConv']


class ContinuousConv(paddle.nn.Layer):
    r"""Continuous Convolution.

    This convolution supports continuous input and output point positions.
    This layer implements the convolution defined in

    *B. Ummenhofer and V. Koltun, Lagrangian Fluid Simulation with Continuous Convolutions, ICLR 2020.*

    The convolution at position :math:`\mathbf x` is defined as

    .. math::
        (f*g)(\mathbf x) = \frac{1}{\psi(\mathbf x)} \sum_{i \in \mathcal N(\mathbf x, R)} a(\mathbf x_i, \mathbf x)\; f_i\; g(\Lambda(\mathbf x_i - \mathbf x)).

    With :math:`f` as the input feature function and :math:`g` as the filter function.
    The input points are :math:`\mathbf x_i` and the input features are :math:`f_i`.
    The normalization :math:`\frac{1}{\psi(\mathbf x)}` can be turned on with the **normalize** parameter.
    The per neighbor value :math:`a(\mathbf x_i, \mathbf x)` can be used to implement window functions; see parameter **window_function**.
    The function :math:`\Lambda` for looking up filter values is defined by the parameters **coordinate_mapping** and **interpolation**.

    Example:
      This shows a minimal example of how to use the layer::

          import paddle
          import open3d.ml.paddle as ml3d

          inp_positions = paddle.randn([20, 3])
          inp_features = paddle.randn([20, 8])
          out_positions = paddle.randn([10, 3])

          conv = ml3d.layers.ContinuousConv(in_channels=8, filters=16, kernel_size=[3,3,3])
          out_features = conv(inp_features, inp_positions, out_positions, extents=2.0)


    Arguments:
        in_channels: The number of input channels.

        filters: The number of filters/output channels.

        kernel_size: The spatial resolution of the filter, e.g. [3,3,3].

        activation: The activation function to use. None means no activation.

        use_bias: If True adds an additive bias vector.

        kernel_initializer: Initializer for the kernel weights.

        bias_initializer: Initializer for the bias vector.

        align_corners: If true then the voxel centers of the outer voxels of the
          filter array are mapped to the boundary of the filter shape.
          If false then the boundary of the filter array is mapped to the
          boundary of the filter shape.

        coordinate_mapping: The mapping that is applied to the input coordinates.
          One of 'ball_to_cube_radial', 'ball_to_cube_volume_preserving',
          'identity'.

            * 'ball_to_cube_radial' uses radial stretching to map a sphere to
              a cube.
            * 'ball_to_cube_volume_preserving' is using a more expensive volume
              preserving mapping to map a sphere to a cube.
            * 'identity' no mapping is applied to the coordinates.

        interpolation: One of 'linear', 'linear_border', 'nearest_neighbor'.
            * 'linear' is trilinear interpolation with coordinate clamping.
            * 'linear_border' uses a zero border if outside the range.
            * 'nearest_neighbor' uses the nearest neighbor instead of interpolation.

        normalize: If true then the result is normalized either by the number of
          points (neighbors_importance is null) or by the sum of the respective
          values in neighbors_importance.

        radius_search_ignore_query_points: If true the points that coincide with the
          center of the search window will be ignored. This excludes the query point
          if 'queries' and 'points' are the same point cloud.

        radius_search_metric: Either L1, L2 or Linf. Default is L2

        offset: A single 3D vector used in the filter coordinate computation.
          The shape is [3].

        window_function: Optional radial window function to steer the importance of
          points based on their distance to the center. The input to the function
          is a 1D tensor of distances (squared distances if radius_search_metric is
          'L2'). The output must be a tensor of the same shape. Example::

            def window_fn(r_sqr):
                return paddle.clamp((1-r_sqr)**3, 0, 1)

        use_dense_layer_for_center: If True a linear dense layer is used to
          process the input features for each point. The result is added to the
          result of the convolution before adding the bias. This option is
          useful when using even kernel sizes that have no center element and
          input and output point sets are the same and
          'radius_search_ignore_query_points' has been set to True.

        dense_kernel_initializer: Initializer for the kernel weights of the
          linear layer used for the center if 'use_dense_layer_for_center'
          is True.
    """

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=paddle.nn.initializer.Uniform(-0.05, 0.05),
                 bias_initializer=paddle.nn.initializer.Constant(),
                 align_corners=True,
                 coordinate_mapping='ball_to_cube_radial',
                 interpolation='linear',
                 normalize=True,
                 radius_search_ignore_query_points=False,
                 radius_search_metric='L2',
                 offset=None,
                 window_function=None,
                 use_dense_layer_for_center=False,
                 dense_kernel_initializer=paddle.nn.initializer.XavierUniform(),
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.align_corners = align_corners
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.normalize = normalize
        self.radius_search_ignore_query_points = radius_search_ignore_query_points
        self.radius_search_metric = radius_search_metric
        self.dense_kernel_initializer = dense_kernel_initializer

        if offset is None:
            offset = paddle.zeros(shape=(3,), dtype=paddle.float32)
        self.register_buffer('offset', offset)

        self.window_function = window_function

        self.fixed_radius_search = FixedRadiusSearch(
            metric=self.radius_search_metric,
            ignore_query_point=self.radius_search_ignore_query_points,
            return_distances=not self.window_function is None)

        self.radius_search = RadiusSearch(
            metric=self.radius_search_metric,
            ignore_query_point=self.radius_search_ignore_query_points,
            return_distances=not self.window_function is None,
            normalize_distances=not self.window_function is None)

        self.use_dense_layer_for_center = use_dense_layer_for_center
        if self.use_dense_layer_for_center:
            self.dense = paddle.nn.Linear(self.in_channels,
                                          self.filters,
                                          bias=False)
            self.dense_kernel_initializer(self.dense.weight)

        kernel_shape = (*self.kernel_size, self.in_channels, self.filters)
        # self.kernel = paddle.nn.Parameter(data=paddle.Tensor(*kernel_shape),
        #                                   requires_grad=True)
        self.kernel = create_parameter(
            kernel_shape,
            dtype=paddle.float32,
            default_initializer=self.kernel_initializer)

        if self.use_bias:
            self.bias = create_parameter((self.filters,),
                                         dtype=paddle.float32,
                                         is_bias=True,
                                         default_initializer=bias_initializer)

    def forward(self,
                inp_features,
                inp_positions,
                out_positions,
                extents,
                inp_importance=None,
                fixed_radius_search_hash_table=None,
                user_neighbors_index=None,
                user_neighbors_row_splits=None,
                user_neighbors_importance=None):
        offset = self.offset
        if isinstance(extents, (float, int)):
            extents = paddle.to_tensor(extents, dtype=inp_positions.dtype)

        if inp_importance is None:
            inp_importance = paddle.empty(
                (0,), dtype=paddle.float32).to(self.kernel.place)

        return_distances = not self.window_function is None

        if not user_neighbors_index is None and not user_neighbors_row_splits is None:

            if user_neighbors_importance is None:
                neighbors_importance = paddle.empty(
                    (0,), dtype=paddle.float32).to(self.kernel.place)
            else:
                neighbors_importance = user_neighbors_importance

            neighbors_index = user_neighbors_index
            neighbors_row_splits = user_neighbors_row_splits

        else:
            if len(extents.shape) == 0:
                radius = 0.5 * extents
                self.nns = self.fixed_radius_search(
                    inp_positions,
                    queries=out_positions,
                    radius=radius,
                    hash_table=fixed_radius_search_hash_table)
                if return_distances:
                    if self.radius_search_metric == 'L2':
                        neighbors_distance_normalized = self.nns.neighbors_distance / (
                            radius * radius)
                    else:  # L1
                        neighbors_distance_normalized = self.nns.neighbors_distance / radius

            elif len(extents.shape) == 1:
                radii = 0.5 * extents
                self.nns = self.radius_search(inp_positions,
                                              queries=out_positions,
                                              radii=radii)

            else:
                raise ValueError("extents rank must be 0 or 1")

            if self.window_function is None:
                neighbors_importance = paddle.empty((0,), dtype=paddle.float32)
            else:
                neighbors_importance = self.window_function(
                    neighbors_distance_normalized)

            neighbors_index = self.nns.neighbors_index
            neighbors_row_splits = self.nns.neighbors_row_splits

        # for stats and debugging
        num_pairs = neighbors_index.shape[0]
        self._avg_neighbors = num_pairs / out_positions.shape[0]

        extents_rank2 = extents
        while len(extents_rank2.shape) < 2:
            extents_rank2 = paddle.unsqueeze(extents_rank2, axis=-1)

        self._conv_values = {
            'filters': self.kernel,
            'out_positions': out_positions,
            'extents': extents_rank2,
            'offset': offset,
            'inp_positions': inp_positions,
            'inp_features': inp_features,
            'inp_importance': inp_importance,
            'neighbors_index': neighbors_index,
            'neighbors_row_splits': neighbors_row_splits,
            'neighbors_importance': neighbors_importance,
            'align_corners': self.align_corners,
            'coordinate_mapping': self.coordinate_mapping,
            'interpolation': self.interpolation,
            'normalize': self.normalize,
            'max_temp_mem_mb': 64
        }

        out_features = ops.continuous_conv(**self._conv_values)

        self._conv_output = out_features

        if self.use_dense_layer_for_center:
            self._dense_output = self.dense(inp_features)
            out_features = out_features + self._dense_output

        if self.use_bias:
            out_features += self.bias
        if not self.activation is None:
            out_features = self.activation(out_features)

        return out_features
