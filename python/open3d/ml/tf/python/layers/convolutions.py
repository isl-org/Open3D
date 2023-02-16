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

from ...python.ops import ops
from .neighbor_search import FixedRadiusSearch, RadiusSearch
import tensorflow as tf
import numpy as np

__all__ = ['ContinuousConv', 'SparseConv', 'SparseConvTranspose']


class ContinuousConv(tf.keras.layers.Layer):
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

          import tensorflow as tf
          import open3d.ml.tf as ml3d

          inp_positions = tf.random.normal([20,3])
          inp_features = tf.random.normal([20,8])
          out_positions = tf.random.normal([10,3])

          conv = ml3d.layers.ContinuousConv(filters=16, kernel_size=[3,3,3])
          out_features = conv(inp_features, inp_positions, out_positions, extents=2.0)


    Arguments:
        filters: The number of filters/output channels.

        kernel_size: The spatial resolution of the filter, e.g. [3,3,3].

        activation: The activation function to use. None means no activation.

        use_bias: If True adds an additive bias vector.

        kernel_initializer: Initializer for the kernel weights.

        bias_initializer: Initializer for the bias vector.

        kernel_regularizer: Regularizer for the kernel weights.

        bias_regularizer: Regularizer for the bias vector.

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

        interpolation: One of 'linear', 'linear_border',
          'nearest_neighbor'.
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
                return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

        use_dense_layer_for_center: If True a linear dense layer is used to
          process the input features for each point. The result is added to the
          result of the convolution before adding the bias. This option is
          useful when using even kernel sizes that have no center element and
          input and output point sets are the same and
          'radius_search_ignore_query_points' has been set to True.

        dense_kernel_initializer: Initializer for the kernel weights of the
          linear layer used for the center if 'use_dense_layer_for_center'
          is True.

        dense_kernel_regularizer: Regularizer for the kernel weights of the
          linear layer used for the center if 'use_dense_layer_for_center'
          is True.

        in_channels: This keyword argument is for compatibility with PyTorch.
          It is not used and in_channels will be inferred at the first execution
          of the layer.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 align_corners=True,
                 coordinate_mapping='ball_to_cube_radial',
                 interpolation='linear',
                 normalize=True,
                 radius_search_ignore_query_points=False,
                 radius_search_metric='L2',
                 offset=None,
                 window_function=None,
                 use_dense_layer_for_center=False,
                 dense_kernel_initializer='glorot_uniform',
                 dense_kernel_regularizer=None,
                 in_channels=None,
                 **kwargs):

        from tensorflow.keras import activations, initializers, regularizers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.align_corners = align_corners
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.normalize = normalize
        self.radius_search_ignore_query_points = radius_search_ignore_query_points
        self.radius_search_metric = radius_search_metric
        self.dense_kernel_initializer = initializers.get(
            dense_kernel_initializer)
        self.dense_kernel_regularizer = regularizers.get(
            dense_kernel_regularizer)

        if offset is None:
            self.offset = tf.zeros(shape=(3,))
        else:
            self.offset = offset

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
            self.dense = tf.keras.layers.Dense(
                self.filters,
                kernel_initializer=dense_kernel_initializer,
                kernel_regularizer=dense_kernel_regularizer,
                use_bias=False)

        super().__init__(**kwargs)

    def build(self, inp_features_shape):
        self.in_channels = inp_features_shape[-1]

        kernel_shape = tf.TensorShape(
            (*self.kernel_size, self.in_channels, self.filters))
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=self.trainable,
        )

        if self.use_bias:
            bias_shape = tf.TensorShape((self.filters,))
            self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=self.trainable,
            )
        super().build(inp_features_shape)

    def call(self,
             inp_features,
             inp_positions,
             out_positions,
             extents,
             inp_importance=None,
             fixed_radius_search_hash_table=None,
             user_neighbors_index=None,
             user_neighbors_row_splits=None,
             user_neighbors_importance=None):
        """This function computes the output features.

        Arguments:
          inp_features: A 2D tensor which stores a feature vector for each input
            point. *This argument must be given as a positional argument!*

          inp_positions: A 2D tensor with the 3D point positions of each input
            point. The coordinates for each point is a vector with format [x,y,z].

          out_positions: A 2D tensor with the 3D point positions of each output
            point. The coordinates for each point is a vector with format [x,y,z].

          extents: The extent defines the spatial size of the filter for each
            output point.
            For 'ball to cube' coordinate mappings the extent defines the
            bounding box of the ball.
            The shape of the tensor is either [1] or [num output points].

          inp_importance: Optional scalar importance value for each input point.

          fixed_radius_search_hash_table: A precomputed hash table generated with
            build_spatial_hash_table().
            This input can be used to explicitly force the reuse of a hash table in
            special cases and is usually not needed.
            Note that the hash table must have been generated with the same 'points'
            array. Note that this parameter is only used if 'extents' is a scalar.

          user_neighbors_index: This parameter together with 'user_neighbors_row_splits'
            and 'user_neighbors_importance' allows to override the automatic neighbor
            search. This is the list of neighbor indices for each output point.
            This is a nested list for which the start and end of each sublist is
            defined by 'user_neighbors_row_splits'.

          user_neighbors_row_splits: Defines the start and end of each neighbors
            list in 'user_neighbors_index'.

          user_neighbors_importance: Defines a scalar importance value for each
            element in 'user_neighbors_index'.


        Returns:
          A tensor of shape [num output points, filters] with the output features.
        """
        offset = self.offset

        if inp_importance is None:
            inp_importance = tf.ones((0,), dtype=tf.float32)

        extents = tf.convert_to_tensor(extents)

        return_distances = not self.window_function is None

        if not user_neighbors_index is None and not user_neighbors_row_splits is None:

            if user_neighbors_importance is None:
                neighbors_importance = tf.ones((0,), dtype=tf.float32)
            else:
                neighbors_importance = user_neighbors_importance

            neighbors_index = user_neighbors_index
            neighbors_row_splits = user_neighbors_row_splits

        else:
            if extents.shape.rank == 0:
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

            elif extents.shape.rank == 1:
                radii = 0.5 * extents
                self.nns = self.radius_search(inp_positions,
                                              queries=out_positions,
                                              radii=radii)

            else:
                raise Exception("extents rank must be 0 or 1")

            if self.window_function is None:
                neighbors_importance = tf.ones((0,), dtype=tf.float32)
            else:
                neighbors_importance = self.window_function(
                    neighbors_distance_normalized)

            neighbors_index = self.nns.neighbors_index
            neighbors_row_splits = self.nns.neighbors_row_splits

        # for stats and debugging
        num_pairs = tf.shape(neighbors_index)[0]
        self._avg_neighbors = tf.dtypes.cast(
            num_pairs, tf.float32) / tf.dtypes.cast(
                tf.shape(out_positions)[0], tf.float32)

        extents_rank2 = extents
        while extents_rank2.shape.rank < 2:
            extents_rank2 = tf.expand_dims(extents_rank2, axis=-1)

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
        }

        out_features = ops.continuous_conv(**self._conv_values)

        self._conv_output = out_features

        if self.use_dense_layer_for_center:
            self._dense_output = self.dense(inp_features)
            out_features = out_features + self._dense_output

        if self.use_bias:
            out_features += self.bias
        out_features = self.activation(out_features)

        return out_features

    def compute_output_shape(self, inp_features_shape):
        return tf.TensorShape((None, self.filters))


class SparseConv(tf.keras.layers.Layer):
    """Sparse Convolution.

    This layer computes a convolution which is only evaluated at the specified output positions.
    The layer assumes that input and output points lie on a regular grid.


    Example:
      This shows a minimal example of how to use the layer::

        import tensorflow as tf
        import open3d.ml.tf as ml3d

        # +0.5 to move the points to the voxel center
        inp_positions = tf.cast(tf.random.uniform([20,3], 0, 10, dtype=tf.int32), tf.float32)+0.5
        inp_features = tf.random.normal([20,8])
        out_positions = tf.cast(tf.random.uniform([20,3], 0, 10, dtype=tf.int32), tf.float32)+0.5

        conv = ml3d.layers.SparseConv(filters=16, kernel_size=[3,3,3])
        out_features = conv(inp_features, inp_positions, out_positions, voxel_size=1.0)


    Arguments:
        filters: The number of filters/output channels.

        kernel_size: The spatial resolution of the filter, e.g. [3,3,3].

        activation: The activation function to use. None means no activation.

        use_bias: If True adds an additive bias vector.

        kernel_initializer: Initializer for the kernel weights.

        bias_initializer: Initializer for the bias vector.

        kernel_regularizer: Regularizer for the kernel weights.

        bias_regularizer: Regularizer for the bias vector.

        normalize: If true then the result is normalized by the number of input points.

        offset: A single 3D vector used in the filter coordinate computation.
          The shape is [3]. This can be used to control how the filters are
          centered. It will be set automatically for kernels with even sizes.

        in_channels: This keyword argument is for compatibility with PyTorch.
          It is not used and in_channels will be inferred at the first execution
          of the layer.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 normalize=False,
                 offset=None,
                 in_channels=None,
                 **kwargs):

        from tensorflow.keras import activations, initializers, regularizers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.normalize = normalize

        if not (np.asarray(kernel_size) == kernel_size[0]).all():
            raise Exception("Only cubic kernel sizes are supported.")

        if offset is None:
            if kernel_size[0] % 2:
                self.offset = tf.zeros(shape=(3,))
            else:
                self.offset = tf.fill([3], -0.5)
        else:
            self.offset = offset

        self.fixed_radius_search = FixedRadiusSearch(metric='Linf',
                                                     ignore_query_point=False,
                                                     return_distances=False)

        super().__init__(**kwargs)

    def build(self, inp_features_shape):
        self.in_channels = inp_features_shape[-1]

        kernel_shape = tf.TensorShape(
            (*self.kernel_size, self.in_channels, self.filters))
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=self.trainable,
        )

        if self.use_bias:
            bias_shape = tf.TensorShape((self.filters,))
            self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=self.trainable,
            )
        super().build(inp_features_shape)

    def call(self,
             inp_features,
             inp_positions,
             out_positions,
             voxel_size,
             inp_importance=None,
             fixed_radius_search_hash_table=None):
        """This function computes the output features.

        Arguments:
          inp_features: A 2D tensor which stores a feature vector for each input
            point. *This argument must be given as a positional argument!*

          inp_positions: A 2D tensor with the 3D point positions of each input
            point. The coordinates for each point is a vector with format [x,y,z].

          out_positions: A 2D tensor with the 3D point positions of each output
            point. The coordinates for each point is a vector with format [x,y,z].

          voxel_size: A scalar float that defines the edge length of a voxel.

          inp_importance: Optional scalar importance value for each input point.

          fixed_radius_search_hash_table: A precomputed hash table generated with
            build_spatial_hash_table(). This input can be used to explicitly force the
            reuse of a hash table in special cases and is usually not needed.
            Note that the hash table must have been generated with the same 'points'
            array. Note that this parameter is only used if 'extents' is a scalar.

        Returns: A tensor of shape [num output points, filters] with the output
          features.
        """
        offset = self.offset
        voxel_size = tf.convert_to_tensor(voxel_size, dtype=inp_positions.dtype)
        if voxel_size.shape.rank != 0:
            raise Exception("voxel_size must be a scalar")

        if inp_importance is None:
            inp_importance = tf.ones((0,), dtype=tf.float32)

        if isinstance(inp_features, tf.RaggedTensor):
            if not (isinstance(inp_positions, tf.RaggedTensor) and
                    isinstance(out_positions, tf.RaggedTensor)):
                raise Exception(
                    "All of inp_positions, inp_features and out_positions must be tf.Tensor, or tf.RaggedTensor"
                )

        hash_table_size_factor = 1 / 64
        self.nns = self.fixed_radius_search(
            inp_positions,
            queries=out_positions - offset * voxel_size,
            radius=self.kernel_size[0] * voxel_size * 0.51,
            hash_table_size_factor=hash_table_size_factor,
            hash_table=fixed_radius_search_hash_table)

        out_positions_split = None
        if isinstance(inp_positions, tf.RaggedTensor):
            inp_positions = inp_positions.values
            inp_features = inp_features.values
            out_positions_split = out_positions.row_splits
            out_positions = out_positions.values

        # for stats and debugging
        num_pairs = tf.shape(self.nns.neighbors_index)[0]
        self._avg_neighbors = num_pairs / tf.shape(out_positions)[0]

        extents_rank2 = tf.fill([1, 1], voxel_size * self.kernel_size[0])

        self._conv_values = {
            'filters': self.kernel,
            'out_positions': out_positions,
            'extents': extents_rank2,
            'offset': offset,
            'inp_positions': inp_positions,
            'inp_features': inp_features,
            'inp_importance': inp_importance,
            'neighbors_index': self.nns.neighbors_index,
            'neighbors_importance': tf.ones((0,), dtype=tf.float32),
            'neighbors_row_splits': self.nns.neighbors_row_splits,
            'align_corners': False,
            'coordinate_mapping': 'identity',
            'interpolation': 'nearest_neighbor',
            'normalize': self.normalize,
        }

        out_features = ops.continuous_conv(**self._conv_values)

        self._conv_output = out_features

        if self.use_bias:
            out_features += self.bias
        out_features = self.activation(out_features)

        if out_positions_split is not None:
            out_features = tf.RaggedTensor.from_row_splits(
                values=out_features, row_splits=out_positions_split)

        return out_features

    def compute_output_shape(self, inp_features_shape):
        return tf.TensorShape((None, self.filters))


class SparseConvTranspose(tf.keras.layers.Layer):
    """Sparse Transposed Convolution. This layer computes a transposed convolution which is only evaluated at the specified output positions.

    Example:
      This shows a minimal example of how to use the layer::

        import tensorflow as tf
        import open3d.ml.tf as ml3d

        # +0.5 to move the points to the voxel center
        inp_positions = tf.cast(tf.random.uniform([20,3], 0, 10, dtype=tf.int32), tf.float32)+0.5
        inp_features = tf.random.normal([20,8])
        out_positions = tf.cast(tf.random.uniform([20,3], 0, 10, dtype=tf.int32), tf.float32)+0.5

        conv = ml3d.layers.SparseConvTranspose(filters=16, kernel_size=[3,3,3])
        out_features = conv(inp_features, inp_positions, out_positions, voxel_size=1.0)


    Arguments:
        filters: The number of filters/output channels.

        kernel_size: The spatial resolution of the filter, e.g. [3,3,3].

        activation: The activation function to use. None means no activation.

        use_bias: If True adds an additive bias vector.

        kernel_initializer: Initializer for the kernel weights.

        bias_initializer: Initializer for the bias vector.

        kernel_regularizer: Regularizer for the kernel weights.

        bias_regularizer: Regularizer for the bias vector.

        normalize: If true then the input features will be normalized with the number of
          output points.

        offset: A single 3D vector used in the filter coordinate computation.
          The shape is [3]. This can be used to control how the filters are
          centered. It will be set automatically for kernels with even sizes.

        in_channels: This keyword argument is for compatibility with PyTorch.
          It is not used and in_channels will be inferred at the first execution
          of the layer.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 normalize=False,
                 offset=None,
                 in_channels=None,
                 **kwargs):

        from tensorflow.keras import activations, initializers, regularizers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.normalize = normalize

        if not (np.asarray(kernel_size) == kernel_size[0]).all():
            raise Exception("Only cubic kernel sizes are supported.")

        if offset is None:
            if kernel_size[0] % 2:
                self.offset = tf.zeros(shape=(3,))
            else:
                self.offset = tf.fill([3], -0.5)
        else:
            self.offset = offset

        self.fixed_radius_search = FixedRadiusSearch(metric='Linf',
                                                     ignore_query_point=False,
                                                     return_distances=False)

        super().__init__(**kwargs)

    def build(self, inp_features_shape):
        self.in_channels = inp_features_shape[-1]

        kernel_shape = tf.TensorShape(
            (*self.kernel_size, self.in_channels, self.filters))
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=self.trainable,
        )

        if self.use_bias:
            bias_shape = tf.TensorShape((self.filters,))
            self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=self.trainable,
            )
        super().build(inp_features_shape)

    def call(self,
             inp_features,
             inp_positions,
             out_positions,
             voxel_size,
             out_importance=None,
             fixed_radius_search_hash_table=None):
        """This function computes the output features.

        Arguments:
          inp_features: A 2D tensor which stores a feature vector for each input
            point. *This argument must be given as a positional argument!*

          inp_positions: A 2D tensor with the 3D point positions of each input
            point. The coordinates for each point is a vector with format [x,y,z].

          out_positions: A 2D tensor with the 3D point positions of each output
            point. The coordinates for each point is a vector with format [x,y,z].

          voxel_size: A scalar float that defines the edge length of a voxel.

          out_importance: Optional scalar importance value for each output point.

          fixed_radius_search_hash_table: A precomputed hash table generated with
            build_spatial_hash_table(). This input can be used to explicitly force the
            reuse of a hash table in special cases and is usually not needed.
            Note that the hash table must have been generated with the same 'points'
            array. Note that this parameter is only used if 'extents' is a scalar.

        Returns: A tensor of shape [num output points, filters] with the output
          features.
        """
        offset = self.offset
        voxel_size = tf.convert_to_tensor(voxel_size, dtype=inp_positions.dtype)
        if voxel_size.shape.rank != 0:
            raise Exception("voxel_size must be a scalar")

        if out_importance is None:
            out_importance = tf.ones((0,), dtype=tf.float32)

        empty_vec = tf.ones((0,), dtype=tf.float32)

        if isinstance(inp_features, tf.RaggedTensor):
            if not (isinstance(inp_positions, tf.RaggedTensor) and
                    isinstance(out_positions, tf.RaggedTensor)):
                raise Exception(
                    "All of inp_positions, inp_features and out_positions must be tf.Tensor, or tf.RaggedTensor"
                )

        hash_table_size_factor = 1 / 64
        self.nns_inp = self.fixed_radius_search(
            out_positions,
            queries=inp_positions - offset * voxel_size,
            radius=self.kernel_size[0] * voxel_size * 0.51,
            hash_table_size_factor=hash_table_size_factor,
            hash_table=fixed_radius_search_hash_table)

        out_positions_split = None
        if isinstance(inp_positions, tf.RaggedTensor):
            inp_positions = inp_positions.values
            inp_features = inp_features.values
            out_positions_split = out_positions.row_splits
            out_positions = out_positions.values

        num_out = tf.shape(out_positions, out_type=tf.int64)[0]

        neighbors_index, neighbors_row_splits, _ = ops.invert_neighbors_list(
            num_out, self.nns_inp.neighbors_index,
            self.nns_inp.neighbors_row_splits, empty_vec)

        # for stats and debugging
        num_pairs = tf.shape(neighbors_index)[0]
        self._avg_neighbors = num_pairs / tf.shape(out_positions)[0]

        extents_rank2 = tf.fill([1, 1], voxel_size * self.kernel_size[0])

        self._conv_values = {
            'filters': self.kernel,
            'out_positions': out_positions,
            'extents': extents_rank2,
            'offset': offset,
            'inp_positions': inp_positions,
            'inp_features': inp_features,
            'out_importance': out_importance,
            'inp_neighbors_index': self.nns_inp.neighbors_index,
            'inp_neighbors_importance_sum': empty_vec,
            'inp_neighbors_row_splits': self.nns_inp.neighbors_row_splits,
            'neighbors_index': neighbors_index,
            'neighbors_importance': empty_vec,
            'neighbors_row_splits': neighbors_row_splits,
            'align_corners': False,
            'coordinate_mapping': 'identity',
            'interpolation': 'nearest_neighbor',
            'normalize': self.normalize,
        }

        out_features = ops.continuous_conv_transpose(**self._conv_values)

        self._conv_output = out_features

        if self.use_bias:
            out_features += self.bias
        out_features = self.activation(out_features)

        if out_positions_split is not None:
            out_features = tf.RaggedTensor.from_row_splits(
                values=out_features, row_splits=out_positions_split)

        return out_features

    def compute_output_shape(self, inp_features_shape):
        return tf.TensorShape((None, self.filters))
