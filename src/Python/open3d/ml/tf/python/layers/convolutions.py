from open3d.ml.tf import ops
import tensorflow as tf


class ContinuousConv(tf.keras.layers.Layer):
    """Continuous Convolution. This op supports continuous input and output point positions.

    This layer computes a continuous convolution on a point cloud at the 
    specified output points.

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
          - 'ball_to_cube_radial' uses radial stretching to map a sphere to
            a cube.
          - 'ball_to_cube_volume_preserving' is using a more expensive volume
            preserving mapping to map a sphere to a cube.
          - 'identity' no mapping is applied to the coordinates.

        interpolation: One of 'linear', 'linear_border', 'nearest_neighbor'.
          - 'linear' is trilinear interpolation with coordinate clamping.
          - 'linear_border' uses a zero border if outside the range.
          - 'nearest_neighbor' uses the neares neighbor instead of interpolation.

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
          is a 1D tensor of distances (squared distances if radius_search_metric is 'L2').
          The output must be a tensor of the same shape. Example:
          
            def window_fn(r_sqr):
                return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

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

        if not offset is None:
            self.offset = tf.zeros(shape=(3,))
        else:
            self.offset = offset

        self.window_function = window_function

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
             inp_importance=None):
        """This function computes the output features.

        Arguments:

          inp_features: A 2D tensor which stores a feature vector for each input
            point.

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

        Returns: A tensor of shape [num output points, filters] with the output
          features.
        """

        offset = self.offset

        result = []

        if inp_importance is None:
            inp_importance = tf.ones((0,), dtype=tf.float32)

        extents = tf.convert_to_tensor(extents)

        return_distances = not self.window_function is None
        if extents.shape.rank == 0:
            radius = 0.5 * extents
            hash_table_size_factor = 1 / 64
            self.nns = ops.fixed_radius_search(
                ignore_query_point=self.radius_search_ignore_query_points,
                return_distances=return_distances,
                metric=self.radius_search_metric,
                points=inp_positions,
                queries=out_positions,
                radius=radius,
                hash_table_size_factor=hash_table_size_factor)
            if return_distances:
                if self.radius_search_metric == 'L2':
                    neighbors_distances_normalized = self.nns.neighbors_distances / (
                        radius * radius)
                else:  # L1
                    neighbors_distances_normalized = self.nns.neighbors_distances / radius

        elif extents.shape.rank == 1:
            radii = 0.5 * extents
            self.nns = ops.radius_search(
                ignore_query_point=self.radius_search_ignore_query_points,
                return_distances=return_distances,
                normalize_distances=return_distances,
                metric=self.radius_search_metric,
                points=inp_positions,
                queries=out_positions,
                radii=radii)

        else:
            raise Exception("extents rank must be 0 or 1")

        if self.window_function is None:
            neighbors_importance = tf.ones((0,), dtype=tf.float32)
        else:
            neighbors_importance = self.window_function(
                neighbors_distances_normalized)

        # for stats and debugging
        num_pairs = tf.shape(self.nns.neighbors_index)[0]
        self._avg_neighbors = num_pairs / tf.shape(out_positions)[0]

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
            'neighbors_index': self.nns.neighbors_index,
            'neighbors_importance': neighbors_importance,
            'neighbors_prefix_sum': self.nns.neighbors_count_prefix_sum,
            'align_corners': self.align_corners,
            'coordinate_mapping': self.coordinate_mapping,
            'interpolation': self.interpolation,
            'normalize': self.normalize,
        }

        out_features = ops.continuous_conv(**self._conv_values)

        self._conv_values['out_features'] = out_features

        self._conv_output = out_features

        if self.use_bias:
            out_features += self.bias
        out_features = self.activation(out_features)

        return out_features

    def compute_output_shape(self, inp_features_shape):
        return tf.TensorShape((None, self.filters))
