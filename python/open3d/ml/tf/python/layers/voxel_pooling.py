# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from ...python.ops import ops
import tensorflow as tf

__all__ = ['VoxelPooling']


class VoxelPooling(tf.keras.layers.Layer):
    """Voxel pooling for 3D point clouds.

    Spatial pooling for point clouds by combining points that fall into the same voxel bin.

    The voxel grid used for pooling is always aligned to the origin (0,0,0) to
    simplify building voxel grid hierarchies. The order of the returned voxels is
    not defined as can be seen in the following example::

      import open3d.ml.tf as ml3d

      positions = [
          [0.1,0.1,0.1],
          [0.5,0.5,0.5],
          [1.7,1.7,1.7],
          [1.8,1.8,1.8],
          [0.3,2.4,1.4]]

      features = [[1.0,2.0],
                  [1.1,2.3],
                  [4.2,0.1],
                  [1.3,3.4],
                  [2.3,1.9]]

      voxel_pooling = ml3d.layers.VoxelPooling(position_fn='center', feature_fn='max')

      ans = voxel_pooling(positions, features, 1.0)

      # returns the voxel centers in
      # ans.pooled_positions = [[0.5, 2.5, 1.5],
      #                         [1.5, 1.5, 1.5],
      #                         [0.5, 0.5, 0.5]]
      #
      # and the max pooled features for each voxel in
      # ans.pooled_features = [[2.3, 1.9],
      #                        [4.2, 3.4],
      #                        [1.1, 2.3]]

    Arguments:
      position_fn: Defines how the new point positions will be computed.
        The options are
          * "average" computes the center of gravity for the points within one voxel.
          * "nearest_neighbor" selects the point closest to the voxel center.
          * "center" uses the voxel center for the position of the generated point.

      feature_fn: Defines how the pooled features will be computed.
        The options are
          * "average" computes the average feature vector.
          * "nearest_neighbor" selects the feature vector of the point closest to the voxel center.
          * "max" uses the maximum feature among all points within the voxel.
    """

    def __init__(self, position_fn='center', feature_fn='max', **kwargs):
        self.position_fn = position_fn
        self.feature_fn = feature_fn
        super().__init__(autocast=False, **kwargs)

    def build(self, inp_shape):
        super().build(inp_shape)

    def call(self, positions, features, voxel_size):
        """This function computes the pooled positions and features.

        Arguments:

          positions: The point positions with shape [N,3] with N as the number of points.
            *This argument must be given as a positional argument!*

          features: The feature vector with shape [N,channels].

          voxel_size: The voxel size.

        Returns:
          2 Tensors in the following order

          pooled_positions
            The output point positions with shape [M,3] and M <= N.

          pooled_features:
            The output point features with shape [M,channels] and M <= N.
        """
        result = ops.voxel_pooling(positions,
                                   features,
                                   voxel_size,
                                   position_fn=self.position_fn,
                                   feature_fn=self.feature_fn)
        return result
