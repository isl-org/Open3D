from open3d.ml.tf import ops
import tensorflow as tf


class FixedRadiusSearch(tf.keras.layers.Layer):
    """Fixed radius search for 3D point clouds.

    This layer computes the neighbors for a fixed radius on a point cloud. 

    Arguments:

    metric:
      Either L1, L2 or Linf. Default is L2

    ignore_query_point:
      If true the points that coincide with the center of the search window will be
      ignored. This excludes the query point if 'queries' and 'points' are the same 
      point cloud.

    return_distances:
      If True the distances for each neighbor will be returned.
      If False a zero length Tensor will be returned instead
    """

    def __init__(self,
                 metric='L2',
                 ignore_query_point=False,
                 return_distances=False,
                 max_hash_table_size=32 * 2**20,
                 **kwargs):
        self.metric = metric
        self.ignore_query_point = ignore_query_point
        self.return_distances = return_distances
        self.max_hash_table_size = max_hash_table_size
        super().__init__(autocast=False, **kwargs)

    def build(self, inp_shape):
        super().build(inp_shape)

    def call(self,
             points,
             queries,
             radius,
             hash_table_size_factor=1 / 64,
             hash_table=None):
        """This function the neighbors within a radius for each query point.

        Arguments:

          points: The 3D positions of the input points.

          queries: The 3D positions of the query points.

          radius: A scalar with the neighborhood radius

          hash_table_size_factor: Scalar. The size of the hash table as fraction of points.

          hash_table: A precomputed hash table generated with build_spatial_hash_table().
            This input can be used to explicitly force the reuse of a hash table in special
            cases and is usually not needed.
            Note that the hash table must have been generated with the same 'points' array.

        Returns: 3 Tensors in the following order
          neighbors_index
            The compact list of indices of the neighbors. The corresponding query point 
            can be inferred from the 'neighbor_count_row_splits' vector.

          neighbors_row_splits
            The exclusive prefix sum of the neighbor count for the query points including
            the total neighbor count as the last element. The size of this array is the 
            number of queries + 1.

          neighbors_distance
            Stores the distance to each neighbor if 'return_distances' is True.
            Note that the distances are squared if metric is L2.
            This is a zero length Tensor if 'return_distances' is False. 

        """
        if hash_table is None:
            table = ops.build_spatial_hash_table(
                max_hash_table_size=self.max_hash_table_size,
                points=points,
                radius=radius,
                hash_table_size_factor=hash_table_size_factor)
        else:
            table = hash_table
        result = ops.fixed_radius_search(
            ignore_query_point=self.ignore_query_point,
            return_distances=self.return_distances,
            metric=self.metric,
            points=points,
            queries=queries,
            radius=radius,
            hash_table_index=table.hash_table_index,
            hash_table_row_splits=table.hash_table_row_splits)
        return result
