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

from ...python.ops import ops
import tensorflow as tf

__all__ = ['FixedRadiusSearch', 'RadiusSearch', 'KNNSearch']


class FixedRadiusSearch(tf.keras.layers.Layer):
    """Fixed radius search for 3D point clouds.

    This layer computes the neighbors for a fixed radius on a point cloud.

    Example:

      This example shows a neighbor search that returns the indices to the
      found neighbors and the distances.::

        import tensorflow as tf
        import open3d.ml.tf as ml3d

        points = tf.random.normal([20,3])
        queries = tf.random.normal([10,3])
        radius = 0.8

        nsearch = ml3d.layers.FixedRadiusSearch(return_distances=True)
        ans = nsearch(points, queries, radius)
        # returns a tuple of neighbors_index, neighbors_row_splits, and neighbors_distance


    Arguments:

      metric: Either L1, L2 or Linf. Default is L2.

      ignore_query_point: If True the points that coincide with the center of
        the search window will be ignored. This excludes the query point if
        'queries' and 'points' are the same point cloud.

      return_distances: If True the distances for each neighbor will be returned.
        If False a zero length Tensor will be returned instead.
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
             points_row_splits=None,
             queries_row_splits=None,
             hash_table_size_factor=1 / 64,
             hash_table=None):
        """This function computes the neighbors within a fixed radius for each query point.

        Arguments:

          points: The 3D positions of the input points. *This argument must be
            given as a positional argument!*

          queries: The 3D positions of the query points.

          radius: A scalar with the neighborhood radius

          points_row_splits: Optional 1D vector with the row splits information
            if points is batched. This vector is [0, num_points] if there is
            only 1 batch item.

          queries_row_splits: Optional 1D vector with the row splits information
            if queries is batched.  This vector is [0, num_queries] if there is
            only 1 batch item.

          hash_table_size_factor: Scalar. The size of the hash table as fraction
            of points.

          hash_table: A precomputed hash table generated with build_spatial_hash_table().
            This input can be used to explicitly force the reuse of a hash table in special
            cases and is usually not needed.
            Note that the hash table must have been generated with the same 'points' array.

        Returns:
          3 Tensors in the following order

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
        if points_row_splits is None:
            points_row_splits = tf.cast(tf.stack([0, tf.shape(points)[0]]),
                                        dtype=tf.int64)
        if queries_row_splits is None:
            queries_row_splits = tf.cast(tf.stack([0, tf.shape(queries)[0]]),
                                         dtype=tf.int64)
        if hash_table is None:
            table = ops.build_spatial_hash_table(
                max_hash_table_size=self.max_hash_table_size,
                points=points,
                radius=radius,
                points_row_splits=points_row_splits,
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
            points_row_splits=points_row_splits,
            queries_row_splits=queries_row_splits,
            hash_table_splits=table.hash_table_splits,
            hash_table_index=table.hash_table_index,
            hash_table_cell_splits=table.hash_table_cell_splits)
        return result


class RadiusSearch(tf.keras.layers.Layer):
    """Radius search for 3D point clouds.

    This layer computes the neighbors for each query point with each query
    having an individual radius.

    Example:

      This example shows a neighbor search that returns the indices to the
      found neighbors and the distances.::

        import tensorflow as tf
        import open3d.ml.tf as ml3d

        points = tf.random.normal([20,3])
        queries = tf.random.normal([10,3])
        radii = tf.random.normal([10], mean=1.0)

        nsearch = ml3d.layers.RadiusSearch(return_distances=True)
        ans = nsearch(points, queries, radii)
        # returns a tuple of neighbors_index, neighbors_row_splits, and neighbors_distance


    Arguments:

      metric: Either L1, L2 or Linf. Default is L2.

      ignore_query_point: If True the points that coincide with the center of the
        search window will be ignored. This excludes the query point if 'queries'
        and 'points' are the same point cloud.

      return_distances: If True the distances for each neighbor will be returned.
        If False a zero length Tensor will be returned instead.

      normalize_distances: If True the returned distances will be normalized with
        the radii.
    """

    def __init__(self,
                 metric='L2',
                 ignore_query_point=False,
                 return_distances=False,
                 normalize_distances=False,
                 **kwargs):
        self.metric = metric
        self.ignore_query_point = ignore_query_point
        self.return_distances = return_distances
        self.normalize_distances = normalize_distances
        super().__init__(autocast=False, **kwargs)

    def build(self, inp_shape):
        super().build(inp_shape)

    def call(self,
             points,
             queries,
             radii,
             points_row_splits=None,
             queries_row_splits=None):
        """This function computes the neighbors within a radius for each query point.

        Arguments:

          points: The 3D positions of the input points. *This argument must be
            given as a positional argument!*

          queries: The 3D positions of the query points.

          radii: A radius for each query point.

          points_row_splits: Optional 1D vector with the row splits information
            if points is batched.  This vector is [0, num_points] if there is
            only 1 batch item.

          queries_row_splits: Optional 1D vector with the row splits information
            if queries is batched. This vector is [0, num_queries] if there is
            only 1 batch item.

        Returns:
          3 Tensors in the following order

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
        if points_row_splits is None:
            points_row_splits = tf.cast(tf.stack([0, tf.shape(points)[0]]),
                                        dtype=tf.int64)
        if queries_row_splits is None:
            queries_row_splits = tf.cast(tf.stack([0, tf.shape(queries)[0]]),
                                         dtype=tf.int64)

        result = ops.radius_search(ignore_query_point=self.ignore_query_point,
                                   return_distances=self.return_distances,
                                   normalize_distances=self.normalize_distances,
                                   metric=self.metric,
                                   points=points,
                                   queries=queries,
                                   radii=radii,
                                   points_row_splits=points_row_splits,
                                   queries_row_splits=queries_row_splits)
        return result


class KNNSearch(tf.keras.layers.Layer):
    """KNN search for 3D point clouds.

    This layer computes the k nearest neighbors for each query point.

    Example:

      This example shows a neighbor search that returns the indices to the
      found neighbors and the distances.::

        import tensorflow as tf
        import open3d.ml.tf as ml3d

        points = tf.random.normal([20,3])
        queries = tf.random.normal([10,3])
        k = 8

        nsearch = ml3d.layers.KNNSearch(return_distances=True)
        ans = nsearch(points, queries, k)
        # returns a tuple of neighbors_index, neighbors_row_splits, and neighbors_distance
        # Since there are more than k points and we do not ignore any points we can
        # reshape the output to [num_queries, k] with
        neighbors_index = tf.reshape(ans.neighbors_index, [10,k])
        neighbors_distance = tf.reshape(ans.neighbors_distance, [10,k])


    Arguments:

      metric: Either L1, L2 or Linf. Default is L2.

      ignore_query_point: If True the points that coincide with the center of the
        search window will be ignored. This excludes the query point if 'queries'
        and 'points' are the same point cloud.

      return_distances: If True the distances for each neighbor will be returned.
        If False a zero length Tensor will be returned instead.
    """

    def __init__(self,
                 metric='L2',
                 ignore_query_point=False,
                 return_distances=False,
                 **kwargs):
        self.metric = metric
        self.ignore_query_point = ignore_query_point
        self.return_distances = return_distances
        super().__init__(autocast=False, **kwargs)

    def build(self, inp_shape):
        super().build(inp_shape)

    def call(self,
             points,
             queries,
             k,
             points_row_splits=None,
             queries_row_splits=None):
        """This function computes the k nearest neighbors for each query point.

        Arguments:

          points: The 3D positions of the input points. *This argument must be
            given as a positional argument!*

          queries: The 3D positions of the query points.

          k: The number of nearest neighbors to search.

          points_row_splits: Optional 1D vector with the row splits information
            if points is batched.
            This vector is [0, num_points] if there is only 1 batch item.

          queries_row_splits: Optional 1D vector with the row splits information
            if queries is batched.
            This vector is [0, num_queries] if there is only 1 batch item.

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
        if points_row_splits is None:
            points_row_splits = tf.cast(tf.stack([0, tf.shape(points)[0]]),
                                        dtype=tf.int64)
        if queries_row_splits is None:
            queries_row_splits = tf.cast(tf.stack([0, tf.shape(queries)[0]]),
                                         dtype=tf.int64)

        result = ops.knn_search(ignore_query_point=self.ignore_query_point,
                                return_distances=self.return_distances,
                                metric=self.metric,
                                points=points,
                                queries=queries,
                                k=k,
                                points_row_splits=points_row_splits,
                                queries_row_splits=queries_row_splits)
        return result
