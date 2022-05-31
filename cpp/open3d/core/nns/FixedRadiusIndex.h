// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NNSIndex.h"
#include "open3d/core/nns/NeighborSearchCommon.h"

namespace open3d {
namespace core {
namespace nns {

/// Builds a spatial hash table for a fixed radius search of 3D points.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \param points    The tensor of 3D points. This tensor may be splitted into
///        multiple batch items by defining points_row_splits_size accordingly.
///
/// \param radius    The radius that will be used for searching.
///
/// \param points_row_splits    Defines the start and end of the points in
///        each batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_points]
///
/// \param hash_table_splits    Tensor defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_index    This is an output tensor storing the values of
/// the
///        hash table, which are the indices to the points. The size of the
///        tensor must be equal to the number of points.
///
/// \param hash_table_cell_splits    This is an output tensor storing the start
///        of each hash table entry. The size of this array defines the size of
///        the hash table.
///        The hash table size is hash_table_cell_splits_size - 1.
///
template <class T>
void BuildSpatialHashTableCPU(const Tensor& points,
                              double radius,
                              const Tensor& points_row_splits,
                              const Tensor& hash_table_splits,
                              Tensor& hash_table_index,
                              Tensor& hash_table_cell_splits);

/// Fixed radius search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and an exclusive prefix
/// sum defines the start and end of list in the array.
/// In addition the function optionally can return the distances for each
/// neighbor in the same format as the indices to the neighbors.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \param points    Tensor with the 3D point positions. This must be the tensor
///        that was used for building the spatial hash table.
///
/// \param queries    Tensor with the 3D query positions. This may be the same
///                   tensor as \p points.
///
/// \param radius    The search radius.
///
/// \param points_row_splits    Defines the start and end of the points in each
///        batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_points]
///
/// \param queries_row_splits    Defines the start and end of the queries in
///        each batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_queries]
///
/// \param hash_table_splits    Tensor defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_index    This is an output of the function
///        BuildSpatialHashTableCPU. This is tensor storing the values of the
///        hash table, which are the indices to the points. The size of the
///        tensor must be equal to the number of points.
///
/// \param hash_table_cell_splits    This is an output of the function
///        BuildSpatialHashTableCPU. The row splits array describing the start
///        and end of each cell.
///
/// \param metric    One of L1, L2, Linf. Defines the distance metric for the
///        search.
///
/// \param ignore_query_point    If true then points with the same position as
///        the query point will be ignored.
///
/// \param return_distances    If true then this function will return the
///        distances for each neighbor to its query point in the same format
///        as the indices.
///        Note that for the L2 metric the squared distances will be returned!!
///
/// \param sort                If true then sort the results in ascending order
///        of distance
///
/// \param neighbors_index     The output tensor that saves the resulting
///        neighbor indices
///
/// \param neighbors_row_splits  Tensor defining the start and end the neighbor
///        indices in each batch item. The size of the tensor is
///        num_query_points + 1
///
/// \param neighbors_distance   The output tensor that saves the resulting
///        neighbor distances.
///
template <class T, class TIndex>
void FixedRadiusSearchCPU(const Tensor& points,
                          const Tensor& queries,
                          double radius,
                          const Tensor& points_row_splits,
                          const Tensor& queries_row_splits,
                          const Tensor& hash_table_splits,
                          const Tensor& hash_table_index,
                          const Tensor& hash_table_cell_splits,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          const bool sort,
                          Tensor& neighbors_index,
                          Tensor& neighbors_row_splits,
                          Tensor& neighbors_distance);

/// Hybrid search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and if there is less
/// neighbors than requested, the output tensor will be assigned with default
/// values, -1 for indices and 0 for distances. In addition the function
/// returns the number of neighbors for each query.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \param points    Tensor with the 3D point positions. This must be the tensor
///        that was used for building the spatial hash table.
///
/// \param queries    Tensor with the 3D query positions. This may be the same
///                   tensor as \p points.
///
/// \param radius    The search radius.
///
/// \param max_knn   The maximum number of neighbor for each query
///
/// \param points_row_splits    Defines the start and end of the points in each
///        batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_points]
///
/// \param queries_row_splits    Defines the start and end of the queries in
///        each batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_queries]
///
/// \param hash_table_splits    Tensor defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_index    This is an output of the function
///        BuildSpatialHashTableCPU. This is tensor storing the values of the
///        hash table, which are the indices to the points. The size of the
///        tensor must be equal to the number of points.
///
/// \param hash_table_cell_splits    This is an output of the function
///        BuildSpatialHashTableCPU. The row splits array describing the start
///        and end of each cell.
///
/// \param metric    One of L1, L2, Linf. Defines the distance metric for the
///        search.
///
/// \param neighbors_index     The output tensor that saves the resulting
///        neighbor indices
///
/// \param neighbors_count     The output tensor that saves the number of
///        neighbors for each query points
///
/// \param neighbors_distance   The output tensor that saves the resulting
///        neighbor distances.
///
template <class T, class TIndex>
void HybridSearchCPU(const Tensor& points,
                     const Tensor& queries,
                     double radius,
                     int max_knn,
                     const Tensor& points_row_splits,
                     const Tensor& queries_row_splits,
                     const Tensor& hash_table_splits,
                     const Tensor& hash_table_index,
                     const Tensor& hash_table_cell_splits,
                     const Metric metric,
                     Tensor& neighbors_index,
                     Tensor& neighbors_count,
                     Tensor& neighbors_distance);

#ifdef BUILD_CUDA_MODULE
/// Builds a spatial hash table for a fixed radius search of 3D points.
///
/// \param points    The tensor of 3D points. This tensor may be splitted into
///        multiple batch items by defining points_row_splits_size accordingly.
///
/// \param radius    The radius that will be used for searching.
///
/// \param points_row_splits    Defines the start and end of the points in
///        each batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_points]
///
/// \param hash_table_splits    Tensor defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_index    This is an output tensor storing the values of
/// the
///        hash table, which are the indices to the points. The size of the
///        tensor must be equal to the number of points.
///
/// \param hash_table_cell_splits    This is an output tensor storing the start
///        of each hash table entry. The size of this array defines the size of
///        the hash table.
///        The hash table size is hash_table_cell_splits_size - 1.
///
template <class T>
void BuildSpatialHashTableCUDA(const Tensor& points,
                               double radius,
                               const Tensor& points_row_splits,
                               const Tensor& hash_table_splits,
                               Tensor& hash_table_index,
                               Tensor& hash_table_cell_splits);

// Fixed radius search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and an exclusive prefix
/// sum defines the start and end of list in the array.
/// In addition the function optionally can return the distances for each
/// neighbor in the same format as the indices to the neighbors.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \param points    Tensor with the 3D point positions. This must be the tensor
///        that was used for building the spatial hash table.
///
/// \param queries    Tensor with the 3D query positions. This may be the same
///                   tensor as \p points.
///
/// \param radius    The search radius.
///
/// \param points_row_splits    Defines the start and end of the points in each
///        batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_points]
///
/// \param queries_row_splits    Defines the start and end of the queries in
/// each
///        batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_queries]
///
/// \param hash_table_splits    Tensor defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_index    This is an output of the function
///        BuildSpatialHashTableCPU. This is tensor storing the values of the
///        hash table, which are the indices to the points. The size of the
///        tensor must be equal to the number of points.
///
/// \param hash_table_cell_splits    This is an output of the function
///        BuildSpatialHashTableCPU. The row splits array describing the start
///        and end of each cell.
///
/// \param metric    One of L1, L2, Linf. Defines the distance metric for the
///        search.
///
/// \param ignore_query_point    If true then points with the same position as
///        the query point will be ignored.
///
/// \param return_distances    If true then this function will return the
///        distances for each neighbor to its query point in the same format
///        as the indices.
///        Note that for the L2 metric the squared distances will be returned!!
///
/// \param sort                If true then sort the results in ascending order
/// of distance
///
/// \param neighbors_index     The output tensor that saves the resulting
/// neighbor indices
///
/// \param neighbors_row_splits  Tensor defining the start and end the neighbor
/// indices in each batch item. The size of the tensor is num_query_points + 1
///
/// \param neighbors_distance   The output tensor that saves the resulting
/// neighbor distances.
///
template <class T, class TIndex>
void FixedRadiusSearchCUDA(const Tensor& points,
                           const Tensor& queries,
                           double radius,
                           const Tensor& points_row_splits,
                           const Tensor& queries_row_splits,
                           const Tensor& hash_table_splits,
                           const Tensor& hash_table_index,
                           const Tensor& hash_table_cell_splits,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           const bool sort,
                           Tensor& neighbors_index,
                           Tensor& neighbors_row_splits,
                           Tensor& neighbors_distance);

/// Hybrid search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and if there is less
/// neighbors than requested, the output tensor will be assigned with default
/// values, -1 for indices and 0 for distances. In addition the function
/// returns the number of neighbors for each query.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \param points    Tensor with the 3D point positions. This must be the tensor
///        that was used for building the spatial hash table.
///
/// \param queries    Tensor with the 3D query positions. This may be the same
///                   tensor as \p points.
///
/// \param radius    The search radius.
///
/// \param max_knn   The maximum number of neighbor for each query
///
/// \param points_row_splits    Defines the start and end of the points in each
///        batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_points]
///
/// \param queries_row_splits    Defines the start and end of the queries in
///        each batch item. The size of the tensor is batch_size+1. If there is
///        only 1 batch item then this tensor is [0, num_queries]
///
/// \param hash_table_splits    Tensor defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_index    This is an output of the function
///        BuildSpatialHashTableCPU. This is tensor storing the values of the
///        hash table, which are the indices to the points. The size of the
///        tensor must be equal to the number of points.
///
/// \param hash_table_cell_splits    This is an output of the function
///        BuildSpatialHashTableCPU. The row splits array describing the start
///        and end of each cell.
///
/// \param metric    One of L1, L2, Linf. Defines the distance metric for the
///        search.
///
/// \param neighbors_index     The output tensor that saves the resulting
///        neighbor indices
///
/// \param neighbors_count     The output tensor that saves the number of
///        neighbors for each query points
///
/// \param neighbors_distance   The output tensor that saves the resulting
///        neighbor distances.
///
template <class T, class TIndex>
void HybridSearchCUDA(const Tensor& points,
                      const Tensor& queries,
                      double radius,
                      int max_knn,
                      const Tensor& points_row_splits,
                      const Tensor& queries_row_splits,
                      const Tensor& hash_table_splits,
                      const Tensor& hash_table_index,
                      const Tensor& hash_table_cell_splits,
                      const Metric metric,
                      Tensor& neighbors_index,
                      Tensor& neighbors_count,
                      Tensor& neighbors_distance);
#endif

/// \class FixedRadiusIndex
///
/// \brief FixedRadiusIndex for nearest neighbor range search.
class FixedRadiusIndex : public NNSIndex {
public:
    /// \brief Default Constructor.
    FixedRadiusIndex();

    /// \brief Parameterized Constructor.
    ///
    /// \param dataset_points Provides a set of data points as Tensor for KDTree
    /// construction.
    FixedRadiusIndex(const Tensor& dataset_points, double radius);
    FixedRadiusIndex(const Tensor& dataset_points,
                     double radius,
                     const Dtype& index_dtype);
    ~FixedRadiusIndex();
    FixedRadiusIndex(const FixedRadiusIndex&) = delete;
    FixedRadiusIndex& operator=(const FixedRadiusIndex&) = delete;

public:
    bool SetTensorData(const Tensor& dataset_points,
                       const Dtype& index_dtype = core::Int64) override {
        utility::LogError(
                "FixedRadiusIndex::SetTensorData without radius not "
                "implemented.");
    }

    bool SetTensorData(const Tensor& dataset_points,
                       double radius,
                       const Dtype& index_dtype = core::Int64) override;
    bool SetTensorData(const Tensor& dataset_points,
                       const Tensor& points_row_splits,
                       double radius,
                       const Dtype& index_dtype = core::Int64);

    std::pair<Tensor, Tensor> SearchKnn(const Tensor& query_points,
                                        int knn) const override {
        utility::LogError("FixedRadiusIndex::SearchKnn not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor& query_points,
            const Tensor& radii,
            bool sort = true) const override {
        utility::LogError(
                "FixedRadiusIndex::SearchRadius with multi-radii not "
                "implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor& query_points,
            double radius,
            bool sort = true) const override;
    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor& query_points,
            const Tensor& queries_row_splits,
            double radius,
            bool sort = true) const;

    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(const Tensor& query_points,
                                                    double radius,
                                                    int max_knn) const override;

    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(
            const Tensor& query_points,
            const Tensor& queries_row_splits,
            double radius,
            int max_knn) const;

    const double hash_table_size_factor = 1.0 / 32;
    const int64_t max_hash_tabls_size = 33554432;

protected:
    Tensor points_row_splits_;
    Tensor hash_table_splits_;
    Tensor hash_table_cell_splits_;
    Tensor hash_table_index_;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
