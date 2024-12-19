// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/NanoFlannIndex.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace core {
namespace nns {

/// \class NearestNeighborSearch
///
/// \brief A Class for nearest neighbor search.
class NearestNeighborSearch {
public:
    /// Constructor.
    ///
    /// \param dataset_points Dataset points for constructing search index. Must
    /// be 2D, with shape {n, d}.
    // NearestNeighborSearch(const Tensor &dataset_points)
    //     : dataset_points_(dataset_points){};
    NearestNeighborSearch(const Tensor &dataset_points,
                          const Dtype &index_dtype = core::Int32)
        : dataset_points_(dataset_points), index_dtype_(index_dtype){};
    ~NearestNeighborSearch();
    NearestNeighborSearch(const NearestNeighborSearch &) = delete;
    NearestNeighborSearch &operator=(const NearestNeighborSearch &) = delete;

public:
    /// Set index for knn search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool KnnIndex();

    /// Set index for multi-radius search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool MultiRadiusIndex();

    /// Set index for fixed-radius search.
    ///
    /// \param radius optional radius parameter. required for gpu fixed radius
    /// index. \return Returns true if building index success, otherwise false.
    bool FixedRadiusIndex(utility::optional<double> radius = {});

    /// Set index for hybrid search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool HybridIndex(utility::optional<double> radius = {});

    /// Perform knn search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param knn Number of neighbors to search per query point.
    /// \return Pair of Tensors, (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype same as index_dtype_.
    /// - distances: Tensor of shape {n, knn}, same dtype with query_points.
    ///              The distances are squared L2 distances.
    std::pair<Tensor, Tensor> KnnSearch(const Tensor &query_points, int knn);

    /// Perform fixed radius search. All query points share the same radius.
    ///
    /// \param query_points Data points for querying. Must be 2D, with shape {n,
    /// d}.
    /// \param radius Radius.
    /// \param sort Sort the results by distance. Default is True.
    /// \return Tuple of Tensors, (indices, distances, num_neighbors):
    /// - indices: Tensor of shape {total_number_of_neighbors,}, with dtype
    /// same as index_dtype_.
    /// - distances: Tensor of shape {total_number_of_neighbors,}, same dtype
    /// with query_points. The distances are squared L2 distances.
    /// - num_neighbors: Tensor of shape {n+1,}, with dtype Int64. The Tensor is
    /// a prefix sum of the number of neighbors for each query point.
    std::tuple<Tensor, Tensor, Tensor> FixedRadiusSearch(
            const Tensor &query_points, double radius, bool sort = true);

    /// Perform multi-radius search. Each query point has an independent radius.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param radii Radii of query points. Each query point has one radius.
    /// Must be 1D, with shape {n,}.
    /// \return Tuple of Tensors, (indices,distances, num_neighbors):
    /// - indices: Tensor of shape {total_number_of_neighbors,}, with dtype
    /// same as index_dtype_.
    /// - distances: Tensor of shape {total_number_of_neighbors,}, same dtype
    /// with query_points. The distances are squared L2 distances.
    /// - num_neighbors: Tensor of shape {n+1,}, with dtype Int64. The Tensor is
    /// a prefix sum of the number of neighbors for each query point.
    std::tuple<Tensor, Tensor, Tensor> MultiRadiusSearch(
            const Tensor &query_points, const Tensor &radii);

    /// Perform hybrid search.
    ///
    /// \param query_points Data points for querying. Must be 2D, with shape {n,
    /// d}.
    /// \param radius Radius.
    /// \param max_knn Maximum number of neighbor to search per query.
    /// \return Tuple of Tensors, (indices, distances, counts):
    /// - indices: Tensor of shape {n, knn}, with dtype same as index_dtype_.
    /// - distances: Tensor of shape {n, knn}, with same dtype with
    /// query_points. The distances are squared L2 distances.
    /// - counts: Counts of neighbour for each query points. [Tensor
    /// of shape {n}, with dtype same as index_dtype_].
    std::tuple<Tensor, Tensor, Tensor> HybridSearch(const Tensor &query_points,
                                                    const double radius,
                                                    const int max_knn) const;

private:
    bool SetIndex();

    /// Assert a Tensor is not CUDA tensoer. This will be removed in the future.
    void AssertNotCUDA(const Tensor &t) const;

protected:
    std::unique_ptr<NanoFlannIndex> nanoflann_index_;
    std::unique_ptr<nns::FixedRadiusIndex> fixed_radius_index_;
    std::unique_ptr<nns::KnnIndex> knn_index_;
    const Tensor dataset_points_;
    const Dtype index_dtype_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d
