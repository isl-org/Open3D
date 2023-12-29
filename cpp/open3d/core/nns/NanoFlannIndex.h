// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NNSIndex.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

/// \class NanoFlann
///
/// \brief KDTree with NanoFlann for nearest neighbor search.
class NanoFlannIndex : public NNSIndex {
public:
    /// \brief Default Constructor.
    NanoFlannIndex();

    /// \brief Parameterized Constructor.
    ///
    /// \param dataset_points Provides a set of data points as Tensor for KDTree
    /// construction.
    NanoFlannIndex(const Tensor &dataset_points);
    NanoFlannIndex(const Tensor &dataset_points, const Dtype &index_dtype);
    ~NanoFlannIndex();
    NanoFlannIndex(const NanoFlannIndex &) = delete;
    NanoFlannIndex &operator=(const NanoFlannIndex &) = delete;

public:
    bool SetTensorData(const Tensor &dataset_points,
                       const Dtype &index_dtype = core::Int64) override;

    bool SetTensorData(const Tensor &dataset_points,
                       double radius,
                       const Dtype &index_dtype = core::Int64) override {
        utility::LogError(
                "NanoFlannIndex::SetTensorData with radius not implemented.");
    }

    /// Perform K nearest neighbor search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param knn Number of nearest neighbor to search.
    /// \return Pair of Tensors: (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype Int32.
    /// - distainces: Tensor of shape {n, knn}, same dtype with dataset_points.
    std::pair<Tensor, Tensor> SearchKnn(const Tensor &query_points,
                                        int knn) const override;

    /// Perform radius search with multiple radii.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param radii list of radius. Must be 1D, with shape {n, }.
    /// \return Tuple of Tensors: (indices, distances, counts):
    /// - indicecs: Tensor of shape {total_num_neighbors,}, dtype Int32.
    /// - distances: Tensor of shape {total_num_neighbors,}, same dtype with
    /// dataset_points.
    /// - counts: Tensor of shape {n,}, dtype Int64.
    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor &query_points,
            const Tensor &radii,
            bool sort = true) const override;

    /// Perform radius search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param radius Radius.
    /// \return Tuple of Tensors, (indices, distances, counts):
    /// - indicecs: Tensor of shape {total_num_neighbors,}, dtype Int32.
    /// - distances: Tensor of shape {total_num_neighbors,}, same dtype with
    /// dataset_points.
    /// - counts: Tensor of shape {n}, dtype Int64.
    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor &query_points,
            double radius,
            bool sort = true) const override;

    /// Perform hybrid search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param radius Radius.
    /// \param max_knn Maximum number of
    /// neighbor to search per query point.
    /// \return Tuple of Tensors, (indices, distances, counts):
    /// - indices: Tensor of shape {n, knn}, with dtype Int32.
    /// - distances: Tensor of shape {n, knn}, with dtype Float32.
    /// - counts: Counts of neighbour for each query points. [Tensor
    /// of shape {n}, with dtype Int32].
    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(const Tensor &query_points,
                                                    double radius,
                                                    int max_knn) const override;

protected:
    // Tensor dataset_points_;
    std::unique_ptr<NanoFlannIndexHolderBase> holder_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d
