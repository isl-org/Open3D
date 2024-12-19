// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace nns {

/// \class NanoFlann
///
/// \brief KDTree with NanoFlann for nearest neighbor search.
class NNSIndex {
public:
    /// \brief Default Constructor.
    NNSIndex() {}
    virtual ~NNSIndex() {}
    NNSIndex(const NNSIndex &) = delete;
    NNSIndex &operator=(const NNSIndex &) = delete;

public:
    /// Set the data for the nearest neighbor search.
    ///
    /// \param dataset_points Dataset points for KDTree construction. Must be
    /// 2D, with shape {n, d}.
    /// \return Returns true if the construction success, otherwise false.
    virtual bool SetTensorData(const Tensor &dataset_points,
                               const Dtype &index_dtype) = 0;

    /// Set the data for the nearest neighbor search.
    ///
    /// \param dataset_points Dataset points for KDTree construction. Must be
    /// 2D, with shape {n, d}.
    /// \return Returns true if the construction success, otherwise false.
    virtual bool SetTensorData(const Tensor &dataset_points,
                               double radius,
                               const Dtype &index_dtype) = 0;

    /// Perform K nearest neighbor search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param knn Number of nearest neighbor to search.
    /// \return Pair of Tensors: (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype Int32.
    /// - distances: Tensor of shape {n, knn}, same dtype with dataset_points.
    virtual std::pair<Tensor, Tensor> SearchKnn(const Tensor &query_points,
                                                int knn) const = 0;

    /// Perform radius search with multiple radii.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param radii list of radius. Must be 1D, with shape {n, }.
    /// \return Tuple of Tensors: (indices, distances, num_neighbors):
    /// - indices: Tensor of shape {total_num_neighbors,}, dtype Int32.
    /// - distances: Tensor of shape {total_num_neighbors,}, same dtype with
    /// dataset_points.
    /// - num_neighbors: Tensor of shape {n,}, dtype Int32.
    virtual std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor &query_points,
            const Tensor &radii,
            bool sort) const = 0;

    /// Perform radius search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param radius Radius.
    /// \return Tuple of Tensors, (indices, distances, num_neighbors):
    /// - indices: Tensor of shape {total_num_neighbors,}, dtype Int32.
    /// - distances: Tensor of shape {total_num_neighbors,}, same dtype with
    /// dataset_points.
    /// - num_neighbors: Tensor of shape {n}, dtype Int32.
    virtual std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor &query_points, double radius, bool sort) const = 0;

    /// Perform hybrid search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param radius Radius.
    /// \param max_knn Maximum number of
    /// neighbor to search per query point.
    /// \return Tuple of Tensors, (indices, distances, counts):
    /// - indices: Tensor of shape {n, knn}, with dtype Int32.
    /// - distances: Tensor of shape {n, knn}, with dtype Float32.
    /// - counts: Tensor of shape {n, 1}, with dtype Int32.
    virtual std::tuple<Tensor, Tensor, Tensor> SearchHybrid(
            const Tensor &query_points, double radius, int max_knn) const = 0;

    /// Get dimension of the dataset points.
    /// \return dimension of dataset points.
    int GetDimension() const;

    /// Get size of the dataset points.
    /// \return number of points in dataset.
    size_t GetDatasetSize() const;

    /// Get dtype of the dataset points.
    /// \return dtype of dataset points.
    Dtype GetDtype() const;

    /// Get device of the dataset points.
    /// \return device of dataset points.
    Device GetDevice() const;

    /// Get dtype of indices.
    /// \return dtype of indices.
    Dtype GetIndexDtype() const;

protected:
    Tensor dataset_points_;
    Dtype index_dtype_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d