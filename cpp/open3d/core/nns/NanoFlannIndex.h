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
    ~NanoFlannIndex();
    NanoFlannIndex(const NanoFlannIndex &) = delete;
    NanoFlannIndex &operator=(const NanoFlannIndex &) = delete;

public:
    bool SetTensorData(const Tensor &dataset_points) override;

    bool SetTensorData(const Tensor &dataset_points, double radius) override {
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
    /// - counts: Tensor of shape {n,}, dtype Int32.
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
    /// - counts: Tensor of shape {n}, dtype Int32.
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
