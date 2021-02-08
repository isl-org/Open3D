// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

namespace open3d {
namespace core {
namespace nns {

/// \class FixedRadiusIndex
///
/// \brief FixedRadiusIndex for nearest neighbor range search.
class FixedRadiusIndex : public NNSIndex {
public:
    /// \brief Default Constructor.
    FixedRadiusIndex();

    /// \brief Parameterized Constructor.
    ///
    /// \param tensor Provides a set of data points as Tensor for KDTree
    /// construction.
    FixedRadiusIndex(const Tensor& dataset_points, double radius);
    ~FixedRadiusIndex();
    FixedRadiusIndex(const FixedRadiusIndex&) = delete;
    FixedRadiusIndex& operator=(const FixedRadiusIndex&) = delete;

public:
    bool SetTensorData(const Tensor& dataset_points) override {
        utility::LogError(
                "FixedRadiusIndex::SetTensorData witout radius not "
                "implemented.");
    }

    bool SetTensorData(const Tensor& dataset_points, double radius) override;

    std::pair<Tensor, Tensor> SearchKnn(const Tensor& query_points,
                                        int knn) const override {
        utility::LogError("FixedRadiusIndex::SearchKnn not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor& query_points, const Tensor& radii) const override {
        utility::LogError(
                "FixedRadiusIndex::SearchRadius with multi-radii not "
                "implemented.");
    }
    /// Perform radius search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param radius Radius.
    /// \return Tuple of Tensors, (indices, distances, num_neighbors):
    /// - indicecs: Tensor of shape {total_num_neighbors,}, dtype Int64.
    /// - distances: Tensor of shape {total_num_neighbors,}, same dtype with
    /// dataset_points.
    /// - num_neighbors: Tensor of shape {n}, dtype Int64.
    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor& query_points, double radius) const override;

    std::pair<Tensor, Tensor> SearchHybrid(const Tensor& query_points,
                                           float radius,
                                           int max_knn) const override {
        utility::LogError("FixedRadiusIndex::SearchHybrid not implemented.");
    }

    const double hash_table_size_factor = 1 / 32;
    const int64_t max_hash_tabls_size = 10000;

protected:
    std::vector<int64_t> points_row_splits_;
    std::vector<uint32_t> hash_table_splits_;
    std::vector<uint32_t> out_hash_table_splits_;
    Tensor hash_table_cell_splits_;
    Tensor hash_table_index_;
};

template <class T>
class NeighborSearchAllocator {
public:
    NeighborSearchAllocator(Device device) : device_(device) {}

    void AllocIndices(int32_t** ptr, size_t num) {
        neighbors_index = Tensor::Empty({int64_t(num)}, Dtype::Int32, device_);
        *ptr = neighbors_index.GetDataPtr<int32_t>();
    }

    void AllocDistances(T** ptr, size_t num) {
        neighbors_distance =
                Tensor::Empty({int64_t(num)}, Dtype::FromType<T>(), device_);
        *ptr = neighbors_distance.GetDataPtr<T>();
    }

    const int32_t* IndicesPtr() const {
        return neighbors_index.GetDataPtr<int32_t>();
    }

    const T* DistancesPtr() const { return neighbors_distance.GetDataPtr<T>(); }

    const Tensor& NeighborsIndex() const { return neighbors_index; }
    const Tensor& NeighborsDistance() const { return neighbors_distance; }

private:
    Tensor neighbors_index;
    Tensor neighbors_distance;
    Device device_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d
