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
    /// \param dataset_points Provides a set of data points as Tensor for KDTree
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

    std::pair<Tensor, Tensor> SearchHybrid(const Tensor& query_points,
                                           double radius,
                                           int max_knn) const override;

    const double hash_table_size_factor = 1.0 / 32;
    const int64_t max_hash_tabls_size = 33554432;

protected:
    std::vector<int64_t> points_row_splits_;
    std::vector<int64_t> hash_table_splits_;
    Tensor hash_table_cell_splits_;
    Tensor hash_table_index_;
};

/// \class NeighborSearchAllocator
/// The object must implement functions
///         AllocIndices(int32_t** ptr, size_t size) and
///         AllocDistances(T** ptr, size_t size). Both functions should
///         allocate memory and return a pointer to that memory in ptr.
///         Argument size specifies the size of the array as the number of
///         elements. Both functions must accept the argument size==0.
///         In this case ptr does not need to be set.
/// \brief A helper class that allocate output arrays for nearest neighbor range
/// search.
template <class T>
class NeighborSearchAllocator {
public:
    /// Constructor.
    ///
    /// \param device Device on which output arrays are allocated.
    NeighborSearchAllocator(Device device) : device_(device) {}

    /// Allocate array for neighbor indices.
    ///
    /// \param ptr pointer to output array
    /// \param num The number of element to allocate
    void AllocIndices(int64_t** ptr, size_t num) {
        indices_ = Tensor::Empty({int64_t(num)}, Dtype::Int64, device_);
        *ptr = indices_.GetDataPtr<int64_t>();
    }

    /// Allocate array for neighbor indices.
    ///
    /// \param ptr pointer to output array
    /// \param num The number of element to allocate
    /// \param value The default value to initialize array
    void AllocIndices(int64_t** ptr, size_t num, int64_t value) {
        indices_ = Tensor::Full({int64_t(num)}, value, Dtype::Int64, device_);
        *ptr = indices_.GetDataPtr<int64_t>();
    }

    /// Allocate array for neighbor distances.
    ///
    /// \param ptr pointer to output array
    /// \param num The number of element to allocate
    ///
    void AllocDistances(T** ptr, size_t num) {
        distances_ =
                Tensor::Empty({int64_t(num)}, Dtype::FromType<T>(), device_);
        *ptr = distances_.GetDataPtr<T>();
    }

    /// Allocate array for neighbor distances.
    ///
    /// \param ptr pointer to output array
    /// \param num The number of element to allocate
    /// \param value The default value to initialize array
    void AllocDistances(T** ptr, size_t num, T value) {
        distances_ = Tensor::Full({int64_t(num)}, value, Dtype::FromType<T>(),
                                  device_);
        *ptr = distances_.GetDataPtr<T>();
    }

    /// Get indices pointer.
    const int64_t* IndicesPtr() const { return indices_.GetDataPtr<int64_t>(); }

    /// Get distances pointer.
    const T* DistancesPtr() const { return distances_.GetDataPtr<T>(); }

    /// Get indices tensor.
    const Tensor& NeighborsIndex() const { return indices_; }
    /// Get distances tensor.
    const Tensor& NeighborsDistance() const { return distances_; }

private:
    Tensor indices_;
    Tensor distances_;
    Device device_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d
