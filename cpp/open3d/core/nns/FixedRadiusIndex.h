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

template <class T>
void BuildSpatialHashTableCPU(const Tensor& points,
                              double radius,
                              const Tensor& points_row_splits,
                              const Tensor& hash_table_splits,
                              Tensor& hash_table_index,
                              Tensor& hash_table_cell_splits);
template <class T>
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

#ifdef BUILD_CUDA_MODULE
template <class T>
void BuildSpatialHashTableCUDA(const Tensor& points,
                               double radius,
                               const Tensor& points_row_splits,
                               const Tensor& hash_table_splits,
                               Tensor& hash_table_index,
                               Tensor& hash_table_cell_splits);

template <class T>
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

template <class T>
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

    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(const Tensor& query_points,
                                                    double radius,
                                                    int max_knn) const override;

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
