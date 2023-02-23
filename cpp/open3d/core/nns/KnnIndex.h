// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NNSIndex.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

#ifdef BUILD_CUDA_MODULE
template <class T, class TIndex>
void KnnSearchCUDA(const Tensor& points,
                   const Tensor& points_row_splits,
                   const Tensor& queries,
                   const Tensor& queries_row_splits,
                   int knn,
                   Tensor& neighbors_index,
                   Tensor& neighbors_row_splits,
                   Tensor& neighbors_distance);
#endif

class KnnIndex : public NNSIndex {
public:
    KnnIndex();

    /// \brief Parameterized Constructor.
    ///
    /// \param dataset_points Provides a set of data points as Tensor for KDTree
    /// construction.
    KnnIndex(const Tensor& dataset_points);
    KnnIndex(const Tensor& dataset_points, const Dtype& index_dtype);
    ~KnnIndex();
    KnnIndex(const KnnIndex&) = delete;
    KnnIndex& operator=(const KnnIndex&) = delete;

public:
    bool SetTensorData(const Tensor& dataset_points,
                       const Dtype& index_dtype = core::Int64) override;
    bool SetTensorData(const Tensor& dataset_points,
                       const Tensor& points_row_splits,
                       const Dtype& index_dtype = core::Int64);
    bool SetTensorData(const Tensor& dataset_points,
                       double radius,
                       const Dtype& index_dtype = core::Int64) override {
        utility::LogError(
                "[KnnIndex::SetTensorData with radius not implemented.");
    }

    std::pair<Tensor, Tensor> SearchKnn(const Tensor& query_points,
                                        int knn) const override;

    std::pair<Tensor, Tensor> SearchKnn(const Tensor& query_points,
                                        const Tensor& queries_row_splits,
                                        int knn) const;

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(const Tensor& query_points,
                                                    const Tensor& radii,
                                                    bool sort) const override {
        utility::LogError("KnnIndex::SearchRadius not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(const Tensor& query_points,
                                                    const double radius,
                                                    bool sort) const override {
        utility::LogError("KnnIndex::SearchRadius not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(
            const Tensor& query_points,
            const double radius,
            const int max_knn) const override {
        utility::LogError("KnnIndex::SearchHybrid not implemented.");
    }

protected:
    Tensor points_row_splits_;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
