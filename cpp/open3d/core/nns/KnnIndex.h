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
    ~KnnIndex();
    KnnIndex(const KnnIndex&) = delete;
    KnnIndex& operator=(const KnnIndex&) = delete;

public:
    bool SetTensorData(const Tensor& dataset_points) override;
    bool SetTensorData(const Tensor& dataset_points,
                       const Tensor& points_row_splits);
    bool SetTensorData(const Tensor& dataset_points, double radius) override {
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
                                                    double radius,
                                                    bool sort) const override {
        utility::LogError("KnnIndex::SearchRadius not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(
            const Tensor& query_points,
            double radius,
            int max_knn) const override {
        utility::LogError("KnnIndex::SearchHybrid not implemented.");
    }

protected:
    Tensor points_row_splits_;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
