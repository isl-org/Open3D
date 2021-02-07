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

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NNSIndex.h"
#include "open3d/utility/Console.h"

// Forward declarations.
namespace faiss {
struct Index;
namespace gpu {
class StandardGpuResources;
}
}  // namespace faiss

namespace open3d {
namespace core {
namespace nns {

/// \class FaissIndex
///
/// \brief Faiss for nearest neighbor search.
class FaissIndex : public NNSIndex {
public:
    /// \brief Default Constructor.
    FaissIndex();
    /// \brief Parameterized Constructor.
    ///
    /// \param tensor Provides tensor from which Faiss Index is constructed.
    FaissIndex(const Tensor &dataset_points);
    ~FaissIndex();
    FaissIndex(const FaissIndex &) = delete;
    FaissIndex &operator=(const FaissIndex &) = delete;

public:
    // dataset_points must be float32.
    bool SetTensorData(const Tensor &dataset_points) override;

    bool SetTensorData(const Tensor &dataset_points, double radius) override {
        utility::LogError(
                "FaissIndex::SetTensorData with radius not implemented.");
    }

    // query_points must be float32.
    std::pair<Tensor, Tensor> SearchKnn(const Tensor &query_points,
                                        int knn) const override;

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(const Tensor &query_points,
                                                    const Tensor &radii,
                                                    bool sort) const override {
        utility::LogError("FaissIndex::SearchHybrid not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(const Tensor &query_points,
                                                    double radius,
                                                    bool sort) const override {
        utility::LogError("FaissIndex::SearchHybrid not implemented.");
    }

    // query_points must be float32.
    std::pair<Tensor, Tensor> SearchHybrid(const Tensor &query_points,
                                           double radius,
                                           int max_knn) const override;

protected:
    std::unique_ptr<faiss::Index> index;
#ifdef BUILD_CUDA_MODULE
    std::unique_ptr<faiss::gpu::StandardGpuResources> res;
#endif
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
