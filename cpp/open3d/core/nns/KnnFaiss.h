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

namespace faiss {
struct Index;
namespace gpu {
class StandardGpuResources;
}
}  // namespace faiss

namespace open3d {
namespace core {
namespace nns {

/// \class KnnFaiss
///
/// \brief Faiss for nearest neighbor search.
class KnnFaiss {
public:
    /// \brief Default Constructor.
    KnnFaiss();
    /// \brief Parameterized Constructor.
    ///
    /// \param tensor Provides tensor from which Faiss Index is constructed.
    KnnFaiss(const Tensor &tensor);
    ~KnnFaiss();
    KnnFaiss(const KnnFaiss &) = delete;
    KnnFaiss &operator=(const KnnFaiss &) = delete;

public:
    /// Sets the data for the Faiss Index from a tensor.
    ///
    /// \param data Data points for Faiss Index Construction. Must be
    /// Float32 type and 2D, with shape {n, d}
    /// \return Returns true if the construction success, otherwise false.
    bool SetTensorData(const Tensor &data);

    /// Perform K nearest neighbor search.
    ///
    /// \param query Query points. Must be Float32 type and 2D, with shape {n,
    /// d}.
    /// \param knn Number of nearest neighbor to search.
    /// \return Pair of Tensors: (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype Int64.
    /// - distances: Tensor of shape {n, knn}, with dtype Float32.
    std::pair<Tensor, Tensor> SearchKnn(const Tensor &query, int knn) const;

    /// Perform hybrid search.
    ///
    /// \param query Query points. Must be Float32 type and 2D, with shape {n,
    /// d}.
    /// \param radius Radius.
    /// \param max_knn Maximum number of neighbor to search per query.
    /// \return Pair of Tensors, (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype Int64.
    /// - distances: Tensor of shape {n, knn}, with dtype Float32.
    std::pair<Tensor, Tensor> SearchHybrid(const Tensor &query,
                                           float radius,
                                           int max_knn) const;

protected:
    std::unique_ptr<faiss::Index> index;
#ifdef BUILD_CUDA_MODULE
    std::unique_ptr<faiss::gpu::StandardGpuResources> res;
#endif
    int64_t dimension_ = 0;
    int64_t dataset_size_ = 0;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
