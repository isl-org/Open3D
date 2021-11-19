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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/KnnSearchImpl.cuh"
#include "open3d/core/nns/NeighborSearchAllocator.h"

#define KNN_CUDA_IMPL_DIM(NDIM)                                 \
    impl::KnnSearchCUDA<T, NeighborSearchAllocator<T>, NDIM>(   \
            stream, points.GetShape(0), points.GetDataPtr<T>(), \
            queries.GetShape(0), queries.GetDataPtr<T>(),       \
            points_row_splits.GetShape(0),                      \
            points_row_splits.GetDataPtr<int64_t>(),            \
            queries_row_splits.GetShape(0),                     \
            queries_row_splits.GetDataPtr<int64_t>(), knn, output_allocator);

namespace open3d {
namespace core {
namespace nns {

template <class T>
void KnnSearchCUDA(const Tensor& points,
                   const Tensor& points_row_splits,
                   const Tensor& queries,
                   const Tensor& queries_row_splits,
                   int knn,
                   Tensor& neighbors_index,
                   Tensor& neighbors_distance) {
    const cudaStream_t stream = cuda::GetStream();

    Device device = points.GetDevice();
    NeighborSearchAllocator<T> output_allocator(device);

    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    knn = num_points > knn ? knn : num_points;

    switch (points.GetShape(1)) {
        case 1:
            KNN_CUDA_IMPL_DIM(1);
            break;
        case 2:
            KNN_CUDA_IMPL_DIM(2);
            break;
        case 3:
            KNN_CUDA_IMPL_DIM(3);
            break;
        case 4:
            KNN_CUDA_IMPL_DIM(4);
            break;
        case 5:
            KNN_CUDA_IMPL_DIM(5);
            break;
        case 6:
            KNN_CUDA_IMPL_DIM(6);
            break;
        case 7:
            KNN_CUDA_IMPL_DIM(7);
            break;
        case 8:
            KNN_CUDA_IMPL_DIM(8);
            break;
        case 9:
            KNN_CUDA_IMPL_DIM(9);
            break;
        case 10:
            KNN_CUDA_IMPL_DIM(10);
            break;
        case 11:
            KNN_CUDA_IMPL_DIM(11);
            break;
        case 12:
            KNN_CUDA_IMPL_DIM(12);
            break;
        case 13:
            KNN_CUDA_IMPL_DIM(13);
            break;
        case 14:
            KNN_CUDA_IMPL_DIM(14);
            break;
        case 15:
            KNN_CUDA_IMPL_DIM(15);
            break;
        case 16:
            KNN_CUDA_IMPL_DIM(16);
            break;
        default:
            utility::LogError(
                    "KnnSearchOps only support data with dimension 1 to 16.");
            break;
    }

    neighbors_index =
            output_allocator.NeighborsIndex().View({num_queries, knn});
    neighbors_distance =
            output_allocator.NeighborsDistance().View({num_queries, knn});
}

#define INSTANTIATE(T)                                                        \
    template void KnnSearchCUDA<T>(                                           \
            const Tensor& points, const Tensor& points_row_splits,            \
            const Tensor& queries, const Tensor& queries_row_splits, int knn, \
            Tensor& neighbors_index, Tensor& neighbors_distance);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace nns
}  // namespace core
}  // namespace open3d
