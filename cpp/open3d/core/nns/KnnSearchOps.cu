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
#define CASE(NDIM)                                                  \
    case NDIM: {                                                    \
        impl::KnnSearchCUDA<T, NeighborSearchAllocator<T>, NDIM>(   \
                stream, points.GetShape(0), points.GetDataPtr<T>(), \
                queries.GetShape(0), queries.GetDataPtr<T>(),       \
                points_row_splits.GetShape(0),                      \
                points_row_splits.GetDataPtr<int64_t>(),            \
                queries_row_splits.GetShape(0),                     \
                queries_row_splits.GetDataPtr<int64_t>(), knn,      \
                output_allocator);                                  \
    } break;
        CASE(1)
        CASE(2)
        CASE(3)
        CASE(4)
        CASE(5)
        CASE(6)
        CASE(7)
        CASE(8)
        CASE(9)
        CASE(10)
        CASE(12)
        CASE(13)
        CASE(14)
        CASE(15)
        CASE(16)
        CASE(17)
        CASE(18)
        CASE(19)
        CASE(20)
        CASE(21)
        CASE(22)
        CASE(23)
        CASE(24)
        CASE(25)
        CASE(26)
        CASE(27)
        CASE(28)
        CASE(29)
        CASE(30)
        CASE(31)
        CASE(32)
        CASE(33)
        CASE(34)
        CASE(35)
        CASE(36)
        CASE(37)
        CASE(38)
        CASE(39)
        CASE(40)
        CASE(41)
        CASE(42)
        CASE(43)
        CASE(44)
        CASE(45)
        CASE(46)
        CASE(47)
        CASE(48)
        CASE(49)
        CASE(50)
        CASE(51)
        CASE(52)
        CASE(53)
        CASE(54)
        CASE(55)
        CASE(56)
        CASE(57)
        CASE(58)
        CASE(59)
        CASE(60)
        CASE(61)
        CASE(62)
        CASE(63)
        CASE(64)
        CASE(65)
        CASE(66)
        CASE(67)
        CASE(68)
        CASE(69)
        CASE(70)
        CASE(71)
        CASE(72)
        CASE(73)
        CASE(74)
        CASE(75)
        CASE(76)
        CASE(77)
        CASE(78)
        CASE(79)
        CASE(80)
        CASE(81)
        CASE(82)
        CASE(83)
        CASE(84)
        CASE(85)
        CASE(86)
        CASE(87)
        CASE(88)
        CASE(89)
        CASE(90)
        CASE(91)
        CASE(92)
        CASE(93)
        CASE(94)
        CASE(95)
        CASE(96)
        CASE(97)
        CASE(98)
        CASE(99)
        CASE(100)
        default:
            break;
#undef CASE
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
