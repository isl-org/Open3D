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
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/KnnSearchImplNew.cuh"
#include "open3d/core/nns/L2Select.cuh"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {
namespace nns {

template <class T>
void KnnSearchCUDASingle(const Tensor& points,
                         const Tensor& queries,
                         int knn,
                         Tensor& neighbors_index,
                         Tensor& neighbors_distance) {
    const cudaStream_t stream = cuda::GetStream();

    Device device = points.GetDevice();
    Dtype dtype = points.GetDtype();
    NeighborSearchAllocator<T> output_allocator(device);

    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    int dim = points.GetShape(1);
    knn = num_points > knn ? knn : num_points;

    // Calculate norms, |d|^2, |q|^2.
    Tensor point_norms = points.Mul(points).Sum({1});
    Tensor query_norms = queries.Mul(queries).Sum({1});

    // Divide queries and points into chunks (rows and cols).
    int tile_rows = 0;
    int tile_cols = 0;
    impl::chooseTileSize(num_queries, num_points, dim, sizeof(T), tile_rows,
                         tile_cols);
    int num_cols = utility::DivUp(num_points, tile_cols);

    // Allocate temporary memory space.
    Tensor temp_distances =
            Tensor::Empty({tile_rows, tile_cols}, dtype, device);
    Tensor buf_distances =
            Tensor::Empty({tile_rows, num_cols * knn}, dtype, device);
    Tensor buf_indices =
            Tensor::Empty({tile_rows, num_cols * knn}, dtype, device);

    for (int i = 0; i < num_queries; i += tile_rows) {
        int num_queries_i = std::min(tile_rows, num_queries - i);
        Tensor queries_i = queries.Slice(0, i, i + num_queries_i);
        Tensor query_norms_i = query_norms.Slice(0, i, i + num_queries_i);
        Tensor buf_distances_row_view =
                buf_distances.Slice(0, 0, num_queries_i);
        Tensor buf_indices_row_view = buf_indices.Slice(0, 0, num_queries_i);
        for (int j = 0; j < num_points; j += tile_cols) {
            int num_points_j = std::min(tile_cols, num_points - j);
            int col_j = j / tile_cols;
            Tensor points_j = points.Slice(0, j, j + num_points_j);
            Tensor point_norms_j = point_norms.Slice(0, j, j + num_points_j);

            Tensor temp_distances_view =
                    temp_distances.Slice(0, 0, num_queries_i)
                            .Slice(1, 0, num_points_j);
            //     temp_distances_view = temp_distances_view.SetItem(0);
            Tensor buf_distances_col_view = buf_distances_row_view.Slice(
                    1, knn * col_j, (knn + 1) * col_j);
            Tensor buf_indices_col_view = buf_indices_row_view.Slice(
                    1, knn * col_j, (knn + 1) * col_j);

            // Calculate -2*d*q
            AddMM(queries_i, points_j.T(), temp_distances_view, -2.0, 0.0);

            // if (tile_cols == num_points_j) {
            runL2SelectMin<T>(stream, temp_distances_view, point_norms_j,
                              buf_distances_col_view, buf_indices_col_view,
                              knn);
            // } else {
            // }
        }
    }
    neighbors_index =
            output_allocator.NeighborsIndex().View({num_queries, knn});
    neighbors_distance =
            output_allocator.NeighborsDistance().View({num_queries, knn});
}
template <class T>
void KnnSearchCUDANew(const Tensor& points,
                      const Tensor& points_row_splits,
                      const Tensor& queries,
                      const Tensor& queries_row_splits,
                      int knn,
                      Tensor& neighbors_index,
                      Tensor& neighbors_distance) {
    KnnSearchCUDASingle<T>(points, queries, knn, neighbors_index,
                           neighbors_distance);
}

#define INSTANTIATE(T)                                                        \
    template void KnnSearchCUDANew<T>(                                        \
            const Tensor& points, const Tensor& points_row_splits,            \
            const Tensor& queries, const Tensor& queries_row_splits, int knn, \
            Tensor& neighbors_index, Tensor& neighbors_distance);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace nns
}  // namespace core
}  // namespace open3d
