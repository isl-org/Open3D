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
//

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/FixedRadiusSearchImpl.cuh"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/core/nns/NeighborSearchCommon.h"

namespace open3d {
namespace core {
namespace nns {

template <class T>
void BuildSpatialHashTableCUDA(const Tensor& points,
                               double radius,
                               const Tensor& points_row_splits,
                               const Tensor& hash_table_splits,
                               Tensor& hash_table_index,
                               Tensor& hash_table_cell_splits) {
    const cudaStream_t stream = 0;
    int texture_alignment = 512;

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            points.GetShape()[0], points.GetDataPtr<T>(), T(radius),
            points_row_splits.GetShape()[0],
            points_row_splits.GetDataPtr<int64_t>(),
            hash_table_splits.GetDataPtr<uint32_t>(),
            hash_table_cell_splits.GetShape()[0],
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            hash_table_index.GetDataPtr<uint32_t>());

    Device device = points.GetDevice();
    Tensor temp_tensor =
            Tensor::Empty({int64_t(temp_size)}, Dtype::UInt8, device);
    temp_ptr = temp_tensor.GetDataPtr();

    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            points.GetShape()[0], points.GetDataPtr<T>(), T(radius),
            points_row_splits.GetShape()[0],
            points_row_splits.GetDataPtr<int64_t>(),
            hash_table_splits.GetDataPtr<uint32_t>(),
            hash_table_cell_splits.GetShape()[0],
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            hash_table_index.GetDataPtr<uint32_t>());
}

template <class T, class TIndex>
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
                           Tensor& neighbors_distance) {
    const cudaStream_t stream = 0;
    int texture_alignment = 512;

    Device device = points.GetDevice();
    Dtype dtype = points.GetDtype();
    Dtype index_dtype = Dtype::FromType<TIndex>();

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    impl::FixedRadiusSearchCUDA<T, TIndex>(
            stream, temp_ptr, temp_size, texture_alignment,
            neighbors_row_splits.GetDataPtr<int64_t>(), points.GetShape()[0],
            points.GetDataPtr<T>(), queries.GetShape()[0],
            queries.GetDataPtr<T>(), T(radius), points_row_splits.GetShape()[0],
            points_row_splits.GetDataPtr<int64_t>(),
            queries_row_splits.GetShape()[0],
            queries_row_splits.GetDataPtr<int64_t>(),
            hash_table_splits.GetDataPtr<uint32_t>(),
            hash_table_cell_splits.GetShape()[0],
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            hash_table_index.GetDataPtr<uint32_t>(), metric, ignore_query_point,
            return_distances, output_allocator);

    Tensor temp_tensor =
            Tensor::Empty({int64_t(temp_size)}, Dtype::UInt8, device);
    temp_ptr = temp_tensor.GetDataPtr();

    impl::FixedRadiusSearchCUDA<T, TIndex>(
            stream, temp_ptr, temp_size, texture_alignment,
            neighbors_row_splits.GetDataPtr<int64_t>(), points.GetShape()[0],
            points.GetDataPtr<T>(), queries.GetShape()[0],
            queries.GetDataPtr<T>(), T(radius), points_row_splits.GetShape()[0],
            points_row_splits.GetDataPtr<int64_t>(),
            queries_row_splits.GetShape()[0],
            queries_row_splits.GetDataPtr<int64_t>(),
            hash_table_splits.GetDataPtr<uint32_t>(),
            hash_table_cell_splits.GetShape()[0],
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            hash_table_index.GetDataPtr<uint32_t>(), metric, ignore_query_point,
            return_distances, output_allocator);

    Tensor indices_unsorted = output_allocator.NeighborsIndex();
    Tensor distances_unsorted = output_allocator.NeighborsDistance();

    if (!sort) {
        neighbors_index = indices_unsorted;
        neighbors_distance = distances_unsorted;
    } else {
        // Sort indices & distances.
        temp_ptr = nullptr;
        temp_size = 0;

        int64_t num_indices = indices_unsorted.GetShape()[0];
        int64_t num_segments = neighbors_row_splits.GetShape()[0] - 1;
        Tensor indices_sorted =
                Tensor::Empty({num_indices}, index_dtype, device);
        Tensor distances_sorted = Tensor::Empty({num_indices}, dtype, device);

        // Determine temp_size for sorting
        impl::SortPairs(temp_ptr, temp_size, texture_alignment, num_indices,
                        num_segments,
                        neighbors_row_splits.GetDataPtr<int64_t>(),
                        indices_unsorted.GetDataPtr<TIndex>(),
                        distances_unsorted.GetDataPtr<T>(),
                        indices_sorted.GetDataPtr<TIndex>(),
                        distances_sorted.GetDataPtr<T>());

        temp_tensor = Tensor::Empty({int64_t(temp_size)}, Dtype::UInt8, device);
        temp_ptr = temp_tensor.GetDataPtr();

        // Actually run the sorting.
        impl::SortPairs(temp_ptr, temp_size, texture_alignment, num_indices,
                        num_segments,
                        neighbors_row_splits.GetDataPtr<int64_t>(),
                        indices_unsorted.GetDataPtr<TIndex>(),
                        distances_unsorted.GetDataPtr<T>(),
                        indices_sorted.GetDataPtr<TIndex>(),
                        distances_sorted.GetDataPtr<T>());
        neighbors_index = indices_sorted;
        neighbors_distance = distances_sorted;
    }
}

template <class T, class TIndex>
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
                      Tensor& neighbors_distance) {
    const cudaStream_t stream = 0;

    Device device = points.GetDevice();

    NeighborSearchAllocator<T, TIndex> output_allocator(device);

    impl::HybridSearchCUDA<T, TIndex>(
            stream, points.GetShape()[0], points.GetDataPtr<T>(),
            queries.GetShape()[0], queries.GetDataPtr<T>(), T(radius), max_knn,
            points_row_splits.GetShape()[0],
            points_row_splits.GetDataPtr<int64_t>(),
            queries_row_splits.GetShape()[0],
            queries_row_splits.GetDataPtr<int64_t>(),
            hash_table_splits.GetDataPtr<uint32_t>(),
            hash_table_cell_splits.GetShape()[0],
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            hash_table_index.GetDataPtr<uint32_t>(), metric, output_allocator);

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
    neighbors_count = output_allocator.NeighborsCount();
}

#define INSTANTIATE_BUILD(T)                                                  \
    template void BuildSpatialHashTableCUDA<T>(                               \
            const Tensor& points, double radius,                              \
            const Tensor& points_row_splits, const Tensor& hash_table_splits, \
            Tensor& hash_table_index, Tensor& hash_table_cell_splits);

#define INSTANTIATE_RADIUS(T, TIndex)                                          \
    template void FixedRadiusSearchCUDA<T, TIndex>(                            \
            const Tensor& points, const Tensor& queries, double radius,        \
            const Tensor& points_row_splits, const Tensor& queries_row_splits, \
            const Tensor& hash_table_splits, const Tensor& hash_table_index,   \
            const Tensor& hash_table_cell_splits, const Metric metric,         \
            const bool ignore_query_point, const bool return_distances,        \
            const bool sort, Tensor& neighbors_index,                          \
            Tensor& neighbors_row_splits, Tensor& neighbors_distance);

#define INSTANTIATE_HYBRID(T, TIndex)                                          \
    template void HybridSearchCUDA<T, TIndex>(                                 \
            const Tensor& points, const Tensor& queries, double radius,        \
            int max_knn, const Tensor& points_row_splits,                      \
            const Tensor& queries_row_splits, const Tensor& hash_table_splits, \
            const Tensor& hash_table_index,                                    \
            const Tensor& hash_table_cell_splits, const Metric metric,         \
            Tensor& neighbors_index, Tensor& neighbors_count,                  \
            Tensor& neighbors_distance);

INSTANTIATE_BUILD(float)
INSTANTIATE_BUILD(double)

INSTANTIATE_RADIUS(float, int32_t)
INSTANTIATE_RADIUS(float, int64_t)
INSTANTIATE_RADIUS(double, int32_t)
INSTANTIATE_RADIUS(double, int64_t)

INSTANTIATE_HYBRID(float, int32_t)
INSTANTIATE_HYBRID(float, int64_t)
INSTANTIATE_HYBRID(double, int32_t)
INSTANTIATE_HYBRID(double, int64_t)

}  // namespace nns
}  // namespace core
}  // namespace open3d
