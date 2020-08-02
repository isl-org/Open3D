// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/ml/PyTorch/TorchHelper.h"
#include "open3d/ml/impl/misc/KnnSearch.h"
#include "torch/script.h"

using namespace open3d::ml::impl;

namespace {
template <class T>
class OutputAllocator {
public:
    OutputAllocator(torch::Tensor& neighbors_index,
                    torch::Tensor& neighbors_distance,
                    torch::DeviceType device_type,
                    int device_idx)
        : neighbors_index(neighbors_index),
          neighbors_distance(neighbors_distance),
          device_type(device_type),
          device_idx(device_idx) {}

    void AllocIndices(int32_t** ptr, size_t num) {
        neighbors_index = torch::empty(
                {int64_t(num)}, torch::dtype(ToTorchDtype<int32_t>())
                                        .device(device_type, device_idx));
        *ptr = neighbors_index.data_ptr<int32_t>();
    }

    void AllocDistances(T** ptr, size_t num) {
        neighbors_distance = torch::empty(
                {int64_t(num)}, torch::dtype(ToTorchDtype<T>())
                                        .device(device_type, device_idx));
        *ptr = neighbors_distance.data_ptr<T>();
    }

private:
    torch::Tensor& neighbors_index;
    torch::Tensor& neighbors_distance;
    torch::DeviceType device_type;
    int device_idx;
};

template <class T>
class OutputAllocatorTmp {
public:
    OutputAllocatorTmp() {}

    void AllocIndices(int32_t** ptr, size_t num) {
        index.resize(num);
        *ptr = index.data();
    }

    void AllocDistances(T** ptr, size_t num) {
        distance.resize(num);
        *ptr = distance.data();
    }

    std::vector<int32_t> index;
    std::vector<T> distance;
};

}  // namespace

template <class T>
void KnnSearchCPU(const torch::Tensor& points,
                  const torch::Tensor& queries,
                  const int64_t k,
                  const torch::Tensor& points_row_splits,
                  const torch::Tensor& queries_row_splits,
                  const Metric metric,
                  const bool ignore_query_point,
                  const bool return_distances,
                  torch::Tensor& neighbors_index,
                  torch::Tensor& neighbors_row_splits,
                  torch::Tensor& neighbors_distance) {
    const int batch_size = points_row_splits.size(0) - 1;
    OutputAllocator<T> output_allocator(neighbors_index, neighbors_distance,
                                        points.device().type(),
                                        points.device().index());
    if (batch_size == 1) {
        KnnSearchCPU((int64_t*)neighbors_row_splits.data_ptr<int64_t>(),
                     points.size(0), points.data_ptr<T>(), queries.size(0),
                     queries.data_ptr<T>(), k, metric, ignore_query_point,
                     return_distances, output_allocator);
    } else {
        // run radius search for each batch item
        std::vector<OutputAllocatorTmp<T>> tmp_output_allocators(batch_size);
        int64_t last_neighbors_count = 0;
        for (int i = 0; i < batch_size; ++i) {
            const T* const points_i =
                    points.data_ptr<T>() +
                    3 * points_row_splits.data_ptr<int64_t>()[i];
            const T* const queries_i =
                    queries.data_ptr<T>() +
                    3 * queries_row_splits.data_ptr<int64_t>()[i];
            size_t num_points_i = points_row_splits.data_ptr<int64_t>()[i + 1] -
                                  points_row_splits.data_ptr<int64_t>()[i];
            size_t num_queries_i =
                    queries_row_splits.data_ptr<int64_t>()[i + 1] -
                    queries_row_splits.data_ptr<int64_t>()[i];

            int64_t* neighbors_row_splits_i =
                    (int64_t*)(neighbors_row_splits.data_ptr<int64_t>() +
                               queries_row_splits.data_ptr<int64_t>()[i]);
            KnnSearchCPU(neighbors_row_splits_i, num_points_i, points_i,
                         num_queries_i, queries_i, k, metric,
                         ignore_query_point, return_distances,
                         tmp_output_allocators[i]);

            if (i > 0) {
                for (size_t j = 0; j <= num_queries_i; ++j)
                    neighbors_row_splits_i[j] += last_neighbors_count;
            }
            last_neighbors_count = neighbors_row_splits_i[num_queries_i];
        }

        // combine results
        int64_t neighbors_index_size = 0;
        int64_t neighbors_distance_size = 0;
        for (const auto a : tmp_output_allocators) {
            neighbors_index_size += a.index.size();
            neighbors_distance_size += a.distance.size();
        }
        int32_t* neighbors_index_data_ptr;
        T* neighbors_distance_data_ptr;
        output_allocator.AllocIndices(&neighbors_index_data_ptr,
                                      neighbors_index_size);
        output_allocator.AllocDistances(&neighbors_distance_data_ptr,
                                        neighbors_distance_size);

        for (int i = 0; i < batch_size; ++i) {
            const auto& a = tmp_output_allocators[i];
            if (a.index.size()) {
                for (const auto index : a.index) {
                    neighbors_index_data_ptr[0] =
                            index + points_row_splits.data_ptr<int64_t>()[i];
                    ++neighbors_index_data_ptr;
                }
            }
            if (a.distance.size()) {
                memcpy(neighbors_distance_data_ptr, a.distance.data(),
                       a.distance.size() * sizeof(T));
                neighbors_distance_data_ptr += a.distance.size();
            }
        }
    }
}

#define INSTANTIATE(T)                                                    \
    template void KnnSearchCPU<T>(                                        \
            const torch::Tensor& points, const torch::Tensor& queries,    \
            const int64_t k, const torch::Tensor& points_row_splits,      \
            const torch::Tensor& queries_row_splits, const Metric metric, \
            const bool ignore_query_point, const bool return_distances,   \
            torch::Tensor& neighbors_index,                               \
            torch::Tensor& neighbors_row_splits,                          \
            torch::Tensor& neighbors_distance);

INSTANTIATE(float)
INSTANTIATE(double)
