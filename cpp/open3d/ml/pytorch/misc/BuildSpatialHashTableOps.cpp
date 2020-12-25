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

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

template <class T>
void BuildSpatialHashTableCPU(const torch::Tensor& points,
                              double radius,
                              const torch::Tensor& points_row_splits,
                              const std::vector<uint32_t>& hash_table_splits,
                              torch::Tensor& hash_table_index,
                              torch::Tensor& hash_table_cell_splits);
#ifdef BUILD_CUDA_MODULE
template <class T>
void BuildSpatialHashTableCUDA(const torch::Tensor& points,
                               double radius,
                               const torch::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               torch::Tensor& hash_table_index,
                               torch::Tensor& hash_table_cell_splits);
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BuildSpatialHashTable(
        torch::Tensor points,
        double radius,
        torch::Tensor points_row_splits,
        double hash_table_size_factor,
        int64_t max_hash_table_size) {
    // ensure that these tensors are on the cpu
    points_row_splits = points_row_splits.to(torch::kCPU);
    points = points.contiguous();
    points_row_splits = points_row_splits.contiguous();
    CHECK_TYPE(points_row_splits, kInt64);

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim batch_size("batch_size");

    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);

    const auto& point_type = points.dtype();

    std::vector<uint32_t> hash_table_splits(batch_size.value() + 1, 0);
    for (int i = 0; i < batch_size.value(); ++i) {
        int64_t num_points_i = points_row_splits.data_ptr<int64_t>()[i + 1] -
                               points_row_splits.data_ptr<int64_t>()[i];

        int64_t hash_table_size = std::min<int64_t>(
                std::max<int64_t>(hash_table_size_factor * num_points_i, 1),
                max_hash_table_size);
        hash_table_splits[i + 1] = hash_table_splits[i] + hash_table_size;
    }

    auto device = points.device().type();
    auto device_idx = points.device().index();

    torch::Tensor hash_table_index = torch::empty(
            {points.size(0)},
            torch::dtype(ToTorchDtype<int32_t>()).device(device, device_idx));

    torch::Tensor hash_table_cell_splits = torch::empty(
            {hash_table_splits.back() + 1},
            torch::dtype(ToTorchDtype<int32_t>()).device(device, device_idx));

    torch::Tensor out_hash_table_splits = torch::empty(
            {batch_size.value() + 1}, torch::dtype(ToTorchDtype<int32_t>()));
    for (size_t i = 0; i < hash_table_splits.size(); ++i) {
        out_hash_table_splits.data_ptr<int32_t>()[i] = hash_table_splits[i];
    }

#define FN_PARAMETERS                                                       \
    points, radius, points_row_splits, hash_table_splits, hash_table_index, \
            hash_table_cell_splits

#define CALL(type, fn)                                                   \
    if (CompareTorchDtype<type>(point_type)) {                           \
        fn<type>(FN_PARAMETERS);                                         \
        return std::make_tuple(hash_table_index, hash_table_cell_splits, \
                               out_hash_table_splits);                   \
    }

    if (points.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(float, BuildSpatialHashTableCUDA)
#else
        TORCH_CHECK(false,
                    "BuildSpatialHashTable was not compiled with CUDA support")
#endif
    } else {
        CALL(float, BuildSpatialHashTableCPU)
        CALL(double, BuildSpatialHashTableCPU)
    }
    TORCH_CHECK(false, "BuildSpatialHashTable does not support " +
                               points.toString() + " as input for points")
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

static auto registry = torch::RegisterOperators(
        "open3d::build_spatial_hash_table(Tensor points, float radius, Tensor "
        "points_row_splits, float hash_table_size_factor, int "
        "max_hash_table_size=33554432) -> (Tensor hash_table_index, Tensor "
        "hash_table_cell_splits, Tensor hash_table_splits)",
        &BuildSpatialHashTable);
