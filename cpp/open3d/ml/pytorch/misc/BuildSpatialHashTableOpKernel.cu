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

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/impl/misc/FixedRadiusSearch.cuh"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

template <class T>
void BuildSpatialHashTableCUDA(const torch::Tensor& points,
                               double radius,
                               const torch::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               torch::Tensor& hash_table_index,
                               torch::Tensor& hash_table_cell_splits) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto cuda_device_props = at::cuda::getCurrentDeviceProperties();
    const int texture_alignment = cuda_device_props->textureAlignment;

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    // determine temp_size
    open3d::ml::impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.size(0),
            points.data_ptr<T>(), T(radius), points_row_splits.size(0),
            points_row_splits.data_ptr<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>());

    auto device = points.device();
    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually build the table
    open3d::ml::impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.size(0),
            points.data_ptr<T>(), T(radius), points_row_splits.size(0),
            points_row_splits.data_ptr<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>());
}

#define INSTANTIATE(T)                                          \
    template void BuildSpatialHashTableCUDA<T>(                 \
            const torch::Tensor&, double, const torch::Tensor&, \
            const std::vector<uint32_t>&, torch::Tensor&, torch::Tensor&);

INSTANTIATE(float)
