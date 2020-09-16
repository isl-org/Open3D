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
#include "open3d/ml/impl/misc/InvertNeighborsList.cuh"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/InvertNeighborsListOpKernel.h"
#include "torch/script.h"

template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsListCUDA(
        int64_t num_points,
        const torch::Tensor& inp_neighbors_index,
        const torch::Tensor& inp_neighbors_row_splits,
        const torch::Tensor& inp_neighbors_attributes) {
    auto device = inp_neighbors_index.device();
    torch::Tensor neighbors_index =
            torch::empty(inp_neighbors_index.sizes(),
                         torch::dtype(ToTorchDtype<TIndex>()).device(device));
    torch::Tensor neighbors_row_splits = torch::empty(
            {num_points + 1}, torch::dtype(torch::kInt64).device(device));
    torch::Tensor neighbors_attributes =
            torch::empty_like(inp_neighbors_attributes);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto cuda_device_props = at::cuda::getCurrentDeviceProperties();
    const int texture_alignment = cuda_device_props->textureAlignment;

    int num_attributes;
    if (inp_neighbors_attributes.size(0) == 0) {
        num_attributes = 0;
    } else {
        num_attributes = 1;
        for (int i = 1; i < inp_neighbors_attributes.dim(); ++i)
            num_attributes *= inp_neighbors_attributes.size(i);
    }

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    // determine temp_size
    open3d::ml::impl::InvertNeighborsListCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            inp_neighbors_index.data_ptr<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data_ptr<TAttr>()
                           : nullptr,
            num_attributes,
            (int64_t*)inp_neighbors_row_splits.data_ptr<int64_t>(),
            inp_neighbors_row_splits.size(0) - 1,
            neighbors_index.data_ptr<TIndex>(),
            num_attributes ? neighbors_attributes.data_ptr<TAttr>() : nullptr,
            neighbors_index.size(0),
            (int64_t*)neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_row_splits.size(0) - 1);

    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually invert the list
    open3d::ml::impl::InvertNeighborsListCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            inp_neighbors_index.data_ptr<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data_ptr<TAttr>()
                           : nullptr,
            num_attributes,
            (int64_t*)inp_neighbors_row_splits.data_ptr<int64_t>(),
            inp_neighbors_row_splits.size(0) - 1,
            neighbors_index.data_ptr<TIndex>(),
            num_attributes ? neighbors_attributes.data_ptr<TAttr>() : nullptr,
            neighbors_index.size(0),
            (int64_t*)neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_row_splits.size(0) - 1);

    return std::make_tuple(neighbors_index, neighbors_row_splits,
                           neighbors_attributes);
}
#define INSTANTIATE(TIndex, TAttr)                                        \
    template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>      \
    InvertNeighborsListCUDA<TIndex, TAttr>(int64_t, const torch::Tensor&, \
                                           const torch::Tensor&,          \
                                           const torch::Tensor&);

INSTANTIATE(int32_t, int32_t)
INSTANTIATE(int32_t, int64_t)
INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
