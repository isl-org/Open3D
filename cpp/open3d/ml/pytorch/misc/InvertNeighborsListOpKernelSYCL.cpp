// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// XPU dispatch wrapper for InvertNeighborsList. Allocates scratch + output
// tensors on the XPU device and delegates to InvertNeighborsListSYCL.h.

#include <c10/xpu/XPUStream.h>

#include "open3d/ml/impl/misc/InvertNeighborsListSYCL.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/InvertNeighborsListOpKernel.h"
#include "torch/script.h"

template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
InvertNeighborsListSYCL(int64_t num_points,
                        const torch::Tensor& inp_neighbors_index,
                        const torch::Tensor& inp_neighbors_row_splits,
                        const torch::Tensor& inp_neighbors_attributes) {
    auto device = inp_neighbors_index.device();

    torch::Tensor neighbors_index =
            torch::empty(inp_neighbors_index.sizes(),
                         torch::dtype(ToTorchDtype<TIndex>()).device(device));
    torch::Tensor neighbors_row_splits = torch::zeros(
            {num_points + 1}, torch::dtype(torch::kInt64).device(device));
    torch::Tensor neighbors_attributes =
            torch::empty_like(inp_neighbors_attributes);

    // Scratch buffer: uint32_t count array of length num_points
    torch::Tensor count_buf =
            torch::empty({num_points},
                         torch::dtype(ToTorchDtype<uint32_t>()).device(device));

    int num_attributes;
    if (inp_neighbors_attributes.size(0) == 0) {
        num_attributes = 0;
    } else {
        num_attributes = 1;
        for (int i = 1; i < inp_neighbors_attributes.dim(); ++i)
            num_attributes *= inp_neighbors_attributes.size(i);
    }

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    open3d::ml::impl::InvertNeighborsListSYCL(
            queue, count_buf.data_ptr<uint32_t>(),
            inp_neighbors_index.data_ptr<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data_ptr<TAttr>()
                           : nullptr,
            num_attributes,
            inp_neighbors_row_splits.data_ptr<int64_t>(),
            inp_neighbors_row_splits.size(0) - 1,
            neighbors_index.data_ptr<TIndex>(),
            num_attributes ? neighbors_attributes.data_ptr<TAttr>() : nullptr,
            neighbors_index.size(0),
            neighbors_row_splits.data_ptr<int64_t>(),
            static_cast<size_t>(num_points));

    return std::make_tuple(neighbors_index, neighbors_row_splits,
                           neighbors_attributes);
}

#define INSTANTIATE(TIndex, TAttr)                                           \
    template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>         \
    InvertNeighborsListSYCL<TIndex, TAttr>(int64_t, const torch::Tensor&,    \
                                           const torch::Tensor&,             \
                                           const torch::Tensor&);

INSTANTIATE(int32_t, uint8_t)
INSTANTIATE(int32_t, int8_t)
INSTANTIATE(int32_t, int16_t)
INSTANTIATE(int32_t, int32_t)
INSTANTIATE(int32_t, int64_t)
INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
