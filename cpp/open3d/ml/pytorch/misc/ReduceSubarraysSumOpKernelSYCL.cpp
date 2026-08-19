// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// XPU dispatch wrapper for ReduceSubarraysSum.

#include <c10/xpu/XPUStream.h>

#include "open3d/ml/impl/misc/ReduceSubarraysSumSYCL.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/ReduceSubarraysSumOpKernel.h"
#include "torch/script.h"

template <class T>
torch::Tensor ReduceSubarraysSumSYCL(const torch::Tensor& values,
                                     const torch::Tensor& row_splits) {
    auto device = values.device();
    torch::Tensor sums =
            torch::empty({row_splits.size(0) - 1},
                         torch::dtype(ToTorchDtype<T>()).device(device));

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    open3d::ml::impl::ReduceSubarraysSumSYCL(
            queue, values.data_ptr<T>(), values.size(0),
            row_splits.data_ptr<int64_t>(), row_splits.size(0) - 1,
            sums.data_ptr<T>());

    return sums;
}

#define INSTANTIATE(T)                                                     \
    template torch::Tensor ReduceSubarraysSumSYCL<T>(const torch::Tensor&, \
                                                     const torch::Tensor&);

INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
