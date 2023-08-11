// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/impl/misc/ReduceSubarraysSum.cuh"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/ReduceSubarraysSumOpKernel.h"
#include "torch/script.h"

template <class T>
torch::Tensor ReduceSubarraysSumCUDA(const torch::Tensor& values,
                                     const torch::Tensor& row_splits) {
    auto device = values.device();
    torch::Tensor sums =
            torch::empty({row_splits.size(0) - 1},
                         torch::dtype(ToTorchDtype<T>()).device(device));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto cuda_device_props = at::cuda::getCurrentDeviceProperties();
    open3d::ml::impl::ReduceSubarraysSumCUDA(
            stream, values.data_ptr<T>(), values.size(0),
            row_splits.data_ptr<int64_t>(), row_splits.size(0) - 1,
            sums.data_ptr<T>());
    return sums;
}
#define INSTANTIATE(T)                                                     \
    template torch::Tensor ReduceSubarraysSumCUDA<T>(const torch::Tensor&, \
                                                     const torch::Tensor&);

INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
