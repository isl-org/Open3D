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
