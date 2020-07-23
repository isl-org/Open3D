#include <ATen/cuda/CUDAContext.h>
#include "open3d/ml/PyTorch/TorchHelper.h"
#include "open3d/ml/impl/misc/ReduceSubarraysSum.cuh"
#include "torch/script.h"

template <class T>
torch::Tensor ReduceSubarraysSumCUDA(torch::Tensor values,
                                     torch::Tensor row_splits) {
    auto device = values.device().type();
    auto device_idx = values.device().index();
    torch::Tensor sums = torch::empty(
            {row_splits.size(0) - 1},
            torch::dtype(ToTorchDtype<T>()).device(device, device_idx));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto cuda_device_props = at::cuda::getCurrentDeviceProperties();
    const int texture_alignment = cuda_device_props->textureAlignment;

    open3d::ml::impl::ReduceSubarraysSumCUDA(
            stream, values.data_ptr<T>(), values.size(0),
            row_splits.data_ptr<int64_t>(), row_splits.size(0) - 1,
            sums.data_ptr<T>());
    return sums;
}
#define INSTANTIATE(T)                                              \
    template torch::Tensor ReduceSubarraysSumCUDA<T>(torch::Tensor, \
                                                     torch::Tensor);

INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
