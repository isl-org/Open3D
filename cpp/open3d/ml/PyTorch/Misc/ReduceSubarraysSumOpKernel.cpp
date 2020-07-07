#include "open3d/ml/PyTorch/TorchHelper.h"
#include "open3d/ml/impl/misc/ReduceSubarraysSum.h"
#include "torch/script.h"

template <class T>
torch::Tensor ReduceSubarraysSumCPU(torch::Tensor values,
                                    torch::Tensor row_splits) {
    torch::Tensor sums = torch::empty({row_splits.size(0) - 1},
                                      torch::dtype(ToTorchDtype<T>()));

    open3d::ml::impl::ReduceSubarraysSumCPU(
            values.data_ptr<T>(), values.size(0),
            row_splits.data_ptr<int64_t>(), row_splits.size(0) - 1,
            sums.data_ptr<T>());
    return sums;
}
#define INSTANTIATE(T)                                             \
    template torch::Tensor ReduceSubarraysSumCPU<T>(torch::Tensor, \
                                                    torch::Tensor);

INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
