#include <vector>
#include "open3d/ml/PyTorch/TorchHelper.h"
#include "torch/script.h"

template <class TAttr>
torch::Tensor ReduceSubarraysSumCPU(torch::Tensor values,
                                    torch::Tensor row_splits);

#ifdef CUDA_ENABLED
template <class TAttr>
torch::Tensor ReduceSubarraysSumCUDA(torch::Tensor values,
                                     torch::Tensor row_splits);
#endif

torch::Tensor ReduceSubarraysSum(torch::Tensor values,
                                 torch::Tensor row_splits) {
    CHECK_CONTIGUOUS(values);
    CHECK_CONTIGUOUS(row_splits);
    CHECK_TYPE(row_splits, kInt64);

    const auto& attr_type = values.dtype();

#define CALL(attr_t, fn)                        \
    if (CompareTorchDtype<attr_t>(attr_type)) { \
        return fn<attr_t>(values, row_splits);  \
    }

    CHECK_SAME_DEVICE_TYPE(values, row_splits);
    if (values.is_cuda()) {
#ifdef CUDA_ENABLED
        // pass to cuda function
        CALL(int32_t, ReduceSubarraysSumCUDA)
        CALL(int64_t, ReduceSubarraysSumCUDA)
        CALL(float, ReduceSubarraysSumCUDA)
        CALL(double, ReduceSubarraysSumCUDA)
#else
        TORCH_CHECK(false,
                    "ReduceSubarraysSum was not compiled with CUDA support")
#endif
    } else {
        CALL(int32_t, ReduceSubarraysSumCPU)
        CALL(int64_t, ReduceSubarraysSumCPU)
        CALL(float, ReduceSubarraysSumCPU)
        CALL(double, ReduceSubarraysSumCPU)
    }
    return torch::Tensor();
}

static auto registry = torch::RegisterOperators(
        "open3d::reduce_subarrays_sum(Tensor values, Tensor row_splits)"
        " -> Tensor sums",
        &ReduceSubarraysSum);
