// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/pytorch/misc/ReduceSubarraysSumOps.h"

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/ReduceSubarraysSumOpKernel.h"
#include "torch/script.h"

torch::Tensor ReduceSubarraysSum(torch::Tensor values,
                                 torch::Tensor row_splits) {
    values = values.contiguous();
    row_splits = row_splits.contiguous();
    CHECK_TYPE(row_splits, kInt64);

    const auto& attr_type = values.dtype();

    // special treatment for empty values vector
    if (values.size(0) == 0) {
        return torch::empty_like(values);
    }

#define CALL(attr_t, fn)                        \
    if (CompareTorchDtype<attr_t>(attr_type)) { \
        return fn<attr_t>(values, row_splits);  \
    }

    CHECK_SAME_DEVICE_TYPE(values, row_splits);

    if (values.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
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
