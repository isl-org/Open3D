// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/paddle/misc/ReduceSubarraysSumOps.h"

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/ReduceSubarraysSumOpKernel.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> ReduceSubarraysSum(paddle::Tensor& values,
                                               paddle::Tensor& row_splits) {
    CHECK_TYPE(row_splits, phi::DataType::INT64);

    const auto& attr_type = values.dtype();

    // special treatment for empty values vector
    if (values.shape()[0] == 0) {
        return {InitializedEmptyTensor(values.dtype(), values.shape(),
                                       values.place())};
    }

#define CALL(attr_t, fn)                         \
    if (ComparePaddleDtype<attr_t>(attr_type)) { \
        return {fn<attr_t>(values, row_splits)}; \
    }

    CHECK_SAME_DEVICE_TYPE(values, row_splits);

    if (values.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(int32_t, ReduceSubarraysSumCUDA)
        CALL(int64_t, ReduceSubarraysSumCUDA)
        CALL(float, ReduceSubarraysSumCUDA)
        CALL(double, ReduceSubarraysSumCUDA)
#else
        PD_CHECK(false,
                 "ReduceSubarraysSum was not compiled with CUDA support");
#endif
    } else {
        CALL(int32_t, ReduceSubarraysSumCPU)
        CALL(int64_t, ReduceSubarraysSumCPU)
        CALL(float, ReduceSubarraysSumCPU)
        CALL(double, ReduceSubarraysSumCPU)
    }
    return {paddle::Tensor()};
}

std::vector<paddle::DataType> ReduceSubarraysSumInferDtype(
        const paddle::DataType values_dtype) {
    return {values_dtype};
}

std::vector<std::vector<int64_t>> ReduceSubarraysSumInferShape(
        std::vector<int64_t> values_shape) {
    return {values_shape};
}

PD_BUILD_OP(open3d_reduce_subarrays_sum)
        .Inputs({"values", "row_splits"})
        .Outputs({"sums"})
        .SetKernelFn(PD_KERNEL(ReduceSubarraysSum))
        .SetInferShapeFn(PD_INFER_SHAPE(ReduceSubarraysSumInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(ReduceSubarraysSumInferDtype));
