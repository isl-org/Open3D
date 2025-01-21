// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/paddle/misc/ReduceSubarraysSumOpKernel.h"

#include "open3d/ml/impl/misc/ReduceSubarraysSum.h"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "paddle/extension.h"

template <class T>
paddle::Tensor ReduceSubarraysSumCPU(const paddle::Tensor& values,
                                     const paddle::Tensor& row_splits) {
    paddle::Tensor sums = paddle::empty({row_splits.shape()[0] - 1},
                                        paddle::DataType(ToPaddleDtype<T>()));

    open3d::ml::impl::ReduceSubarraysSumCPU(
            values.data<T>(), values.shape()[0], row_splits.data<int64_t>(),
            row_splits.shape()[0] - 1, sums.data<T>());
    return sums;
}
#define INSTANTIATE(T)                                                      \
    template paddle::Tensor ReduceSubarraysSumCPU<T>(const paddle::Tensor&, \
                                                     const paddle::Tensor&);

INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
