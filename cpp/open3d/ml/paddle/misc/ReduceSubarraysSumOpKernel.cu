// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/impl/misc/ReduceSubarraysSum.cuh"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/ReduceSubarraysSumOpKernel.h"
#include "paddle/extension.h"

template <class T>
paddle::Tensor ReduceSubarraysSumCUDA(const paddle::Tensor& values,
                                      const paddle::Tensor& row_splits) {
    auto place = values.place();
    paddle::Tensor sums =
            paddle::empty({row_splits.shape()[0] - 1},
                          paddle::DataType(ToPaddleDtype<T>()), place);

    auto stream = values.stream();
    open3d::ml::impl::ReduceSubarraysSumCUDA(
            stream, values.data<T>(), values.shape()[0],
            row_splits.data<int64_t>(), row_splits.shape()[0] - 1,
            sums.data<T>());
    return sums;
}
#define INSTANTIATE(T)                                                       \
    template paddle::Tensor ReduceSubarraysSumCUDA<T>(const paddle::Tensor&, \
                                                      const paddle::Tensor&);

INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
