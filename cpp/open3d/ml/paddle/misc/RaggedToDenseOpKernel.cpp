// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/paddle/misc/RaggedToDenseOpKernel.h"

#include "open3d/ml/impl/misc/RaggedToDense.h"
#include "open3d/ml/paddle/PaddleHelper.h"

template <class T>
paddle::Tensor RaggedToDenseCPU(const paddle::Tensor& values,
                                const paddle::Tensor& row_splits,
                                const int64_t out_col_size,
                                const paddle::Tensor& default_value) {
    auto out_shape = values.shape();
    out_shape.erase(out_shape.begin());
    out_shape.insert(out_shape.begin(),
                     {row_splits.shape()[0] - 1, out_col_size});
    paddle::Tensor out =
            paddle::empty(out_shape, paddle::DataType(ToPaddleDtype<T>()));

    open3d::ml::impl::RaggedToDenseCPU(
            values.data<T>(), row_splits.data<int64_t>(), row_splits.shape()[0],
            out_col_size, default_value.data<T>(), default_value.numel(),
            out.data<T>());

    return out;
}

#define INSTANTIATE(T)                                                   \
    template paddle::Tensor RaggedToDenseCPU<T>(                         \
            const paddle::Tensor&, const paddle::Tensor&, const int64_t, \
            const paddle::Tensor&);

INSTANTIATE(uint8_t)
INSTANTIATE(int8_t)
INSTANTIATE(int16_t)
INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
