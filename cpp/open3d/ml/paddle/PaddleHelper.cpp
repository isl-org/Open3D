// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PaddleHelper.h"

paddle::Tensor InitializedEmptyTensor(const phi::DataType dtype,
                                      const phi::IntArray& shape,
                                      const phi::Place& place) {
    switch (dtype) {
        case phi::DataType::INT8:
            return InitializedEmptyTensor<int8_t>(shape, place);
            break;
        case phi::DataType::UINT8:
            return InitializedEmptyTensor<uint8_t>(shape, place);
            break;
        case phi::DataType::INT16:
            return InitializedEmptyTensor<int16_t>(shape, place);
            break;
        case phi::DataType::FLOAT32:
            return InitializedEmptyTensor<float>(shape, place);
            break;
        case phi::DataType::INT32:
            return InitializedEmptyTensor<int>(shape, place);
            break;
        case phi::DataType::FLOAT64:
            return InitializedEmptyTensor<double>(shape, place);
            break;
        case phi::DataType::INT64:
            return InitializedEmptyTensor<int64_t>(shape, place);
            break;
        default:
            PD_CHECK(false,
                     "Only support phi::DataType as `INT8`, `UINT8`, `INT16`, "
                     "`FLOAT32`, `FLOAT64`, "
                     "`INT32` and `INT64` but got %s.",
                     phi::DataTypeToString(dtype));
    }
}

paddle::Tensor Arange(const int end, const paddle::Place& place) {
    PD_CHECK(end > 0, "end:%d ,end must greater than 0", end);
    auto start_tensor = paddle::zeros({1}, paddle::DataType::INT32, place);
    auto end_tensor = paddle::experimental::full(
            {1}, end, paddle::DataType::INT32, place);
    auto step_tensor =
            paddle::experimental::full({1}, 1, paddle::DataType::INT32, place);
    return paddle::experimental::arange(start_tensor, end_tensor, step_tensor,
                                        paddle::DataType::INT32, place);
}

paddle::Tensor Transpose(const paddle::Tensor& t, int64_t dim0, int64_t dim1) {
    int len = t.shape().size();
    dim0 = dim0 >= 0 ? dim0 : len + dim0;
    dim1 = dim1 >= 0 ? dim1 : len + dim1;
    PD_CHECK(dim0 >= 0 && dim0 < len,
             "dim0 not in range"
             "dim0:%d ,range:%d",
             dim0, len);
    PD_CHECK(dim1 >= 0 && dim1 < len,
             "dim1 not in range"
             "dim1:%d ,range:%d",
             dim1, len);
    std::vector<int> transpose_perm(len);
    std::iota(transpose_perm.begin(), transpose_perm.end(), 0);
    transpose_perm[dim0] = dim1;
    transpose_perm[dim1] = dim0;
    return paddle::experimental::transpose(t, transpose_perm);
}
