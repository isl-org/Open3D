// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
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
