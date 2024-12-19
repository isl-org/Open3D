// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/RaggedToDenseOpKernel.h"
#include "torch/script.h"

torch::Tensor RaggedToDense(torch::Tensor values,
                            torch::Tensor row_splits,
                            const int64_t out_col_size,
                            torch::Tensor default_value) {
    values = values.contiguous();
    row_splits = row_splits.contiguous();
    default_value = default_value.contiguous();
    CHECK_TYPE(row_splits, kInt64);
    CHECK_SAME_DTYPE(values, default_value);

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim num_rows("num_rows");
        CHECK_SHAPE(row_splits, num_rows + 1);
        if (default_value.sizes().size()) {
            Dim item_size("item_size");
            CHECK_SHAPE_COMBINE_LAST_DIMS(default_value, item_size);
            CHECK_SHAPE_COMBINE_LAST_DIMS(values, Dim(), item_size);
            // check shape tail
            std::vector<int64_t> item_shape(values.sizes().begin() + 1,
                                            values.sizes().end());
            TORCH_CHECK(default_value.sizes().vec() == item_shape,
                        "default_value " + default_value.toString() +
                                " has incompatible with the shape of items in "
                                "values " +
                                values.toString());
        } else  // scalar default_value
        {
            Dim num_values("num_values");
            CHECK_SHAPE_COMBINE_LAST_DIMS(values, num_values);
        }
    }

    // make sure everything is on the same device as 'values'
    auto device = values.device();
    row_splits = row_splits.to(device);
    default_value = default_value.to(device);

    const auto& value_type = values.dtype();

#define CALL(value_t, fn)                                                    \
    if (CompareTorchDtype<value_t>(value_type)) {                            \
        return fn<value_t>(values, row_splits, out_col_size, default_value); \
    }

    if (values.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(uint8_t, RaggedToDenseCUDA)
        CALL(int8_t, RaggedToDenseCUDA)
        CALL(int16_t, RaggedToDenseCUDA)
        CALL(int32_t, RaggedToDenseCUDA)
        CALL(int64_t, RaggedToDenseCUDA)
        CALL(float, RaggedToDenseCUDA)
        CALL(double, RaggedToDenseCUDA)
#else
        TORCH_CHECK(false, "RaggedToDense was not compiled with CUDA support")
#endif
    } else {
        CALL(uint8_t, RaggedToDenseCPU)
        CALL(int8_t, RaggedToDenseCPU)
        CALL(int16_t, RaggedToDenseCPU)
        CALL(int32_t, RaggedToDenseCPU)
        CALL(int64_t, RaggedToDenseCPU)
        CALL(float, RaggedToDenseCPU)
        CALL(double, RaggedToDenseCPU)
    }
    TORCH_CHECK(false, "RaggedToDense does not support " + values.toString() +
                               " as input for values")
    return torch::Tensor();
}

static auto registry = torch::RegisterOperators(
        "open3d::ragged_to_dense(Tensor values, Tensor row_splits, int "
        "out_col_size, Tensor default_value)"
        " -> Tensor out",
        &RaggedToDense);
