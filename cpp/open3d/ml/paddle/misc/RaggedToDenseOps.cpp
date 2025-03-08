// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/RaggedToDenseOpKernel.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> RaggedToDense(paddle::Tensor& values,
                                          paddle::Tensor& row_splits,
                                          paddle::Tensor& default_value,
                                          const int64_t out_col_size) {
    CHECK_TYPE(row_splits, phi::DataType::INT64);
    CHECK_SAME_DTYPE(values, default_value);

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim num_rows("num_rows");
        CHECK_SHAPE(row_splits, num_rows + 1);
        if (default_value.shape().size()) {
            Dim item_size("item_size");
            CHECK_SHAPE_COMBINE_LAST_DIMS(default_value, item_size);
            CHECK_SHAPE_COMBINE_LAST_DIMS(values, Dim(), item_size);
            auto value_shape = values.shape();

            // check shape tail
            std::vector<int64_t> item_shape(value_shape.begin() + 1,
                                            value_shape.end());
            auto default_value_shape = default_value.shape();
            PD_CHECK(default_value_shape == item_shape,
                     "default_value " +
                             phi::DataTypeToString(default_value.dtype()) +
                             "has incompatible with the shape of items in "
                             "values" +
                             TensorInfoStr({values}));
        } else  // scalar default_value
        {
            Dim num_values("num_values");
            CHECK_SHAPE_COMBINE_LAST_DIMS(values, num_values);
        }
    }

    // make sure everything is on the same place as 'values'
    auto place = values.place();
    row_splits = row_splits.copy_to(place, false);
    default_value = default_value.copy_to(place, false);

    const auto& value_type = values.dtype();

#define CALL(value_t, fn)                                                      \
    if (ComparePaddleDtype<value_t>(value_type)) {                             \
        return {fn<value_t>(values, row_splits, out_col_size, default_value)}; \
    }

    if (values.is_gpu()) {
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
        PD_CHECK(false, "RaggedToDense was not compiled with CUDA support");
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
    PD_CHECK(false, "RaggedToDense does not support " +
                            phi::DataTypeToString(values.dtype()) +
                            " as input for values");
}

std::vector<paddle::DataType> RaggedToDenseInferDtype(
        const paddle::DataType values_dtype) {
    return {values_dtype};
}

std::vector<std::vector<int64_t>> RaggedToDenseInferShape(
        std::vector<int64_t> values_shape,
        std::vector<int64_t> row_splits_shape,
        const int64_t out_col_size) {
    auto out_shape = values_shape;
    out_shape.erase(out_shape.begin());
    out_shape.insert(out_shape.begin(),
                     {row_splits_shape[0] - 1, out_col_size});
    return {out_shape};
}

PD_BUILD_OP(open3d_ragged_to_dense)
        .Inputs({"values", "row_splits", "default_value"})
        .Attrs({"out_col_size: int64_t"})
        .Outputs({"out"})
        .SetKernelFn(PD_KERNEL(RaggedToDense))
        .SetInferShapeFn(PD_INFER_SHAPE(RaggedToDenseInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(RaggedToDenseInferDtype));
