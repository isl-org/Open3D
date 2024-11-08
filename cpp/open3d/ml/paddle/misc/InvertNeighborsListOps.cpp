// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/paddle/misc/InvertNeighborsListOps.h"

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/InvertNeighborsListOpKernel.h"

std::vector<paddle::Tensor> InvertNeighborsList(
        paddle::Tensor& inp_neighbors_index,
        paddle::Tensor& inp_neighbors_row_splits,
        paddle::Tensor& inp_neighbors_attributes,
        int64_t num_points) {
    CHECK_TYPE(inp_neighbors_row_splits, paddle::DataType::INT64);

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim num_neighbors("num_neighbors");

        CHECK_SHAPE(inp_neighbors_index, num_neighbors);
        CHECK_SHAPE_IGNORE_LAST_DIMS(inp_neighbors_attributes,
                                     num_neighbors || 0);
        CHECK_SHAPE(inp_neighbors_row_splits, Dim());
    }

    const auto& index_type = inp_neighbors_index.dtype();
    const auto& attr_type = inp_neighbors_attributes.dtype();

#define FN_PARAMETERS                                          \
    num_points, inp_neighbors_index, inp_neighbors_row_splits, \
            inp_neighbors_attributes

#define CALL(idx_t, attr_t, fn)                  \
    if (ComparePaddleDtype<idx_t>(index_type) && \
        ComparePaddleDtype<attr_t>(attr_type)) { \
        return fn<idx_t, attr_t>(FN_PARAMETERS); \
    }

    CHECK_SAME_DEVICE_TYPE(inp_neighbors_index, inp_neighbors_row_splits,
                           inp_neighbors_attributes);
    if (inp_neighbors_index.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(int32_t, uint8_t, InvertNeighborsListCUDA)
        CALL(int32_t, int8_t, InvertNeighborsListCUDA)
        CALL(int32_t, int16_t, InvertNeighborsListCUDA)
        CALL(int32_t, int32_t, InvertNeighborsListCUDA)
        CALL(int32_t, int64_t, InvertNeighborsListCUDA)
        CALL(int32_t, float, InvertNeighborsListCUDA)
        CALL(int32_t, double, InvertNeighborsListCUDA)
#else
        PD_CHECK(false,
                 "InvertNeighborsList was not compiled with CUDA support");
#endif
    } else {
        CALL(int32_t, uint8_t, InvertNeighborsListCPU)
        CALL(int32_t, int8_t, InvertNeighborsListCPU)
        CALL(int32_t, int16_t, InvertNeighborsListCPU)
        CALL(int32_t, int32_t, InvertNeighborsListCPU)
        CALL(int32_t, int64_t, InvertNeighborsListCPU)
        CALL(int32_t, float, InvertNeighborsListCPU)
        CALL(int32_t, double, InvertNeighborsListCPU)
    }

    PD_CHECK(false,
             "InvertNeighborsList does not support " +
                     phi::DataTypeToString(inp_neighbors_index.dtype()) +
                     " as input for inp_neighbors_index and " +
                     phi::DataTypeToString(inp_neighbors_attributes.dtype()) +
                     " as input for inp_neighbors_attributes");
    return {};
}

std::vector<paddle::DataType> InvertNeighborsListInferDtype(
        const paddle::DataType inp_neighbors_attributes_dtype) {
    return {paddle::DataType::INT32, paddle::DataType::INT64,
            inp_neighbors_attributes_dtype};
}

std::vector<std::vector<int64_t>> InvertNeighborsListInferShape(
        int64_t num_points,
        std::vector<int64_t> inp_neighbors_index_shape,
        std::vector<int64_t> inp_neighbors_attributes_shape) {
    return {inp_neighbors_index_shape,
            {num_points + 1},
            inp_neighbors_attributes_shape};
}
PD_BUILD_OP(open3d_invert_neighbors_list)
        .Inputs({"inp_neighbors_index", "inp_neighbors_row_splits",
                 "inp_neighbors_attributes"})
        .Outputs({"neighbors_index", "neighbors_row_splits",
                  "neighbors_attributes"})
        .Attrs({"num_points: int64_t"})
        .SetKernelFn(PD_KERNEL(InvertNeighborsList))
        .SetInferShapeFn(PD_INFER_SHAPE(InvertNeighborsListInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(InvertNeighborsListInferDtype));
