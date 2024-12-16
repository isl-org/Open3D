// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/pytorch/misc/InvertNeighborsListOps.h"

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/InvertNeighborsListOpKernel.h"
#include "torch/script.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsList(
        int64_t num_points,
        torch::Tensor inp_neighbors_index,
        torch::Tensor inp_neighbors_row_splits,
        torch::Tensor inp_neighbors_attributes) {
    inp_neighbors_index = inp_neighbors_index.contiguous();
    inp_neighbors_row_splits = inp_neighbors_row_splits.contiguous();
    inp_neighbors_attributes = inp_neighbors_attributes.contiguous();
    CHECK_TYPE(inp_neighbors_row_splits, kInt64);

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
    if (CompareTorchDtype<idx_t>(index_type) &&  \
        CompareTorchDtype<attr_t>(attr_type)) {  \
        return fn<idx_t, attr_t>(FN_PARAMETERS); \
    }

    CHECK_SAME_DEVICE_TYPE(inp_neighbors_index, inp_neighbors_row_splits,
                           inp_neighbors_attributes);
    if (inp_neighbors_index.is_cuda()) {
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
        TORCH_CHECK(false,
                    "InvertNeighborsList was not compiled with CUDA support")
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

    TORCH_CHECK(false, "InvertNeighborsList does not support " +
                               inp_neighbors_index.toString() +
                               " as input for inp_neighbors_index and " +
                               inp_neighbors_attributes.toString() +
                               " as input for inp_neighbors_attributes")
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

static auto registry = torch::RegisterOperators(
        "open3d::invert_neighbors_list(int num_points, Tensor "
        "inp_neighbors_index, Tensor inp_neighbors_row_splits, Tensor "
        "inp_neighbors_attributes) -> (Tensor neighbors_index, Tensor "
        "neighbors_row_splits, Tensor neighbors_attributes)",
        &InvertNeighborsList);
