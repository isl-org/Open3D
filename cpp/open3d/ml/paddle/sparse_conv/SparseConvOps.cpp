// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/InvertNeighborsListOps.h"
#include "open3d/ml/paddle/misc/ReduceSubarraysSumOps.h"
#include "open3d/ml/paddle/sparse_conv/SparseConvBackpropFilterOpKernel.h"
#include "open3d/ml/paddle/sparse_conv/SparseConvOpKernel.h"
#include "open3d/ml/paddle/sparse_conv/SparseConvTransposeOpKernel.h"

std::vector<paddle::Tensor> SparseConvForward(
        paddle::Tensor& filters,
        paddle::Tensor& inp_features,
        paddle::Tensor& inp_importance,
        paddle::Tensor& neighbors_index,
        paddle::Tensor& neighbors_kernel_index,
        paddle::Tensor& neighbors_importance,
        paddle::Tensor& neighbors_row_splits,
        const bool normalize,
        const int64_t max_temp_mem_MB) {
    CHECK_TYPE(neighbors_row_splits, paddle::DataType::INT64);
    CHECK_SAME_DTYPE(filters, inp_features, inp_importance,
                     neighbors_importance);
    CHECK_SAME_DEVICE_TYPE(filters, inp_features, inp_importance);

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_kernel_elements("num_kernel_elements");
    Dim in_channels("in_channels");
    Dim out_channels("out_channels");
    Dim num_out_points("num_out_points");
    Dim num_inp_points("num_inp_points");
    Dim num_neighbors("nun_neighbors");

    CHECK_SHAPE_COMBINE_FIRST_DIMS(filters, num_kernel_elements, in_channels,
                                   out_channels);
    CHECK_SHAPE(inp_features, num_inp_points, in_channels);
    CHECK_SHAPE(inp_importance, num_inp_points || 0);
    CHECK_SHAPE(neighbors_index, num_neighbors);
    CHECK_SHAPE(neighbors_kernel_index, num_neighbors);
    CHECK_SHAPE(neighbors_importance, num_neighbors || 0);
    CHECK_SHAPE(neighbors_row_splits, num_out_points + 1);

    // make sure that these are on the same place as the filters and feats
    auto place = inp_features.place();
    neighbors_index = neighbors_index.copy_to(place, false);
    neighbors_kernel_index = neighbors_kernel_index.copy_to(place, false);
    neighbors_importance = neighbors_importance.copy_to(place, false);
    neighbors_row_splits = neighbors_row_splits.copy_to(place, false);

    const auto& feat_dtype = filters.dtype();
    const auto& index_dtype = neighbors_index.dtype();
    const auto& kernel_index_dtype = neighbors_kernel_index.dtype();

    paddle::Tensor out_features = paddle::empty(
            {num_out_points.value(), out_channels.value()}, feat_dtype, place);
#define FN_PARAMETERS                                       \
    filters, inp_features, inp_importance, neighbors_index, \
            neighbors_kernel_index, neighbors_importance,   \
            neighbors_row_splits, normalize, max_temp_mem_MB, out_features

#define CALL(feat_t, out_t, index_t, kernel_index_t, fn)           \
    if (ComparePaddleDtype<feat_t>(feat_dtype) &&                  \
        ComparePaddleDtype<index_t>(index_dtype) &&                \
        ComparePaddleDtype<kernel_index_t>(kernel_index_dtype)) {  \
        fn<feat_t, out_t, index_t, kernel_index_t>(FN_PARAMETERS); \
        return {out_features};                                     \
    }

    if (inp_features.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        CALL(float, float, int32_t, uint8_t, ::SparseConvCUDA)
#else
        PD_CHECK(false, "SparseConv was not compiled with CUDA support");
#endif
    } else {
        CALL(float, float, int32_t, uint8_t, ::SparseConvCPU)
    }
#undef FN_PARAMETERS
#undef CALL

    PD_CHECK(false,
             "SparseConv does not support " +
                     phi::DataTypeToString(inp_features.dtype()) +
                     " as input for inp_features, and " +
                     phi::DataTypeToString(neighbors_index.dtype()) +
                     " as input for neighbors_index, and " +
                     phi::DataTypeToString(neighbors_kernel_index.dtype()) +
                     " as input for neighbors_kernel_indexcgcgcc");
    return {paddle::Tensor()};
}

std::vector<paddle::Tensor> SparseConvBackward(
        paddle::Tensor& filters,
        paddle::Tensor& inp_features,
        paddle::Tensor& inp_importance,
        paddle::Tensor& neighbors_index,
        paddle::Tensor& neighbors_kernel_index,
        paddle::Tensor& neighbors_importance,
        paddle::Tensor& neighbors_row_splits,
        paddle::Tensor& out_features_gradient,
        const bool normalize,
        const int64_t max_temp_mem_MB) {
    auto place = inp_features.place();
    const auto& feat_dtype = filters.dtype();
    const auto& index_dtype = neighbors_index.dtype();
    const auto& kernel_index_dtype = neighbors_kernel_index.dtype();
    CHECK_SAME_DTYPE(out_features_gradient, inp_features, filters);
    CHECK_SAME_DEVICE_TYPE(out_features_gradient, inp_features, filters);

    // output vars
    paddle::Tensor filters_backprop;
    paddle::Tensor inp_features_backprop;

#define CALL(feat_t, out_t, index_t, kernel_index_t, fn_suffix)                \
    if (ComparePaddleDtype<feat_t>(feat_dtype) &&                              \
        ComparePaddleDtype<index_t>(index_dtype) &&                            \
        ComparePaddleDtype<kernel_index_t>(kernel_index_dtype)) {              \
        filters_backprop = paddle::empty(filters.shape(), feat_dtype, place);  \
        SparseConvBackpropFilter##fn_suffix<feat_t, out_t, index_t,            \
                                            kernel_index_t>(                   \
                filters, inp_features, inp_importance, neighbors_index,        \
                neighbors_kernel_index, neighbors_importance,                  \
                neighbors_row_splits, out_features_gradient, normalize,        \
                max_temp_mem_MB, filters_backprop);                            \
                                                                               \
        paddle::Tensor inv_neighbors_index, inv_neighbors_row_splits,          \
                inv_neighbors_importance, inv_arange;                          \
        paddle::Tensor arange = Arange(neighbors_index.shape()[0], place);     \
        auto inv = InvertNeighborsList(neighbors_index, neighbors_row_splits,  \
                                       arange, inp_features.shape()[0]);       \
        inv_neighbors_index = inv[0];                                          \
        inv_neighbors_row_splits = inv[1];                                     \
        inv_arange = inv[2];                                                   \
        paddle::Tensor inv_neighbors_kernel_index =                            \
                paddle::experimental::gather(neighbors_kernel_index,           \
                                             inv_arange);                      \
        if (neighbors_importance.shape()[0] > 0) {                             \
            inv_neighbors_importance = paddle::experimental::gather(           \
                    neighbors_importance, inv_arange);                         \
        } else {                                                               \
            inv_neighbors_importance = paddle::empty({0}, feat_dtype, place);  \
        }                                                                      \
                                                                               \
        auto neighbors_importance_sum = ReduceSubarraysSum(                    \
                neighbors_importance, neighbors_row_splits)[0];                \
        inp_features_backprop =                                                \
                paddle::ones(inp_features.shape(), feat_dtype, place);         \
        auto filters_transposed = Transpose(filters, -1, -2).contiguous();     \
                                                                               \
        SparseConvTranspose##fn_suffix<feat_t, out_t, index_t,                 \
                                       kernel_index_t>(                        \
                filters_transposed, inp_importance, out_features_gradient,     \
                neighbors_importance_sum, neighbors_row_splits,                \
                inv_neighbors_index, inv_neighbors_kernel_index,               \
                inv_neighbors_importance, inv_neighbors_row_splits, normalize, \
                max_temp_mem_MB, inp_features_backprop);                       \
        dispatch_success = true;                                               \
    }

    bool dispatch_success = false;
    if (inp_features.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        CALL(float, float, int32_t, uint8_t, CUDA)
#else
        PD_CHECK(false,
                 "SparseConv backward was not compiled "
                 "with CUDA support");
#endif
    } else {
        CALL(float, float, int32_t, uint8_t, CPU)
    }
    PD_CHECK(dispatch_success,
             "SparseConv backward does not support " +
                     phi::DataTypeToString(inp_features.dtype()) +
                     " as input for inp_features, and " +
                     phi::DataTypeToString(neighbors_index.dtype()) +
                     " as input for neighbors_index, and " +
                     phi::DataTypeToString(neighbors_kernel_index.dtype()) +
                     " as input for neighbors_kernel_index");

    return {filters_backprop, inp_features_backprop};
}

std::vector<paddle::DataType> SparseConvInferDtype(
        paddle::DataType filters_dtype) {
    return {filters_dtype};
}

PD_BUILD_OP(open3d_sparse_conv)
        .Inputs({"filters", "inp_features", "inp_importance", "neighbors_index",
                 "neighbors_kernel_index", "neighbors_importance",
                 "neighbors_row_splits"})
        .Outputs({"out_features"})
        .Attrs({"normalize:bool", "max_temp_mem_MB:int64_t"})
        .SetKernelFn(PD_KERNEL(SparseConvForward))
        .SetInferDtypeFn(PD_INFER_DTYPE(SparseConvInferDtype));

PD_BUILD_GRAD_OP(open3d_sparse_conv)
        .Inputs({"filters", "inp_features", "inp_importance", "neighbors_index",
                 "neighbors_kernel_index", "neighbors_importance",
                 "neighbors_row_splits", paddle::Grad("out_features")})
        .Outputs({paddle::Grad("filters"), paddle::Grad("inp_features")})
        .Attrs({"normalize:bool", "max_temp_mem_MB:int64_t"})
        .SetKernelFn(PD_KERNEL(SparseConvBackward));