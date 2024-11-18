// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/continuous_conv/ContinuousConvHelper.h"
#include "open3d/ml/paddle/continuous_conv/ContinuousConvOpKernel.h"
#include "open3d/ml/paddle/continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.h"
#include "open3d/ml/paddle/continuous_conv/ContinuousConvTransposeOpKernel.h"
#include "open3d/ml/paddle/misc/InvertNeighborsListOps.h"
#include "open3d/ml/paddle/misc/ReduceSubarraysSumOps.h"

using namespace open3d::ml::impl;

std::vector<paddle::Tensor> ContinuousConvTransposeForward(
        paddle::Tensor& filters,
        paddle::Tensor& out_positions,
        paddle::Tensor& out_importance,
        paddle::Tensor& extents,
        paddle::Tensor& offset,
        paddle::Tensor& inp_positions,
        paddle::Tensor& inp_features,
        paddle::Tensor& inp_neighbors_index,
        paddle::Tensor& inp_neighbors_importance_sum,
        paddle::Tensor& inp_neighbors_row_splits,
        paddle::Tensor& neighbors_index,
        paddle::Tensor& neighbors_importance,
        paddle::Tensor& neighbors_row_splits,
        const bool align_corners,
        const std::string& coordinate_mapping_str,
        const bool normalize,
        const std::string& interpolation_str,
        const int64_t max_temp_mem_MB) {
    CoordinateMapping coordinate_mapping =
            ParseCoordinateMappingStr(coordinate_mapping_str);

    InterpolationMode interpolation = ParseInterpolationStr(interpolation_str);

    CHECK_TYPE(neighbors_row_splits, paddle::DataType::INT64);
    CHECK_TYPE(inp_neighbors_row_splits, paddle::DataType::INT64);
    CHECK_SAME_DTYPE(neighbors_index, inp_neighbors_index);
    CHECK_SAME_DTYPE(filters, out_positions, extents, offset, inp_positions,
                     inp_features, out_importance, neighbors_importance);
    CHECK_SAME_DEVICE_TYPE(filters, out_positions, inp_positions, inp_features,
                           out_importance);

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim kernel_depth("kernel_depth");
    Dim kernel_height("kernel_height");
    Dim kernel_width("kernel_width");
    Dim out_channels("out_channels");
    Dim in_channels("in_channels");
    Dim num_out_points("num_out_points");
    Dim num_inp_points("num_inp_points");
    Dim num_neighbors("nun_neighbors");

    CHECK_SHAPE(filters, kernel_depth, kernel_height, kernel_width, in_channels,
                out_channels);
    CHECK_SHAPE(out_positions, num_out_points, 3);
    CHECK_SHAPE(inp_positions, num_inp_points, 3);
    CHECK_SHAPE(extents, num_inp_points || 1, Dim(3) || 1);
    CHECK_SHAPE(offset, 3);
    CHECK_SHAPE(inp_features, num_inp_points, in_channels);
    CHECK_SHAPE(out_importance, num_out_points || 0);
    CHECK_SHAPE(inp_neighbors_index, num_neighbors);
    CHECK_SHAPE(inp_neighbors_importance_sum, num_inp_points || 0);
    CHECK_SHAPE(inp_neighbors_row_splits, num_inp_points + 1);
    CHECK_SHAPE(neighbors_index, num_neighbors);
    CHECK_SHAPE(neighbors_importance, num_neighbors || 0);
    CHECK_SHAPE(neighbors_row_splits, num_out_points + 1);

    // make sure that these are on the same place as the filters, positions
    // and feats
    auto place = inp_features.place();
    offset = offset.copy_to(place, false);
    extents = extents.copy_to(place, false);
    neighbors_index = neighbors_index.copy_to(place, false);
    neighbors_importance = neighbors_importance.copy_to(place, false);
    neighbors_row_splits = neighbors_row_splits.copy_to(place, false);
    inp_neighbors_index = inp_neighbors_index.copy_to(place, false);
    inp_neighbors_importance_sum =
            inp_neighbors_importance_sum.copy_to(place, false);
    inp_neighbors_row_splits = inp_neighbors_row_splits.copy_to(place, false);

    const auto& feat_dtype = filters.dtype();
    const auto& real_dtype = inp_positions.dtype();
    const auto& index_dtype = neighbors_index.dtype();

    paddle::Tensor out_features = paddle::empty(
            {num_out_points.value(), out_channels.value()}, real_dtype, place);
#define FN_PARAMETERS                                                        \
    filters, out_positions, out_importance, extents, offset, inp_positions,  \
            inp_features, inp_neighbors_index, inp_neighbors_importance_sum, \
            inp_neighbors_row_splits, neighbors_index, neighbors_importance, \
            neighbors_row_splits, align_corners, coordinate_mapping,         \
            normalize, interpolation, max_temp_mem_MB, out_features

#define CALL(feat_t, out_t, real_t, index_t, fn)           \
    if (ComparePaddleDtype<feat_t>(feat_dtype) &&          \
        ComparePaddleDtype<real_t>(real_dtype) &&          \
        ComparePaddleDtype<index_t>(index_dtype)) {        \
        fn<feat_t, out_t, real_t, index_t>(FN_PARAMETERS); \
        return {out_features};                             \
    }

    if (inp_features.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        CALL(float, float, float, int32_t, ::ContinuousConvTransposeCUDA)
#else
        PD_CHECK(false,
                 "ContinuousConvTranspose was not compiled with CUDA "
                 "support");
#endif
    } else {
        CALL(float, float, float, int32_t, ::ContinuousConvTransposeCPU)
    }
#undef FN_PARAMETERS
#undef CALL

    PD_CHECK(false, "ContinuousConv does not support " +
                            phi::DataTypeToString(inp_features.dtype()) +
                            " as input for inp_features and " +
                            phi::DataTypeToString(neighbors_index.dtype()) +
                            " as input for neighbors_index");
    return {paddle::Tensor()};
}

std::vector<paddle::Tensor> ContinuousConvTransposeBackward(
        paddle::Tensor& filters,
        paddle::Tensor& out_positions,
        paddle::Tensor& out_importance,
        paddle::Tensor& extents,
        paddle::Tensor& offset,
        paddle::Tensor& inp_positions,
        paddle::Tensor& inp_features,
        paddle::Tensor& inp_neighbors_importance_sum,
        paddle::Tensor& inp_neighbors_row_splits,
        paddle::Tensor& neighbors_index,
        paddle::Tensor& neighbors_importance,
        paddle::Tensor& neighbors_row_splits,
        paddle::Tensor& out_features_gradient,
        const bool align_corners,
        const std::string& coordinate_mapping_str,
        const bool normalize,
        const std::string& interpolation_str,
        const int64_t max_temp_mem_MB) {
    CoordinateMapping coordinate_mapping =
            ParseCoordinateMappingStr(coordinate_mapping_str);

    InterpolationMode interpolation = ParseInterpolationStr(interpolation_str);

    auto place = inp_features.place();
    const auto& feat_dtype = filters.dtype();
    const auto& real_dtype = inp_features.dtype();
    const auto& index_dtype = neighbors_index.dtype();
    CHECK_SAME_DTYPE(out_features_gradient, inp_features, filters);
    CHECK_SAME_DEVICE_TYPE(out_features_gradient, inp_features, filters);

    // output vars
    paddle::Tensor filters_backprop;
    paddle::Tensor inp_features_backprop;

#define CALL(feat_t, out_t, real_t, index_t, fn_suffix)                       \
    if (ComparePaddleDtype<feat_t>(feat_dtype) &&                             \
        ComparePaddleDtype<real_t>(real_dtype) &&                             \
        ComparePaddleDtype<index_t>(index_dtype)) {                           \
        filters_backprop = paddle::empty(filters.shape(), real_dtype, place); \
        ContinuousConvTransposeBackpropFilter##fn_suffix<feat_t, out_t,       \
                                                         real_t, index_t>(    \
                filters, out_positions, out_importance, extents, offset,      \
                inp_positions, inp_features, inp_neighbors_importance_sum,    \
                inp_neighbors_row_splits, neighbors_index,                    \
                neighbors_importance, neighbors_row_splits,                   \
                out_features_gradient, align_corners, coordinate_mapping,     \
                normalize, interpolation, max_temp_mem_MB, filters_backprop); \
                                                                              \
        paddle::Tensor inv_neighbors_index, inv_neighbors_row_splits,         \
                inv_neighbors_importance;                                     \
        auto inv = InvertNeighborsList(neighbors_index, neighbors_row_splits, \
                                       neighbors_importance,                  \
                                       inp_positions.shape()[0]);             \
        inv_neighbors_index = inv[0];                                         \
        inv_neighbors_row_splits = inv[1];                                    \
        inv_neighbors_importance = inv[2];                                    \
        InvertNeighborsList(neighbors_index, neighbors_row_splits,            \
                            neighbors_importance, inp_positions.shape()[0]);  \
        inp_features_backprop =                                               \
                paddle::ones(inp_features.shape(), real_dtype, place);        \
        auto filters_transposed = Transpose(filters, 3, 4).contiguous();      \
                                                                              \
        ContinuousConv##fn_suffix<feat_t, out_t, real_t, index_t>(            \
                filters_transposed, inp_positions, extents, offset,           \
                out_positions, out_features_gradient, out_importance,         \
                inv_neighbors_index, inv_neighbors_importance,                \
                inp_neighbors_row_splits, align_corners, coordinate_mapping,  \
                normalize, interpolation, max_temp_mem_MB,                    \
                inp_features_backprop);                                       \
        dispatch_success = true;                                              \
    }

    bool dispatch_success = false;
    if (inp_features.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        CALL(float, float, float, int32_t, CUDA)
#else
        PD_CHECK(false,
                 "ContinuousConvTranspose backward was not compiled "
                 "with CUDA support");
#endif
    } else {
        CALL(float, float, float, int32_t, CPU)
    }
    PD_CHECK(dispatch_success,
             "ContinuousConvTranspose backward does not support " +
                     phi::DataTypeToString(inp_features.dtype()) +
                     " as input for inp_features and " +
                     phi::DataTypeToString(neighbors_index.dtype()) +
                     " as input for neighbors_index");

    return {filters_backprop, inp_features_backprop};
}

std::vector<paddle::DataType> ContinuousConvTransposeInferDtype(
        paddle::DataType inp_positions_dtype) {
    return {inp_positions_dtype};
}

PD_BUILD_OP(open3d_continuous_conv_transpose)
        .Inputs({"filters", "out_positions", "out_importance", "extents",
                 "offset", "inp_positions", "inp_features",
                 "inp_neighbors_index", "inp_neighbors_importance_sum",
                 "inp_neighbors_row_splits", "neighbors_index",
                 "neighbors_importance", "neighbors_row_splits"})
        .Outputs({"out_features"})
        .Attrs({"align_corners:bool", "coordinate_mapping:std::string",
                "normalize:bool", "interpolation:std::string",
                "max_temp_mem_MB:int64_t"})
        .SetKernelFn(PD_KERNEL(ContinuousConvTransposeForward))
        .SetInferDtypeFn(PD_INFER_DTYPE(ContinuousConvTransposeInferDtype));

PD_BUILD_GRAD_OP(open3d_continuous_conv_transpose)
        .Inputs({"filters", "out_positions", "out_importance", "extents",
                 "offset", "inp_positions", "inp_features",
                 "inp_neighbors_importance_sum", "inp_neighbors_row_splits",
                 "neighbors_index", "neighbors_importance",
                 "neighbors_row_splits", paddle::Grad("out_features")})
        .Outputs({paddle::Grad("filters"), paddle::Grad("inp_features")})
        .Attrs({"align_corners:bool", "coordinate_mapping:std::string",
                "normalize:bool", "interpolation:std::string",
                "max_temp_mem_MB:int64_t"})
        .SetKernelFn(PD_KERNEL(ContinuousConvTransposeBackward));
