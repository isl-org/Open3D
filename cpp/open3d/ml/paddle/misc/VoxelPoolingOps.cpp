// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/impl/misc/VoxelPooling.h"
#include "open3d/ml/paddle/PaddleHelper.h"

using namespace open3d::ml::impl;

template <class TReal, class TFeat>
std::vector<paddle::Tensor> VoxelPoolingCPU(const paddle::Tensor& positions,
                                            const paddle::Tensor& features,
                                            const double voxel_size,
                                            const AccumulationFn position_fn,
                                            const AccumulationFn feature_fn,
                                            const bool debug);

template <class TReal, class TFeat>
void VoxelPoolingGradCPU(paddle::Tensor& features_backprop,
                         const paddle::Tensor& positions,
                         const paddle::Tensor& features,
                         const paddle::Tensor& pooled_positions,
                         const paddle::Tensor& pooled_features_gradient,
                         const double voxel_size,
                         const AccumulationFn position_fn,
                         const AccumulationFn feature_fn);

std::vector<paddle::Tensor> VoxelPoolingForward(
        paddle::Tensor& positions,
        paddle::Tensor& features,
        const double voxel_size,
        const std::string& position_fn_str,
        const std::string& feature_fn_str,
        const bool debug) {
    AccumulationFn position_fn = AVERAGE;
    if (position_fn_str == "average") {
        position_fn = AVERAGE;
    } else if (position_fn_str == "nearest_neighbor") {
        position_fn = NEAREST_NEIGHBOR;
    } else if (position_fn_str == "center") {
        position_fn = CENTER;
    } else {
        PD_CHECK(false,
                 "position_fn must be one of ('average', "
                 "'nearest_neighbor', 'center') but got " +
                         position_fn_str);
    }
    AccumulationFn feature_fn = AVERAGE;
    if (feature_fn_str == "average") {
        feature_fn = AVERAGE;
    } else if (feature_fn_str == "nearest_neighbor") {
        feature_fn = NEAREST_NEIGHBOR;
    } else if (feature_fn_str == "max") {
        feature_fn = MAX;
    } else {
        PD_CHECK(false,
                 "feature_fn must be one of ('average', "
                 "'nearest_neighbor', 'max') but got " +
                         feature_fn_str);
    }

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim num_points("num_points");
        Dim num_channels("num_channels");

        CHECK_SHAPE(positions, num_points, 3);
        CHECK_SHAPE_COMBINE_LAST_DIMS(features, num_points, num_channels);
    }

    // ctx->saved_data["position_fn_str"] = position_fn_str;
    // ctx->saved_data["feature_fn_str"] = feature_fn_str;
    // ctx->saved_data["voxel_size"] = voxel_size;

    const auto& positions_type = positions.dtype();
    const auto& features_type = features.dtype();

#define FN_PARAMETERS \
    positions, features, voxel_size, position_fn, feature_fn, debug

#define CALL(real_t, feat_t, fn)                      \
    if (ComparePaddleDtype<real_t>(positions_type) && \
        ComparePaddleDtype<feat_t>(features_type)) {  \
        return fn<real_t, feat_t>(FN_PARAMETERS);     \
    }

    CHECK_SAME_DEVICE_TYPE(positions, features);
    if (positions.is_gpu()) {
        PD_CHECK(false, "VoxelPooling does not support CUDA");
    } else {
        CALL(float, float, VoxelPoolingCPU)
        CALL(float, int32_t, VoxelPoolingCPU)
        CALL(float, int64_t, VoxelPoolingCPU)
        CALL(float, double, VoxelPoolingCPU)
        CALL(double, float, VoxelPoolingCPU)
        CALL(double, int32_t, VoxelPoolingCPU)
        CALL(double, int64_t, VoxelPoolingCPU)
        CALL(double, double, VoxelPoolingCPU)
    }
#undef FN_PARAMETERS
#undef CALL

    PD_CHECK(false, "VoxelPooling does not support " +
                            phi::DataTypeToString(positions.dtype()) +
                            " as input for positions and " +
                            phi::DataTypeToString(features.dtype()) +
                            " as input for features");
    return {paddle::Tensor(), paddle::Tensor()};
}

std::vector<paddle::Tensor> VoxelPoolingBackward(
        paddle::Tensor& positions,
        paddle::Tensor& features,
        paddle::Tensor& pooled_positions,
        paddle::Tensor& pooled_features_gradient,
        const double voxel_size,
        const std::string& position_fn_str,
        const std::string& feature_fn_str) {
    AccumulationFn position_fn = AVERAGE;
    if (position_fn_str == "average") {
        position_fn = AVERAGE;
    } else if (position_fn_str == "nearest_neighbor") {
        position_fn = NEAREST_NEIGHBOR;
    } else if (position_fn_str == "center") {
        position_fn = CENTER;
    } else {
        PD_CHECK(false,
                 "position_fn must be one of ('average', "
                 "'nearest_neighbor', 'center') but got " +
                         position_fn_str);
    }
    AccumulationFn feature_fn = AVERAGE;
    if (feature_fn_str == "average") {
        feature_fn = AVERAGE;
    } else if (feature_fn_str == "nearest_neighbor") {
        feature_fn = NEAREST_NEIGHBOR;
    } else if (feature_fn_str == "max") {
        feature_fn = MAX;
    } else {
        PD_CHECK(false,
                 "feature_fn must be one of ('average', "
                 "'nearest_neighbor', 'max') but got " +
                         feature_fn_str);
    }

    // auto pooled_positions = saved_vars[2];

    paddle::Tensor features_backprop =
            paddle::empty(features.shape(), features.dtype());

    const auto& positions_type = positions.dtype();
    const auto& features_type = features.dtype();

#define FN_PARAMETERS                                         \
    features_backprop, positions, features, pooled_positions, \
            pooled_features_gradient, voxel_size, position_fn, feature_fn

#define CALL(real_t, feat_t, fn)                      \
    if (ComparePaddleDtype<real_t>(positions_type) && \
        ComparePaddleDtype<feat_t>(features_type)) {  \
        fn<real_t, feat_t>(FN_PARAMETERS);            \
        return {features_backprop};                   \
    }

    CHECK_SAME_DEVICE_TYPE(positions, features);
    if (positions.is_gpu()) {
        PD_CHECK(false, "VoxelPooling backward does not support CUDA");
    } else {
        CALL(float, float, VoxelPoolingGradCPU)
        CALL(float, double, VoxelPoolingGradCPU)
        CALL(double, float, VoxelPoolingGradCPU)
        CALL(double, double, VoxelPoolingGradCPU)
        PD_CHECK(false, "VoxelPooling backward does not support " +
                                phi::DataTypeToString(positions.dtype()) +
                                " as input for positions and " +
                                phi::DataTypeToString(features.dtype()) +
                                " as input for features");
    }
#undef FN_PARAMETERS
#undef CALL

    return {};
}

std::vector<paddle::DataType> VoxelPoolingInferDtype(
        paddle::DataType positions_dtype, paddle::DataType features_dtype) {
    return {positions_dtype, features_dtype};
}

PD_BUILD_OP(open3d_voxel_pooling)
        .Inputs({"positions", "features"})
        .Outputs({"pooled_positions", "pooled_features"})
        .Attrs({"voxel_size:double", "position_fn:std::string",
                "feature_fn:std::string", "debug:bool"})
        .SetKernelFn(PD_KERNEL(VoxelPoolingForward))
        .SetInferDtypeFn(PD_INFER_DTYPE(VoxelPoolingInferDtype));

PD_BUILD_GRAD_OP(open3d_voxel_pooling)
        .Inputs({"positions", "features", "pooled_positions",
                 paddle::Grad("pooled_features")})
        .Outputs({paddle::Grad("features")})
        .Attrs({"voxel_size:double", "position_fn:std::string",
                "feature_fn:std::string"})
        .SetKernelFn(PD_KERNEL(VoxelPoolingBackward));
