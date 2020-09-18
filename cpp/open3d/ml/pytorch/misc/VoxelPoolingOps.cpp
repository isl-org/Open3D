// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/impl/misc/VoxelPooling.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

using namespace open3d::ml::impl;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using torch::autograd::variable_list;

template <class TReal, class TFeat>
std::tuple<torch::Tensor, torch::Tensor> VoxelPoolingCPU(
        const torch::Tensor& positions,
        const torch::Tensor& features,
        const double voxel_size,
        const AccumulationFn position_fn,
        const AccumulationFn feature_fn,
        const bool debug);

template <class TReal, class TFeat>
void VoxelPoolingGradCPU(torch::Tensor& features_backprop,
                         const torch::Tensor& positions,
                         const torch::Tensor& features,
                         const torch::Tensor& pooled_positions,
                         const torch::Tensor& pooled_features_gradient,
                         const double voxel_size,
                         const AccumulationFn position_fn,
                         const AccumulationFn feature_fn);

class VoxelPoolingFunction : public Function<VoxelPoolingFunction> {
public:
    static variable_list forward(AutogradContext* ctx,
                                 Variable positions,
                                 Variable features,
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
            TORCH_CHECK(false,
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
            TORCH_CHECK(false,
                        "feature_fn must be one of ('average', "
                        "'nearest_neighbor', 'max') but got " +
                                feature_fn_str);
        }
        positions = positions.contiguous();
        features = features.contiguous();

        // check input shapes
        {
            using namespace open3d::ml::op_util;
            Dim num_points("num_points");
            Dim num_channels("num_channels");

            CHECK_SHAPE(positions, num_points, 3);
            CHECK_SHAPE_COMBINE_LAST_DIMS(features, num_points, num_channels);
        }
        ctx->saved_data["position_fn_str"] = position_fn_str;
        ctx->saved_data["feature_fn_str"] = feature_fn_str;
        ctx->saved_data["voxel_size"] = voxel_size;

        const auto& positions_type = positions.dtype();
        const auto& features_type = features.dtype();

#define FN_PARAMETERS \
    positions, features, voxel_size, position_fn, feature_fn, debug

#define CALL(real_t, feat_t, fn)                                         \
    if (CompareTorchDtype<real_t>(positions_type) &&                     \
        CompareTorchDtype<feat_t>(features_type)) {                      \
        auto ans = fn<real_t, feat_t>(FN_PARAMETERS);                    \
        ctx->save_for_backward({positions, features, std::get<0>(ans)}); \
        return {std::get<0>(ans), std::get<1>(ans)};                     \
    }

        CHECK_SAME_DEVICE_TYPE(positions, features);
        if (positions.is_cuda()) {
            TORCH_CHECK(false, "VoxelPooling does not support CUDA")
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

        TORCH_CHECK(false,
                    "VoxelPooling does not support " + positions.toString() +
                            " as input for positions and " +
                            features.toString() + " as input for features")
        return {torch::Tensor(), torch::Tensor()};
    }

    static variable_list backward(AutogradContext* ctx,
                                  variable_list grad_output) {
        const std::string position_fn_str =
                ctx->saved_data["position_fn_str"].toStringRef();
        const std::string feature_fn_str =
                ctx->saved_data["feature_fn_str"].toStringRef();
        const double voxel_size = ctx->saved_data["voxel_size"].toDouble();
        AccumulationFn position_fn = AVERAGE;
        if (position_fn_str == "average") {
            position_fn = AVERAGE;
        } else if (position_fn_str == "nearest_neighbor") {
            position_fn = NEAREST_NEIGHBOR;
        } else if (position_fn_str == "center") {
            position_fn = CENTER;
        } else {
            TORCH_CHECK(false,
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
            TORCH_CHECK(false,
                        "feature_fn must be one of ('average', "
                        "'nearest_neighbor', 'max') but got " +
                                feature_fn_str);
        }

        auto saved_vars = ctx->get_saved_variables();
        auto positions = saved_vars[0];
        auto features = saved_vars[1];
        auto pooled_positions = saved_vars[2];
        auto pooled_features_gradient = grad_output[1].contiguous();
        positions = positions.contiguous();
        features = features.contiguous();
        pooled_positions = pooled_positions.contiguous();

        torch::Tensor features_backprop =
                torch::empty(features.sizes(), features.dtype());

        const auto& positions_type = positions.dtype();
        const auto& features_type = features.dtype();

#define FN_PARAMETERS                                         \
    features_backprop, positions, features, pooled_positions, \
            pooled_features_gradient, voxel_size, position_fn, feature_fn

#define CALL(real_t, feat_t, fn)                     \
    if (CompareTorchDtype<real_t>(positions_type) && \
        CompareTorchDtype<feat_t>(features_type)) {  \
        fn<real_t, feat_t>(FN_PARAMETERS);           \
        dispatch_success = true;                     \
    }

        CHECK_SAME_DEVICE_TYPE(positions, features);
        if (positions.is_cuda()) {
            TORCH_CHECK(false, "VoxelPooling backward does not support CUDA")
        } else {
            bool dispatch_success = false;
            CALL(float, float, VoxelPoolingGradCPU)
            CALL(float, double, VoxelPoolingGradCPU)
            CALL(double, float, VoxelPoolingGradCPU)
            CALL(double, double, VoxelPoolingGradCPU)
            TORCH_CHECK(dispatch_success,
                        "VoxelPooling backward does not support " +
                                positions.toString() +
                                " as input for positions and " +
                                features.toString() + " as input for features")
        }
#undef FN_PARAMETERS
#undef CALL

        return {Variable(), features_backprop, Variable(),
                Variable(), Variable(),        Variable()};
    }
};
std::tuple<torch::Tensor, torch::Tensor> VoxelPooling(
        const torch::Tensor& positions,
        const torch::Tensor& features,
        const double voxel_size,
        const std::string& position_fn_str,
        const std::string& feature_fn_str,
        const bool debug) {
    auto ans =
            VoxelPoolingFunction::apply(positions, features, voxel_size,
                                        position_fn_str, feature_fn_str, debug);
    return std::make_tuple(ans[0], ans[1]);
}

static auto registry = torch::RegisterOperators(
        "open3d::voxel_pooling(Tensor positions, Tensor features, float "
        "voxel_size, str position_fn=\"average\", str feature_fn=\"average\", "
        "bool debug=False) -> "
        "(Tensor pooled_positions, Tensor pooled_features)",
        &::VoxelPooling);
