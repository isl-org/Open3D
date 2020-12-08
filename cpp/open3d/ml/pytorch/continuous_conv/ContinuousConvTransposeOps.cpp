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

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/continuous_conv/ContinuousConvHelper.h"
#include "open3d/ml/pytorch/continuous_conv/ContinuousConvOpKernel.h"
#include "open3d/ml/pytorch/continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.h"
#include "open3d/ml/pytorch/continuous_conv/ContinuousConvTransposeOpKernel.h"
#include "open3d/ml/pytorch/misc/InvertNeighborsListOps.h"
#include "open3d/ml/pytorch/misc/ReduceSubarraysSumOps.h"
#include "torch/script.h"

using namespace open3d::ml::impl;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class ContinuousConvTransposeFunction
    : public Function<ContinuousConvTransposeFunction> {
public:
    static Variable forward(AutogradContext* ctx,
                            Variable filters,
                            Variable out_positions,
                            Variable out_importance,
                            Variable extents,
                            Variable offset,
                            Variable inp_positions,
                            Variable inp_features,
                            Variable inp_neighbors_index,
                            Variable inp_neighbors_importance_sum,
                            Variable inp_neighbors_row_splits,
                            Variable neighbors_index,
                            Variable neighbors_importance,
                            Variable neighbors_row_splits,
                            const bool align_corners,
                            const std::string& coordinate_mapping_str,
                            const bool normalize,
                            const std::string& interpolation_str,
                            const int64_t max_temp_mem_MB) {
        CoordinateMapping coordinate_mapping =
                ParseCoordinateMappingStr(coordinate_mapping_str);

        InterpolationMode interpolation =
                ParseInterpolationStr(interpolation_str);

        CHECK_TYPE(neighbors_row_splits, kInt64);
        CHECK_TYPE(inp_neighbors_row_splits, kInt64);
        CHECK_SAME_DTYPE(neighbors_index, inp_neighbors_index);
        CHECK_SAME_DTYPE(filters, out_positions, extents, offset, inp_positions,
                         inp_features, out_importance, neighbors_importance);
        CHECK_SAME_DEVICE_TYPE(filters, out_positions, inp_positions,
                               inp_features, out_importance);

        filters = filters.contiguous();
        out_positions = out_positions.contiguous();
        out_importance = out_importance.contiguous();
        extents = extents.contiguous();
        offset = offset.contiguous();
        inp_positions = inp_positions.contiguous();
        inp_features = inp_features.contiguous();
        inp_neighbors_index = inp_neighbors_index.contiguous();
        inp_neighbors_importance_sum =
                inp_neighbors_importance_sum.contiguous();
        inp_neighbors_row_splits = inp_neighbors_row_splits.contiguous();
        neighbors_index = neighbors_index.contiguous();
        neighbors_importance = neighbors_importance.contiguous();
        neighbors_row_splits = neighbors_row_splits.contiguous();

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

        CHECK_SHAPE(filters, kernel_depth, kernel_height, kernel_width,
                    in_channels, out_channels);
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

        // make sure that these are on the same device as the filters, positions
        // and feats
        auto device = inp_features.device();
        offset = offset.to(device);
        extents = extents.to(device);
        neighbors_index = neighbors_index.to(device);
        neighbors_importance = neighbors_importance.to(device);
        neighbors_row_splits = neighbors_row_splits.to(device);
        inp_neighbors_index = inp_neighbors_index.to(device);
        inp_neighbors_importance_sum = inp_neighbors_importance_sum.to(device);
        inp_neighbors_row_splits = inp_neighbors_row_splits.to(device);

        ctx->saved_data["align_corners"] = align_corners;
        ctx->saved_data["coordinate_mapping_str"] = coordinate_mapping_str;
        ctx->saved_data["normalize"] = normalize;
        ctx->saved_data["interpolation_str"] = interpolation_str;
        ctx->saved_data["max_temp_mem_MB"] = max_temp_mem_MB;

        ctx->save_for_backward({filters, out_positions, out_importance, extents,
                                offset, inp_positions, inp_features,
                                inp_neighbors_importance_sum,
                                inp_neighbors_row_splits, neighbors_index,
                                neighbors_importance, neighbors_row_splits});

        const auto& real_dtype = filters.dtype();
        const auto& index_dtype = neighbors_index.dtype();

        torch::Tensor out_features =
                torch::empty({num_out_points.value(), out_channels.value()},
                             torch::dtype(real_dtype).device(device));
#define FN_PARAMETERS                                                        \
    filters, out_positions, out_importance, extents, offset, inp_positions,  \
            inp_features, inp_neighbors_index, inp_neighbors_importance_sum, \
            inp_neighbors_row_splits, neighbors_index, neighbors_importance, \
            neighbors_row_splits, align_corners, coordinate_mapping,         \
            normalize, interpolation, max_temp_mem_MB, out_features

#define CALL(real_t, index_t, fn)                  \
    if (CompareTorchDtype<real_t>(real_dtype) &&   \
        CompareTorchDtype<index_t>(index_dtype)) { \
        fn<real_t, index_t>(FN_PARAMETERS);        \
        return out_features;                       \
    }

        if (inp_features.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
            CALL(float, int32_t, ::ContinuousConvTransposeCUDA)
#else
            TORCH_CHECK(false,
                        "ContinuousConvTranspose was not compiled with CUDA "
                        "support")
#endif
        } else {
            CALL(float, int32_t, ::ContinuousConvTransposeCPU)
        }
#undef FN_PARAMETERS
#undef CALL

        TORCH_CHECK(false, "ContinuousConv does not support " +
                                   inp_features.toString() +
                                   " as input for inp_features and " +
                                   neighbors_index.toString() +
                                   " as input for neighbors_index")
        return torch::Tensor();
    }

    static variable_list backward(AutogradContext* ctx,
                                  variable_list grad_output) {
        const bool align_corners = ctx->saved_data["align_corners"].toBool();
        const std::string coordinate_mapping_str =
                ctx->saved_data["coordinate_mapping_str"].toStringRef();
        const bool normalize = ctx->saved_data["normalize"].toBool();
        const std::string interpolation_str =
                ctx->saved_data["interpolation_str"].toStringRef();
        const int64_t max_temp_mem_MB =
                ctx->saved_data["max_temp_mem_MB"].toInt();

        CoordinateMapping coordinate_mapping =
                ParseCoordinateMappingStr(coordinate_mapping_str);

        InterpolationMode interpolation =
                ParseInterpolationStr(interpolation_str);

        auto saved_vars = ctx->get_saved_variables();
        auto filters = saved_vars[0];
        auto out_positions = saved_vars[1];
        auto out_importance = saved_vars[2];
        auto extents = saved_vars[3];
        auto offset = saved_vars[4];
        auto inp_positions = saved_vars[5];
        auto inp_features = saved_vars[6];
        auto inp_neighbors_importance_sum = saved_vars[7];
        auto inp_neighbors_row_splits = saved_vars[8];
        auto neighbors_index = saved_vars[9];
        auto neighbors_importance = saved_vars[10];
        auto neighbors_row_splits = saved_vars[11];

        auto device = inp_features.device();
        const auto& real_dtype = filters.dtype();
        const auto& index_dtype = neighbors_index.dtype();
        auto out_features_gradient = grad_output[0].contiguous();
        CHECK_SAME_DTYPE(out_features_gradient, inp_features, filters);
        CHECK_SAME_DEVICE_TYPE(out_features_gradient, inp_features, filters);

        // output vars
        torch::Tensor filters_backprop;
        torch::Tensor inp_features_backprop;

#define CALL(real_t, index_t, fn_suffix)                                      \
    if (CompareTorchDtype<real_t>(real_dtype) &&                              \
        CompareTorchDtype<index_t>(index_dtype)) {                            \
        filters_backprop = torch::empty(                                      \
                filters.sizes(), torch::dtype(real_dtype).device(device));    \
        ContinuousConvTransposeBackpropFilter##fn_suffix<real_t, index_t>(    \
                filters, out_positions, out_importance, extents, offset,      \
                inp_positions, inp_features, inp_neighbors_importance_sum,    \
                inp_neighbors_row_splits, neighbors_index,                    \
                neighbors_importance, neighbors_row_splits,                   \
                out_features_gradient, align_corners, coordinate_mapping,     \
                normalize, interpolation, max_temp_mem_MB, filters_backprop); \
                                                                              \
        torch::Tensor inv_neighbors_index, _inv_neighbors_row_splits,         \
                inv_neighbors_importance;                                     \
        std::tie(inv_neighbors_index, _inv_neighbors_row_splits,              \
                 inv_neighbors_importance) =                                  \
                InvertNeighborsList(inp_positions.size(0), neighbors_index,   \
                                    neighbors_row_splits,                     \
                                    neighbors_importance);                    \
        inp_features_backprop =                                               \
                torch::ones(inp_features.sizes(),                             \
                            torch::dtype(real_dtype).device(device));         \
        auto filters_transposed = filters.transpose(3, 4).contiguous();       \
                                                                              \
        ContinuousConv##fn_suffix<real_t, index_t>(                           \
                filters_transposed, inp_positions, extents, offset,           \
                out_positions, out_features_gradient, out_importance,         \
                inv_neighbors_index, inv_neighbors_importance,                \
                inp_neighbors_row_splits, align_corners, coordinate_mapping,  \
                normalize, interpolation, max_temp_mem_MB,                    \
                inp_features_backprop);                                       \
        dispatch_success = true;                                              \
    }

        bool dispatch_success = false;
        if (inp_features.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
            CALL(float, int32_t, CUDA)
#else
            TORCH_CHECK(false,
                        "ContinuousConvTranspose backward was not compiled "
                        "with CUDA support")
#endif
        } else {
            CALL(float, int32_t, CPU)
        }
        TORCH_CHECK(dispatch_success,
                    "ContinuousConvTranspose backward does not support " +
                            inp_features.toString() +
                            " as input for inp_features and " +
                            neighbors_index.toString() +
                            " as input for neighbors_index")

        return {filters_backprop,
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                inp_features_backprop,
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable()};
    }
};
torch::Tensor ContinuousConvTranspose(
        const torch::Tensor& filters,
        const torch::Tensor& out_positions,
        const torch::Tensor& out_importance,
        const torch::Tensor& extents,
        const torch::Tensor& offset,
        const torch::Tensor& inp_positions,
        const torch::Tensor& inp_features,
        const torch::Tensor& inp_neighbors_index,
        const torch::Tensor& inp_neighbors_importance_sum,
        const torch::Tensor& inp_neighbors_row_splits,
        const torch::Tensor& neighbors_index,
        const torch::Tensor& neighbors_importance,
        const torch::Tensor& neighbors_row_splits,
        const bool align_corners,
        const std::string& coordinate_mapping_str,
        const bool normalize,
        const std::string& interpolation_str,
        const int64_t max_temp_mem_MB) {
    auto ans = ContinuousConvTransposeFunction::apply(
            filters, out_positions, out_importance, extents, offset,
            inp_positions, inp_features, inp_neighbors_index,
            inp_neighbors_importance_sum, inp_neighbors_row_splits,
            neighbors_index, neighbors_importance, neighbors_row_splits,
            align_corners, coordinate_mapping_str, normalize, interpolation_str,
            max_temp_mem_MB);
    return ans;
}

static auto registry = torch::RegisterOperators(
        "open3d::continuous_conv_transpose(Tensor filters, Tensor "
        "out_positions, Tensor out_importance, Tensor extents, Tensor offset, "
        "Tensor inp_positions, Tensor inp_features, Tensor "
        "inp_neighbors_index, Tensor inp_neighbors_importance_sum, Tensor "
        "inp_neighbors_row_splits, Tensor neighbors_index, Tensor "
        "neighbors_importance, Tensor neighbors_row_splits, bool "
        "align_corners=False, str coordinate_mapping=\"ball_to_cube_radial\", "
        "bool normalize=False, str interpolation=\"linear\", int "
        "max_temp_mem_MB=64) -> Tensor",
        &::ContinuousConvTranspose);
