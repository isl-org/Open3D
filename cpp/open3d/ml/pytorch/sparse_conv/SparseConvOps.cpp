// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <torch/script.h>

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/InvertNeighborsListOps.h"
#include "open3d/ml/pytorch/misc/ReduceSubarraysSumOps.h"
#include "open3d/ml/pytorch/sparse_conv/SparseConvBackpropFilterOpKernel.h"
#include "open3d/ml/pytorch/sparse_conv/SparseConvOpKernel.h"
#include "open3d/ml/pytorch/sparse_conv/SparseConvTransposeOpKernel.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SparseConvFunction : public Function<SparseConvFunction> {
public:
    static Variable forward(AutogradContext* ctx,
                            Variable filters,
                            Variable inp_features,
                            Variable inp_importance,
                            Variable neighbors_index,
                            Variable neighbors_kernel_index,
                            Variable neighbors_importance,
                            Variable neighbors_row_splits,
                            const bool normalize,
                            const int64_t max_temp_mem_MB) {
        CHECK_TYPE(neighbors_row_splits, kInt64);
        CHECK_SAME_DTYPE(filters, inp_features, inp_importance,
                         neighbors_importance);
        CHECK_SAME_DEVICE_TYPE(filters, inp_features, inp_importance);

        filters = filters.contiguous();
        inp_features = inp_features.contiguous();
        inp_importance = inp_importance.contiguous();
        neighbors_index = neighbors_index.contiguous();
        neighbors_kernel_index = neighbors_kernel_index.contiguous();
        neighbors_importance = neighbors_importance.contiguous();
        neighbors_row_splits = neighbors_row_splits.contiguous();

        // check input shapes
        using namespace open3d::ml::op_util;
        Dim num_kernel_elements("num_kernel_elements");
        Dim in_channels("in_channels");
        Dim out_channels("out_channels");
        Dim num_out_points("num_out_points");
        Dim num_inp_points("num_inp_points");
        Dim num_neighbors("nun_neighbors");

        CHECK_SHAPE_COMBINE_FIRST_DIMS(filters, num_kernel_elements,
                                       in_channels, out_channels);
        CHECK_SHAPE(inp_features, num_inp_points, in_channels);
        CHECK_SHAPE(inp_importance, num_inp_points || 0);
        CHECK_SHAPE(neighbors_index, num_neighbors);
        CHECK_SHAPE(neighbors_kernel_index, num_neighbors);
        CHECK_SHAPE(neighbors_importance, num_neighbors || 0);
        CHECK_SHAPE(neighbors_row_splits, num_out_points + 1);

        // make sure that these are on the same device as the filters and feats
        auto device = inp_features.device();
        neighbors_index = neighbors_index.to(device);
        neighbors_kernel_index = neighbors_kernel_index.to(device);
        neighbors_importance = neighbors_importance.to(device);
        neighbors_row_splits = neighbors_row_splits.to(device);

        ctx->saved_data["normalize"] = normalize;
        ctx->saved_data["max_temp_mem_MB"] = max_temp_mem_MB;

        ctx->save_for_backward({filters, inp_features, inp_importance,
                                neighbors_index, neighbors_kernel_index,
                                neighbors_importance, neighbors_row_splits});

        const auto& feat_dtype = filters.dtype();
        const auto& index_dtype = neighbors_index.dtype();
        const auto& kernel_index_dtype = neighbors_kernel_index.dtype();

        torch::Tensor out_features =
                torch::empty({num_out_points.value(), out_channels.value()},
                             torch::dtype(feat_dtype).device(device));
#define FN_PARAMETERS                                       \
    filters, inp_features, inp_importance, neighbors_index, \
            neighbors_kernel_index, neighbors_importance,   \
            neighbors_row_splits, normalize, max_temp_mem_MB, out_features

#define CALL(feat_t, out_t, index_t, kernel_index_t, fn)           \
    if (CompareTorchDtype<feat_t>(feat_dtype) &&                   \
        CompareTorchDtype<index_t>(index_dtype) &&                 \
        CompareTorchDtype<kernel_index_t>(kernel_index_dtype)) {   \
        fn<feat_t, out_t, index_t, kernel_index_t>(FN_PARAMETERS); \
        return out_features;                                       \
    }

        if (inp_features.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
            CALL(float, float, int32_t, uint8_t, ::SparseConvCUDA)
#else
            TORCH_CHECK(false, "SparseConv was not compiled with CUDA support")
#endif
        } else {
            CALL(float, float, int32_t, uint8_t, ::SparseConvCPU)
        }
#undef FN_PARAMETERS
#undef CALL

        TORCH_CHECK(false, "SparseConv does not support " +
                                   inp_features.toString() +
                                   " as input for inp_features, and " +
                                   neighbors_index.toString() +
                                   " as input for neighbors_index, and " +
                                   neighbors_kernel_index.toString() +
                                   " as input for neighbors_kernel_indexcgcgcc")
        return torch::Tensor();
    }

    static variable_list backward(AutogradContext* ctx,
                                  variable_list grad_output) {
        const bool normalize = ctx->saved_data["normalize"].toBool();
        const int64_t max_temp_mem_MB =
                ctx->saved_data["max_temp_mem_MB"].toInt();

        auto saved_vars = ctx->get_saved_variables();
        auto filters = saved_vars[0];
        auto inp_features = saved_vars[1];
        auto inp_importance = saved_vars[2];
        auto neighbors_index = saved_vars[3];
        auto neighbors_kernel_index = saved_vars[4];
        auto neighbors_importance = saved_vars[5];
        auto neighbors_row_splits = saved_vars[6];

        auto device = inp_features.device();
        const auto& feat_dtype = filters.dtype();
        const auto& index_dtype = neighbors_index.dtype();
        const auto& kernel_index_dtype = neighbors_kernel_index.dtype();
        auto out_features_gradient = grad_output[0].contiguous();
        CHECK_SAME_DTYPE(out_features_gradient, inp_features, filters);
        CHECK_SAME_DEVICE_TYPE(out_features_gradient, inp_features, filters);

        // output vars
        torch::Tensor filters_backprop;
        torch::Tensor inp_features_backprop;

#define CALL(feat_t, out_t, index_t, kernel_index_t, fn_suffix)                \
    if (CompareTorchDtype<feat_t>(feat_dtype) &&                               \
        CompareTorchDtype<index_t>(index_dtype) &&                             \
        CompareTorchDtype<kernel_index_t>(kernel_index_dtype)) {               \
        filters_backprop = torch::empty(                                       \
                filters.sizes(), torch::dtype(feat_dtype).device(device));     \
        SparseConvBackpropFilter##fn_suffix<feat_t, out_t, index_t,            \
                                            kernel_index_t>(                   \
                filters, inp_features, inp_importance, neighbors_index,        \
                neighbors_kernel_index, neighbors_importance,                  \
                neighbors_row_splits, out_features_gradient, normalize,        \
                max_temp_mem_MB, filters_backprop);                            \
                                                                               \
        torch::Tensor inv_neighbors_index, inv_neighbors_row_splits,           \
                inv_neighbors_importance, inv_arange;                          \
        torch::Tensor arange =                                                 \
                torch::arange(neighbors_index.size(0), torch::device(device)); \
        std::tie(inv_neighbors_index, inv_neighbors_row_splits, inv_arange) =  \
                InvertNeighborsList(inp_features.size(0), neighbors_index,     \
                                    neighbors_row_splits, arange);             \
        torch::Tensor inv_neighbors_kernel_index =                             \
                neighbors_kernel_index.index({inv_arange}).contiguous();       \
        if (neighbors_importance.size(0) > 0) {                                \
            inv_neighbors_importance =                                         \
                    neighbors_importance.index({inv_arange}).contiguous();     \
        } else {                                                               \
            inv_neighbors_importance = torch::empty(                           \
                    {0}, torch::dtype(feat_dtype).device(device));             \
        }                                                                      \
                                                                               \
        auto neighbors_importance_sum = ReduceSubarraysSum(                    \
                neighbors_importance, neighbors_row_splits);                   \
        inp_features_backprop =                                                \
                torch::ones(inp_features.sizes(),                              \
                            torch::dtype(feat_dtype).device(device));          \
        auto filters_transposed = filters.transpose(-2, -1).contiguous();      \
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
        if (inp_features.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
            CALL(float, float, int32_t, uint8_t, CUDA)
#else
            TORCH_CHECK(false,
                        "SparseConv backward was not compiled "
                        "with CUDA support")
#endif
        } else {
            CALL(float, float, int32_t, uint8_t, CPU)
        }
        TORCH_CHECK(dispatch_success,
                    "SparseConv backward does not support " +
                            inp_features.toString() +
                            " as input for inp_features, and " +
                            neighbors_index.toString() +
                            " as input for neighbors_index, and " +
                            neighbors_kernel_index.toString() +
                            " as input for neighbors_kernel_index")

        return {filters_backprop, inp_features_backprop,
                Variable(),       Variable(),
                Variable(),       Variable(),
                Variable(),       Variable(),
                Variable()};
    }
};
torch::Tensor SparseConv(const torch::Tensor& filters,
                         const torch::Tensor& inp_features,
                         const torch::Tensor& inp_importance,
                         const torch::Tensor& neighbors_index,
                         const torch::Tensor& neighbors_kernel_index,
                         const torch::Tensor& neighbors_importance,
                         const torch::Tensor& neighbors_row_splits,
                         const bool normalize,
                         const int64_t max_temp_mem_MB) {
    auto ans = SparseConvFunction::apply(
            filters, inp_features, inp_importance, neighbors_index,
            neighbors_kernel_index, neighbors_importance, neighbors_row_splits,
            normalize, max_temp_mem_MB);
    return ans;
}

static auto registry = torch::RegisterOperators(
        "open3d::sparse_conv(Tensor filters, Tensor inp_features, Tensor "
        "inp_importance, Tensor neighbors_index, Tensor "
        "neighbors_kernel_index, Tensor neighbors_importance, Tensor "
        "neighbors_row_splits, bool normalize=False, int max_temp_mem_MB=64) "
        "-> Tensor",
        &::SparseConv);
