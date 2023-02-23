// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SparseConvTransposeBackpropFilterOpKernel.h"

#include "open3d/ml/impl/sparse_conv/SparseConvTransposeBackpropFilter.h"

using namespace open3d;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
class SparseConvTransposeBackpropFilterOpKernelCPU
    : public SparseConvTransposeBackpropFilterOpKernel<TIndex> {
public:
    explicit SparseConvTransposeBackpropFilterOpKernelCPU(
            OpKernelConstruction* construction)
        : SparseConvTransposeBackpropFilterOpKernel<TIndex>(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& filter,
                const tensorflow::Tensor& out_importance,
                const tensorflow::Tensor& inp_features,
                const tensorflow::Tensor& inp_neighbors_importance_sum,
                const tensorflow::Tensor& inp_neighbors_row_splits,
                const tensorflow::Tensor& neighbors_index,
                const tensorflow::Tensor& neighbors_kernel_index,
                const tensorflow::Tensor& neighbors_importance,
                const tensorflow::Tensor& neighbors_row_splits,
                const tensorflow::Tensor& out_features_gradient,
                const std::vector<int>& filter_dims,
                const bool point_importances,
                const bool has_neighbors_importances,
                tensorflow::Tensor& filter_backprop) {
        SparseConvTransposeBackpropFilterCPU<TFeat, TOut, TIndex, TKernelIndex>(
                filter_backprop.flat<TOut>().data(), filter_dims,
                neighbors_row_splits.shape().dim_size(0) - 1,
                point_importances ? out_importance.flat<TFeat>().data()
                                  : nullptr,
                inp_features.shape().dim_size(0),
                inp_features.flat<TFeat>().data(),
                has_neighbors_importances
                        ? inp_neighbors_importance_sum.flat<TFeat>().data()
                        : nullptr,
                (int64_t*)inp_neighbors_row_splits.flat<int64>().data(),
                (TIndex*)neighbors_index.flat<TIndex>().data(),
                (TKernelIndex*)neighbors_kernel_index.flat<TKernelIndex>()
                        .data(),
                has_neighbors_importances
                        ? neighbors_importance.flat<TFeat>().data()
                        : nullptr,
                (int64_t*)neighbors_row_splits.flat<int64>().data(),
                out_features_gradient.flat<TFeat>().data(), this->normalize);
    }
};

#define REG_KB(feattype, outtype, indextype, kernelindextype)         \
    REGISTER_KERNEL_BUILDER(                                          \
            Name("Open3DSparseConvTransposeBackpropFilter")           \
                    .Device(DEVICE_CPU)                               \
                    .TypeConstraint<feattype>("TFeat")                \
                    .TypeConstraint<outtype>("output_type")           \
                    .TypeConstraint<indextype>("TIndex")              \
                    .TypeConstraint<kernelindextype>("TKernelIndex"), \
            SparseConvTransposeBackpropFilterOpKernelCPU<             \
                    feattype, outtype, indextype, kernelindextype>);
REG_KB(float, float, int32, int16_t)
REG_KB(float, float, int32, uint8_t)
REG_KB(bfloat16, float, int32, int16_t)
REG_KB(bfloat16, float, int32, uint8_t)
REG_KB(bfloat16, bfloat16, int32, int16_t)
REG_KB(bfloat16, bfloat16, int32, uint8_t)
#undef REG_KB
