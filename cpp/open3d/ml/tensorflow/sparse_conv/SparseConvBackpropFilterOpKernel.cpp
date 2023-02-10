// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SparseConvBackpropFilterOpKernel.h"

#include "open3d/ml/impl/sparse_conv/SparseConvBackpropFilter.h"

using namespace open3d;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
class SparseConvBackpropFilterOpKernelCPU
    : public SparseConvBackpropFilterOpKernel<TIndex> {
public:
    explicit SparseConvBackpropFilterOpKernelCPU(
            OpKernelConstruction* construction)
        : SparseConvBackpropFilterOpKernel<TIndex>(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& filters,
                const tensorflow::Tensor& inp_features,
                const tensorflow::Tensor& inp_importance,
                const tensorflow::Tensor& neighbors_index,
                const tensorflow::Tensor& neighbors_kernel_index,
                const tensorflow::Tensor& neighbors_importance,
                const tensorflow::Tensor& neighbors_row_splits,
                const tensorflow::Tensor& out_features_gradient,
                const std::vector<int>& filter_dims,
                const bool point_importances,
                const bool has_neighbors_importances,
                tensorflow::Tensor& filter_backprop) {
        SparseConvBackpropFilterCPU<TFeat, TOut, TIndex>(
                filter_backprop.flat<TOut>().data(), filter_dims,
                neighbors_row_splits.shape().dim_size(0) - 1,
                inp_features.shape().dim_size(0),
                inp_features.flat<TFeat>().data(),
                point_importances ? inp_importance.flat<TFeat>().data()
                                  : nullptr,
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

#define REG_KB(feattype, outtype, indextype, kernelindextype)                 \
    REGISTER_KERNEL_BUILDER(                                                  \
            Name("Open3DSparseConvBackpropFilter")                            \
                    .Device(DEVICE_CPU)                                       \
                    .TypeConstraint<feattype>("TFeat")                        \
                    .TypeConstraint<outtype>("output_type")                   \
                    .TypeConstraint<indextype>("TIndex")                      \
                    .TypeConstraint<kernelindextype>("TKernelIndex"),         \
            SparseConvBackpropFilterOpKernelCPU<feattype, outtype, indextype, \
                                                kernelindextype>);
REG_KB(float, float, int32, int16_t)
REG_KB(float, float, int32, uint8_t)
REG_KB(bfloat16, float, int32, int16_t)
REG_KB(bfloat16, float, int32, uint8_t)
REG_KB(bfloat16, bfloat16, int32, int16_t)
REG_KB(bfloat16, bfloat16, int32, uint8_t)
#undef REG_KB
