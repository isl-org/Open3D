// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SparseConvOpKernel.h"

#include "open3d/ml/impl/sparse_conv/SparseConv.h"

using namespace open3d;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
class SparseConvOpKernelCPU : public SparseConvOpKernel<TIndex> {
public:
    explicit SparseConvOpKernelCPU(OpKernelConstruction* construction)
        : SparseConvOpKernel<TIndex>(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& filters,
                const tensorflow::Tensor& inp_features,
                const tensorflow::Tensor& inp_importance,
                const tensorflow::Tensor& neighbors_index,
                const tensorflow::Tensor& neighbors_kernel_index,
                const tensorflow::Tensor& neighbors_importance,
                const tensorflow::Tensor& neighbors_row_splits,
                const std::vector<int>& filter_dims,
                const bool point_importances,
                const bool has_neighbors_importances,
                tensorflow::Tensor& out_features) {
        SparseConvComputeFeaturesCPU<TFeat, TOut, TIndex, TKernelIndex>(
                out_features.flat<TOut>().data(), filter_dims,
                filters.flat<TFeat>().data(),
                neighbors_row_splits.dim_size(0) - 1, inp_features.dim_size(0),
                inp_features.flat<TFeat>().data(),
                point_importances ? inp_importance.flat<TFeat>().data()
                                  : nullptr,
                neighbors_index.shape().dim_size(0),
                (TIndex*)neighbors_index.flat<TIndex>().data(),
                (TKernelIndex*)neighbors_kernel_index.flat<TKernelIndex>()
                        .data(),
                has_neighbors_importances
                        ? neighbors_importance.flat<TFeat>().data()
                        : nullptr,
                (int64_t*)neighbors_row_splits.flat<int64>().data(),
                this->normalize);
    }
};

#define REG_KB(feattype, outtype, indextype, kernelindextype)         \
    REGISTER_KERNEL_BUILDER(                                          \
            Name("Open3DSparseConv")                                  \
                    .Device(DEVICE_CPU)                               \
                    .TypeConstraint<feattype>("TFeat")                \
                    .TypeConstraint<outtype>("output_type")           \
                    .TypeConstraint<indextype>("TIndex")              \
                    .TypeConstraint<kernelindextype>("TKernelIndex"), \
            SparseConvOpKernelCPU<feattype, outtype, indextype,       \
                                  kernelindextype>);
REG_KB(float, float, int32, int16)
REG_KB(float, float, int32, uint8_t)
REG_KB(bfloat16, float, int32, int16_t)
REG_KB(bfloat16, float, int32, uint8_t)
REG_KB(bfloat16, bfloat16, int32, int16_t)
REG_KB(bfloat16, bfloat16, int32, uint8_t)
#undef REG_KB
