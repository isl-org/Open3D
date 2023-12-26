// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ContinuousConvBackpropFilterOpKernel.h"

#include "open3d/ml/impl/continuous_conv/ContinuousConvBackpropFilter.h"

using namespace open3d;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TReal, class TIndex>
class ContinuousConvBackpropFilterOpKernelCPU
    : public ContinuousConvBackpropFilterOpKernel<TIndex> {
public:
    explicit ContinuousConvBackpropFilterOpKernelCPU(
            OpKernelConstruction* construction)
        : ContinuousConvBackpropFilterOpKernel<TIndex>(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& filter,
                const tensorflow::Tensor& out_positions,
                const tensorflow::Tensor& extents,
                const tensorflow::Tensor& offset,
                const tensorflow::Tensor& inp_positions,
                const tensorflow::Tensor& inp_features,
                const tensorflow::Tensor& inp_importance,
                const tensorflow::Tensor& neighbors_index,
                const tensorflow::Tensor& neighbors_importance,
                const tensorflow::Tensor& neighbors_row_splits,
                const tensorflow::Tensor& out_features_gradient,
                const std::vector<int>& filter_dims,
                const bool individual_extents,
                const bool isotropic_extents,
                const bool point_importances,
                const bool has_neighbors_importances,
                tensorflow::Tensor& filter_backprop) {
        CConvBackpropFilterCPU<TFeat, TOut, TReal, TIndex>(
                filter_backprop.flat<TOut>().data(), filter_dims,
                out_positions.shape().dim_size(0),
                out_positions.flat<TReal>().data(),
                inp_positions.shape().dim_size(0),
                inp_positions.flat<TReal>().data(),
                inp_features.flat<TFeat>().data(),
                point_importances ? inp_importance.flat<TFeat>().data()
                                  : nullptr,
                neighbors_index.shape().dim_size(0),
                (TIndex*)neighbors_index.flat<TIndex>().data(),
                has_neighbors_importances
                        ? neighbors_importance.flat<TFeat>().data()
                        : nullptr,
                (int64_t*)neighbors_row_splits.flat<int64>().data(),
                extents.flat<TReal>().data(), offset.flat<TReal>().data(),
                out_features_gradient.flat<TFeat>().data(), this->interpolation,
                this->coordinate_mapping, this->align_corners,
                individual_extents, isotropic_extents, this->normalize);
    }
};

#define REG_KB(feattype, outtype, realtype, indextype)                 \
    REGISTER_KERNEL_BUILDER(                                           \
            Name("Open3DContinuousConvBackpropFilter")                 \
                    .Device(DEVICE_CPU)                                \
                    .TypeConstraint<feattype>("TFeat")                 \
                    .TypeConstraint<outtype>("output_type")            \
                    .TypeConstraint<realtype>("TReal")                 \
                    .TypeConstraint<indextype>("TIndex"),              \
            ContinuousConvBackpropFilterOpKernelCPU<feattype, outtype, \
                                                    realtype, indextype>);
REG_KB(float, float, float, int32)
REG_KB(bfloat16, float, float, int32)
REG_KB(bfloat16, bfloat16, float, int32)
#undef REG_KB
