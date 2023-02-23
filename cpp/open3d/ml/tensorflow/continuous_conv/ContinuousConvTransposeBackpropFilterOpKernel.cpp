// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ContinuousConvTransposeBackpropFilterOpKernel.h"

#include "open3d/ml/impl/continuous_conv/ContinuousConvTransposeBackpropFilter.h"

using namespace open3d;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TReal, class TIndex>
class ContinuousConvTransposeBackpropFilterOpKernelCPU
    : public ContinuousConvTransposeBackpropFilterOpKernel<TIndex> {
public:
    explicit ContinuousConvTransposeBackpropFilterOpKernelCPU(
            OpKernelConstruction* construction)
        : ContinuousConvTransposeBackpropFilterOpKernel<TIndex>(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& filter,
                const tensorflow::Tensor& out_positions,
                const tensorflow::Tensor& out_importance,
                const tensorflow::Tensor& extents,
                const tensorflow::Tensor& offset,
                const tensorflow::Tensor& inp_positions,
                const tensorflow::Tensor& inp_features,
                const tensorflow::Tensor& inp_neighbors_importance_sum,
                const tensorflow::Tensor& inp_neighbors_row_splits,
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
        CConvTransposeBackpropFilterCPU<TFeat, TOut, TReal, TIndex>(
                filter_backprop.flat<TOut>().data(), filter_dims,
                out_positions.shape().dim_size(0),
                out_positions.flat<TReal>().data(),
                point_importances ? out_importance.flat<TFeat>().data()
                                  : nullptr,
                inp_positions.shape().dim_size(0),
                inp_positions.flat<TReal>().data(),
                inp_features.flat<TFeat>().data(),
                has_neighbors_importances
                        ? inp_neighbors_importance_sum.flat<TFeat>().data()
                        : nullptr,
                (int64_t*)inp_neighbors_row_splits.flat<int64>().data(),
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

#define REG_KB(feattype, outtype, realtype, indextype)          \
    REGISTER_KERNEL_BUILDER(                                    \
            Name("Open3DContinuousConvTransposeBackpropFilter") \
                    .Device(DEVICE_CPU)                         \
                    .TypeConstraint<feattype>("TFeat")          \
                    .TypeConstraint<outtype>("output_type")     \
                    .TypeConstraint<realtype>("TReal")          \
                    .TypeConstraint<indextype>("TIndex"),       \
            ContinuousConvTransposeBackpropFilterOpKernelCPU<   \
                    feattype, outtype, realtype, indextype>);
REG_KB(float, float, float, int32)
REG_KB(bfloat16, float, float, int32)
REG_KB(bfloat16, bfloat16, float, int32)
#undef REG_KB
