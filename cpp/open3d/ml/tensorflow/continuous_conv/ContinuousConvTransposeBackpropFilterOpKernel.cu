// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#define EIGEN_USE_GPU
#include "ContinuousConvTransposeBackpropFilterOpKernel.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/ml/impl/continuous_conv/ContinuousConvTransposeBackpropFilter.cuh"

using namespace open3d;
using namespace open3d::ml;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TReal, class TIndex>
class ContinuousConvTransposeBackpropFilterOpKernelCUDA
    : public ContinuousConvTransposeBackpropFilterOpKernel<TIndex> {
public:
    explicit ContinuousConvTransposeBackpropFilterOpKernelCUDA(
            OpKernelConstruction* construction)
        : ContinuousConvTransposeBackpropFilterOpKernel<TIndex>(construction) {
        texture_alignment =
                open3d::core::GetCUDACurrentDeviceTextureAlignment();
    }

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
        auto device = context->eigen_gpu_device();

        void* temp_ptr = nullptr;
        size_t temp_size = 0;
        size_t max_temp_size = 0;

        // determine temp_size
        CConvTransposeBackpropFilterCUDA<TFeat, TOut, TReal, TIndex>(
                device.stream(), temp_ptr, temp_size, max_temp_size,
                texture_alignment, filter_backprop.flat<TOut>().data(),
                filter_dims, out_positions.shape().dim_size(0),
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
                out_features_gradient.flat<TReal>().data(), this->interpolation,
                this->coordinate_mapping, this->align_corners,
                individual_extents, isotropic_extents, this->normalize);

        temp_size =
                std::max(std::min(size_t(this->max_temp_mem_MB) * 1024 * 1024,
                                  max_temp_size),
                         temp_size);

        Tensor temp_tensor;
        TensorShape temp_shape({ssize_t(temp_size)});
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<uint8_t>::v(),
                                              temp_shape, &temp_tensor));
        temp_ptr = temp_tensor.flat<uint8_t>().data();

        // actually run the operation
        CConvTransposeBackpropFilterCUDA<TFeat, TOut, TReal, TIndex>(
                device.stream(), temp_ptr, temp_size, max_temp_size,
                texture_alignment, filter_backprop.flat<TOut>().data(),
                filter_dims, out_positions.shape().dim_size(0),
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
                out_features_gradient.flat<TReal>().data(), this->interpolation,
                this->coordinate_mapping, this->align_corners,
                individual_extents, isotropic_extents, this->normalize);
    }

private:
    int texture_alignment;
};

#define REG_KB(feattype, outtype, realtype, indextype)          \
    REGISTER_KERNEL_BUILDER(                                    \
            Name("Open3DContinuousConvTransposeBackpropFilter") \
                    .Device(DEVICE_GPU)                         \
                    .TypeConstraint<feattype>("TFeat")          \
                    .TypeConstraint<outtype>("output_type")     \
                    .TypeConstraint<realtype>("TReal")          \
                    .TypeConstraint<indextype>("TIndex"),       \
            ContinuousConvTransposeBackpropFilterOpKernelCUDA<  \
                    feattype, outtype, realtype, indextype>);
REG_KB(float, float, float, int32)
#undef REG_KB
