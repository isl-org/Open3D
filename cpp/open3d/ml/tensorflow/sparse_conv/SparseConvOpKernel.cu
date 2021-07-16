// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#define EIGEN_USE_GPU
#include "SparseConvOpKernel.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/ml/impl/sparse_conv/SparseConv.cuh"

using namespace open3d::ml;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
class SparseConvOpKernelCUDA : public SparseConvOpKernel<TIndex> {
public:
    explicit SparseConvOpKernelCUDA(OpKernelConstruction* construction)
        : SparseConvOpKernel<TIndex>(construction) {
        texture_alignment =
                open3d::core::GetCUDACurrentDeviceTextureAlignment();
    }

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& filter,
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
        auto device = context->eigen_gpu_device();

        void* temp_ptr = nullptr;
        size_t temp_size = 0;
        size_t max_temp_size = 0;

        // determine temp_size
        SparseConvComputeFeaturesCUDA<TFeat, TOut, TIndex, TKernelIndex>(
                device.stream(), temp_ptr, temp_size, max_temp_size,
                texture_alignment, out_features.flat<TOut>().data(),
                filter_dims, filter.flat<TFeat>().data(),
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
        SparseConvComputeFeaturesCUDA<TFeat, TOut, TIndex, TKernelIndex>(
                device.stream(), temp_ptr, temp_size, max_temp_size,
                texture_alignment, out_features.flat<TOut>().data(),
                filter_dims, filter.flat<TFeat>().data(),
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

private:
    int texture_alignment;
};

#define REG_KB(feattype, outtype, indextype, kernelindextype)         \
    REGISTER_KERNEL_BUILDER(                                          \
            Name("Open3DSparseConv")                                  \
                    .Device(DEVICE_GPU)                               \
                    .TypeConstraint<feattype>("TFeat")                \
                    .TypeConstraint<outtype>("output_type")           \
                    .TypeConstraint<indextype>("TIndex")              \
                    .TypeConstraint<kernelindextype>("TKernelIndex"), \
            SparseConvOpKernelCUDA<feattype, outtype, indextype,      \
                                   kernelindextype>);
REG_KB(float, float, int32, int16)
REG_KB(float, float, int32, uint8_t)
#undef REG_KB
