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

#include "SparseConvTransposeOpKernel.h"
#include "open3d/ml/impl/sparse_conv/SparseConvTranspose.h"

using namespace open3d;
using namespace open3d::ml::impl;
using namespace tensorflow;

template <class TReal, class TIndex, class TKernelIndex>
class SparseConvTransposeOpKernelCPU
    : public SparseConvTransposeOpKernel<TIndex> {
public:
    explicit SparseConvTransposeOpKernelCPU(OpKernelConstruction* construction)
        : SparseConvTransposeOpKernel<TIndex>(construction) {}

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
                const std::vector<int>& filter_dims,
                const bool point_importances,
                const bool has_neighbors_importances,
                tensorflow::Tensor& out_features) {
        SparseConvTransposeComputeFeaturesCPU<TReal, TIndex, TKernelIndex>(
                out_features.flat<TReal>().data(), filter_dims,
                filter.flat<TReal>().data(),
                neighbors_row_splits.shape().dim_size(0) - 1,
                point_importances ? out_importance.flat<TReal>().data()
                                  : nullptr,
                inp_features.shape().dim_size(0),
                inp_features.flat<TReal>().data(),
                has_neighbors_importances
                        ? inp_neighbors_importance_sum.flat<TReal>().data()
                        : nullptr,
                (int64_t*)inp_neighbors_row_splits.flat<int64>().data(),
                (TIndex*)neighbors_index.flat<TIndex>().data(),
                (TKernelIndex*)neighbors_kernel_index.flat<TKernelIndex>()
                        .data(),
                has_neighbors_importances
                        ? neighbors_importance.flat<TReal>().data()
                        : nullptr,
                (int64_t*)neighbors_row_splits.flat<int64>().data(),
                this->normalize);
    }
};

#define REG_KB(type, indextype, kernelindextype)                      \
    REGISTER_KERNEL_BUILDER(                                          \
            Name("Open3DSparseConvTranspose")                         \
                    .Device(DEVICE_CPU)                               \
                    .TypeConstraint<type>("TReal")                    \
                    .TypeConstraint<indextype>("TIndex")              \
                    .TypeConstraint<kernelindextype>("TKernelIndex"), \
            SparseConvTransposeOpKernelCPU<type, indextype, kernelindextype>);
REG_KB(float, int32, int16_t)
REG_KB(float, int32, uint8_t)
#undef REG_KB
