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

#include "InvertNeighborsListOpKernel.h"

#include "open3d/ml/impl/misc/InvertNeighborsList.h"

using namespace open3d::ml::impl;
using namespace invert_neighbors_list_opkernel;
using namespace tensorflow;

template <class TIndex, class TAttr>
class InvertNeighborsListOpKernelCPU : public InvertNeighborsListOpKernel {
public:
    explicit InvertNeighborsListOpKernelCPU(OpKernelConstruction* construction)
        : InvertNeighborsListOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& inp_neighbors_index,
                const tensorflow::Tensor& inp_neighbors_row_splits,
                const tensorflow::Tensor& inp_neighbors_attributes,
                const int num_attributes,
                tensorflow::Tensor& neighbors_index,
                tensorflow::Tensor& neighbors_row_splits,
                tensorflow::Tensor& neighbors_attributes) {
        InvertNeighborsListCPU(
                inp_neighbors_index.flat<TIndex>().data(),
                num_attributes ? inp_neighbors_attributes.flat<TAttr>().data()
                               : nullptr,
                num_attributes,
                (int64_t*)inp_neighbors_row_splits.flat<int64>().data(),
                inp_neighbors_row_splits.shape().dim_size(0) - 1,
                neighbors_index.flat<TIndex>().data(),
                num_attributes ? neighbors_attributes.flat<TAttr>().data()
                               : nullptr,
                neighbors_index.shape().dim_size(0),
                (int64_t*)neighbors_row_splits.flat<int64>().data(),
                neighbors_row_splits.shape().dim_size(0) - 1);
    }
};

#define REG_KB(type, attrtype)                                          \
    REGISTER_KERNEL_BUILDER(Name("Open3DInvertNeighborsList")           \
                                    .Device(DEVICE_CPU)                 \
                                    .TypeConstraint<type>("TIndex")     \
                                    .TypeConstraint<attrtype>("TAttr"), \
                            InvertNeighborsListOpKernelCPU<type, attrtype>);
REG_KB(int32_t, uint8_t)
REG_KB(int32_t, int8_t)
REG_KB(int32_t, int16_t)
REG_KB(int32_t, int32_t)
REG_KB(int32_t, int64)
REG_KB(int32_t, float)
REG_KB(int32_t, double)
#undef REG_KB
