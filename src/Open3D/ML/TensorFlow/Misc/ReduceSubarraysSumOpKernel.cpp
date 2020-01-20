// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "Open3D/ML/Misc/Detail/ReduceSubarraysSum.h"
#include "ReduceSubarraysSumOpKernel.h"

using namespace open3d::ml::detail;
using namespace reduce_subarrays_sum_opkernel;
using namespace tensorflow;

template <class T>
class ReduceSubarraysSumOpKernelCPU : public ReduceSubarraysSumOpKernel {
public:
    explicit ReduceSubarraysSumOpKernelCPU(OpKernelConstruction* construction)
        : ReduceSubarraysSumOpKernel(construction) {}

    void Kernel(OpKernelContext* context,
                const tensorflow::Tensor& values,
                const tensorflow::Tensor& prefix_sum,
                tensorflow::Tensor& sums) {
        ReduceSubarraysSumCPU(
                values.flat<T>().data(), values.shape().dim_size(0),
                (int64_t*)prefix_sum.flat<int64>().data(),
                prefix_sum.shape().dim_size(0), sums.flat<T>().data());
    }
};

#define REG_KB(type)                                            \
    REGISTER_KERNEL_BUILDER(Name("Open3DReduceSubarraysSum")    \
                                    .Device(DEVICE_CPU)         \
                                    .TypeConstraint<type>("T"), \
                            ReduceSubarraysSumOpKernelCPU<type>);
REG_KB(int32_t)
REG_KB(int64)
REG_KB(float)
REG_KB(double)
#undef REG_KB
