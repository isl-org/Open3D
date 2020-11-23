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

#include "open3d/ml/tensorflow/misc/NmsOpKernel.h"

#include "open3d/ml/impl/misc/Nms.h"

using namespace nms_opkernel;
using namespace tensorflow;

class NmsOpKernelCPU : public NmsOpKernel {
public:
    explicit NmsOpKernelCPU(OpKernelConstruction* construction)
        : NmsOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& boxes,
                const tensorflow::Tensor& scores) {
        std::vector<int64_t> keep_indices = open3d::ml::impl::NmsCPUKernel(
                boxes.flat<float>().data(), scores.flat<float>().data(),
                boxes.dim_size(0), this->nms_overlap_thresh);

        OutputAllocator output_allocator(context);
        int64_t* ret_keep_indices = nullptr;
        output_allocator.AllocKeepIndices(&ret_keep_indices,
                                          keep_indices.size());
        memcpy(ret_keep_indices, keep_indices.data(),
               keep_indices.size() * sizeof(int64_t));
    }
};

#define REG_KB(type)                                                        \
    REGISTER_KERNEL_BUILDER(                                                \
            Name("Open3DNms").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
            NmsOpKernelCPU);
REG_KB(float)
#undef REG_KB
