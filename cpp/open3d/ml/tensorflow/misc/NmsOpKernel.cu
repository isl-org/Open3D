// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/Helper.h"
#include "open3d/ml/contrib/Nms.h"
#include "open3d/ml/tensorflow/misc/NmsOpKernel.h"

using namespace nms_opkernel;
using namespace tensorflow;

class NmsOpKernelCUDA : public NmsOpKernel {
public:
    explicit NmsOpKernelCUDA(OpKernelConstruction* construction)
        : NmsOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& boxes,
                const tensorflow::Tensor& scores) {
        // NmsCUDAKernel() writes into a caller-allocated device buffer of
        // capacity >= n and only returns the final `count` (the actual
        // output size is not known ahead of time). TF's output tensor size
        // must be fixed at allocation time, so first write into a scratch
        // device buffer sized for the worst case (n), then allocate the
        // real (exactly `count`-sized) output tensor and copy the valid
        // prefix into it.
        const int n = boxes.dim_size(0);
        int64_t* scratch_keep_indices = nullptr;
        OPEN3D_CUDA_CHECK(
                cudaMalloc(&scratch_keep_indices, n * sizeof(int64_t)));
        int count = open3d::ml::contrib::NmsCUDAKernel(
                boxes.flat<float>().data(), scores.flat<float>().data(), n,
                this->nms_overlap_thresh, scratch_keep_indices);

        OutputAllocator output_allocator(context);
        int64_t* ret_keep_indices = nullptr;
        output_allocator.AllocKeepIndices(&ret_keep_indices, count);
        OPEN3D_CUDA_CHECK(cudaMemcpy(ret_keep_indices, scratch_keep_indices,
                                     count * sizeof(int64_t),
                                     cudaMemcpyDeviceToDevice));
        OPEN3D_CUDA_CHECK(cudaFree(scratch_keep_indices));
    }
};

#define REG_KB(type)                                                        \
    REGISTER_KERNEL_BUILDER(                                                \
            Name("Open3DNms").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
            NmsOpKernelCUDA);
REG_KB(float)
#undef REG_KB
