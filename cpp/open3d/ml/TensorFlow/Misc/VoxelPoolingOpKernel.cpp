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

#include "VoxelPoolingOpKernel.h"

#include "open3d/ml/impl/misc/VoxelPooling.h"

using namespace open3d::ml::impl;
using namespace voxel_pooling_opkernel;
using namespace tensorflow;

template <class TReal, class TFeat>
class VoxelPoolingOpKernelCPU : public VoxelPoolingOpKernel {
public:
    explicit VoxelPoolingOpKernelCPU(OpKernelConstruction* construction)
        : VoxelPoolingOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& positions,
                const tensorflow::Tensor& features,
                const tensorflow::Tensor& voxel_size) {
        OutputAllocator<TReal, TFeat> output_allocator(context);

        if (debug) {
            std::string err;
            OP_REQUIRES(context,
                        CheckVoxelSize(err, positions.shape().dim_size(0),
                                       positions.flat<TReal>().data(),
                                       voxel_size.scalar<TReal>()()),
                        errors::InvalidArgument(err));
        }

        VoxelPooling<TReal, TFeat>(
                positions.shape().dim_size(0), positions.flat<TReal>().data(),
                features.shape().dim_size(1), features.flat<TFeat>().data(),
                voxel_size.scalar<TReal>()(), output_allocator, position_fn,
                feature_fn);
    }
};

#define REG_KB(type, typefeat)                                          \
    REGISTER_KERNEL_BUILDER(Name("Open3DVoxelPooling")                  \
                                    .Device(DEVICE_CPU)                 \
                                    .TypeConstraint<type>("TReal")      \
                                    .TypeConstraint<typefeat>("TFeat"), \
                            VoxelPoolingOpKernelCPU<type, typefeat>);
REG_KB(float, float)
REG_KB(float, int)
REG_KB(float, int64)
REG_KB(float, double)
REG_KB(double, float)
REG_KB(double, int)
REG_KB(double, int64)
REG_KB(double, double)
#undef REG_KB
