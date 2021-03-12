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

#define EIGEN_USE_GPU
#include "VoxelizeOpKernel.h"
#include "open3d/ml/Helper.h"
#include "open3d/ml/impl/misc/Voxelize.cuh"

using namespace open3d::ml;
using namespace open3d::ml::impl;
using namespace voxelize_opkernel;
using namespace tensorflow;

template <class T>
class VoxelizeOpKernelCUDA : public VoxelizeOpKernel {
public:
    explicit VoxelizeOpKernelCUDA(OpKernelConstruction* construction)
        : VoxelizeOpKernel(construction) {
        texture_alignment = GetCUDACurrentDeviceTextureAlignment();
    }

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& points,
                const tensorflow::Tensor& voxel_size,
                const tensorflow::Tensor& points_range_min,
                const tensorflow::Tensor& points_range_max) {
        auto device = context->eigen_gpu_device();

        OutputAllocator output_allocator(context);

        switch (points.dim_size(1)) {
#define CASE(NDIM)                                                          \
    case NDIM: {                                                            \
        void* temp_ptr = nullptr;                                           \
        size_t temp_size = 0;                                               \
        VoxelizeCUDA<T, NDIM>(                                              \
                device.stream(), temp_ptr, temp_size, texture_alignment,    \
                points.dim_size(0), points.flat<T>().data(),                \
                voxel_size.flat<T>().data(),                                \
                points_range_min.flat<T>().data(),                          \
                points_range_max.flat<T>().data(), max_points_per_voxel,    \
                max_voxels, output_allocator);                              \
                                                                            \
        Tensor temp_tensor;                                                 \
        TensorShape temp_shape({ssize_t(temp_size)});                       \
        OP_REQUIRES_OK(context,                                             \
                       context->allocate_temp(DataTypeToEnum<uint8_t>::v(), \
                                              temp_shape, &temp_tensor));   \
        temp_ptr = temp_tensor.flat<uint8_t>().data();                      \
                                                                            \
        VoxelizeCUDA<T, NDIM>(                                              \
                device.stream(), temp_ptr, temp_size, texture_alignment,    \
                points.dim_size(0), points.flat<T>().data(),                \
                voxel_size.flat<T>().data(),                                \
                points_range_min.flat<T>().data(),                          \
                points_range_max.flat<T>().data(), max_points_per_voxel,    \
                max_voxels, output_allocator);                              \
    } break;
            CASE(1)
            CASE(2)
            CASE(3)
            CASE(4)
            CASE(5)
            CASE(6)
            CASE(7)
            CASE(8)
            default:
                break;  // will be handled by the base class

#undef CASE
        }
    }

private:
    int texture_alignment;
};

#define REG_KB(type)                                                 \
    REGISTER_KERNEL_BUILDER(Name("Open3DVoxelize")                   \
                                    .Device(DEVICE_GPU)              \
                                    .TypeConstraint<type>("T")       \
                                    .HostMemory("voxel_size")        \
                                    .HostMemory("points_range_min")  \
                                    .HostMemory("points_range_max"), \
                            VoxelizeOpKernelCUDA<type>);
REG_KB(float)
REG_KB(double)
#undef REG_KB
