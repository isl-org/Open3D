// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VoxelizeOpKernel.h"

#include "open3d/ml/impl/misc/Voxelize.h"

using namespace open3d::ml::impl;
using namespace voxelize_opkernel;
using namespace tensorflow;

template <class T>
class VoxelizeOpKernelCPU : public VoxelizeOpKernel {
public:
    explicit VoxelizeOpKernelCPU(OpKernelConstruction* construction)
        : VoxelizeOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& points,
                const tensorflow::Tensor& row_splits,
                const tensorflow::Tensor& voxel_size,
                const tensorflow::Tensor& points_range_min,
                const tensorflow::Tensor& points_range_max) {
        OutputAllocator output_allocator(context);

        switch (points.dim_size(1)) {
#define CASE(NDIM)                                                             \
    case NDIM:                                                                 \
        VoxelizeCPU<T, NDIM>(                                                  \
                points.dim_size(0), points.flat<T>().data(),                   \
                row_splits.dim_size(0) - 1, row_splits.flat<int64_t>().data(), \
                voxel_size.flat<T>().data(),                                   \
                points_range_min.flat<T>().data(),                             \
                points_range_max.flat<T>().data(), max_points_per_voxel,       \
                max_voxels, output_allocator);                                 \
        break;
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
};

#define REG_KB(type)                                            \
    REGISTER_KERNEL_BUILDER(Name("Open3DVoxelize")              \
                                    .Device(DEVICE_CPU)         \
                                    .TypeConstraint<type>("T"), \
                            VoxelizeOpKernelCPU<type>);
REG_KB(float)
REG_KB(double)
#undef REG_KB
