// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/VoxelizeOpKernel.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> Voxelize(paddle::Tensor& points,
                                     paddle::Tensor& row_splits,
                                     paddle::Tensor& voxel_size,
                                     paddle::Tensor& points_range_min,
                                     paddle::Tensor& points_range_max,
                                     const int64_t max_points_per_voxel,
                                     const int64_t max_voxels) {
    CHECK_TYPE(row_splits, phi::DataType::INT64);

    auto cpu_place = paddle::CPUPlace();
    // make sure that these tensors are on the cpu
    voxel_size = voxel_size.copy_to(cpu_place, false);
    points_range_min = points_range_min.copy_to(cpu_place, false);
    points_range_max = points_range_max.copy_to(cpu_place, false);

    CHECK_SAME_DTYPE(points, voxel_size, points_range_min, points_range_max);

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim num_points("num_points");
        Dim ndim("ndim");
        CHECK_SHAPE(points, num_points, ndim);
        CHECK_SHAPE(voxel_size, ndim);
        CHECK_SHAPE(points_range_min, ndim);
        CHECK_SHAPE(points_range_max, ndim);
        PD_CHECK(0 < ndim.value() && ndim.value() < 9,
                 "the number of dimensions must be in [1,..,8]");
    }

    const auto& points_dtype = points.dtype();

    // output tensors
    paddle::Tensor voxel_coords, voxel_point_indices, voxel_point_row_splits,
            voxel_batch_splits;

#define CALL(point_t, fn)                                                      \
    if (ComparePaddleDtype<point_t>(points_dtype)) {                           \
        fn<point_t>(points, row_splits, voxel_size, points_range_min,          \
                    points_range_max, max_points_per_voxel, max_voxels,        \
                    voxel_coords, voxel_point_indices, voxel_point_row_splits, \
                    voxel_batch_splits);                                       \
        return {voxel_coords, voxel_point_indices, voxel_point_row_splits,     \
                voxel_batch_splits};                                           \
    }

    if (points.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(float, VoxelizeCUDA)
        CALL(double, VoxelizeCUDA)
#else
        PD_CHECK(false, "Voxelize was not compiled with CUDA support");
#endif
    } else {
        CALL(float, VoxelizeCPU)
        CALL(double, VoxelizeCPU)
    }

    PD_CHECK(false, "Voxelize does not support " +
                            phi::DataTypeToString(points.dtype()) +
                            " as input for values");

    return std::vector<paddle::Tensor>();
}

std::vector<paddle::DataType> VoxelizeInferDtype() {
    return {paddle::DataType::INT32, paddle::DataType::INT64,
            paddle::DataType::INT64, paddle::DataType::INT64};
}

PD_BUILD_OP(open3d_voxelize)
        .Inputs({"points", "row_splits", "voxel_size", "points_range_min",
                 "points_range_max"})
        .Outputs({"voxel_coords", "voxel_point_indices",
                  "voxel_point_row_splits", "voxel_batch_splits"})
        .Attrs({
                "max_points_per_voxel: int64_t",
                "max_voxels: int64_t",
        })
        .SetKernelFn(PD_KERNEL(Voxelize))
        .SetInferDtypeFn(PD_INFER_DTYPE(VoxelizeInferDtype));