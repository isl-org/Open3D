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
//

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/VoxelizeOpKernel.h"
#include "torch/script.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Voxelize(
        torch::Tensor points,
        torch::Tensor voxel_size,
        torch::Tensor points_range_min,
        torch::Tensor points_range_max,
        const int64_t max_points_per_voxel,
        const int64_t max_voxels) {
    points = points.contiguous();

    // make sure that these tensors are on the cpu
    voxel_size = voxel_size.to(torch::kCPU).contiguous();
    points_range_min = points_range_min.to(torch::kCPU).contiguous();
    points_range_max = points_range_max.to(torch::kCPU).contiguous();

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
        TORCH_CHECK(0 < ndim.value() && ndim.value() < 9,
                    "the number of dimensions must be in [1,..,8]");
    }

    const auto& points_dtype = points.dtype();

    // output tensors
    torch::Tensor voxel_coords, voxel_point_indices, voxel_point_row_splits;

#define CALL(point_t, fn)                                                   \
    if (CompareTorchDtype<point_t>(points_dtype)) {                         \
        fn<point_t>(points, voxel_size, points_range_min, points_range_max, \
                    max_points_per_voxel, max_voxels, voxel_coords,         \
                    voxel_point_indices, voxel_point_row_splits);           \
        return std::make_tuple(voxel_coords, voxel_point_indices,           \
                               voxel_point_row_splits);                     \
    }

    if (points.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(float, VoxelizeCUDA)
        CALL(double, VoxelizeCUDA)
#else
        TORCH_CHECK(false, "Voxelize was not compiled with CUDA support")
#endif
    } else {
        CALL(float, VoxelizeCPU)
        CALL(double, VoxelizeCPU)
    }
    TORCH_CHECK(false, "Voxelize does not support " + points.toString() +
                               " as input for values")
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

static auto registry = torch::RegisterOperators(
        "open3d::voxelize(Tensor points, Tensor voxel_size, Tensor "
        "points_range_min, Tensor points_range_max, int "
        "max_points_per_voxel=9223372036854775807 , "
        "int max_voxels=9223372036854775807) -> (Tensor voxel_coords, Tensor "
        "voxel_point_indices, "
        "Tensor voxel_point_row_splits)",
        &Voxelize);
