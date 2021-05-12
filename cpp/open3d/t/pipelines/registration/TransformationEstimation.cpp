// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/t/pipelines/registration/TransformationEstimation.h"

#include "open3d/t/pipelines/kernel/ComputeTransform.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

double TransformationEstimationPointToPoint::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &correspondences) const {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    double error;
    // TODO: Revist to support Float32 and 64 without type conversion.
    // TODO: Optimise using kernel.
    core::Tensor source_select =
            source.GetPoints().IndexGet({correspondences.first.Reshape({-1})});
    core::Tensor target_select =
            target.GetPoints().IndexGet({correspondences.second.Reshape({-1})});

    core::Tensor error_t = (source_select - target_select);
    error_t.Mul_(error_t);
    error = static_cast<double>(error_t.Sum({0, 1}).Item<float>());
    return std::sqrt(error /
                     static_cast<double>(correspondences.second.GetLength()));
}

core::Tensor TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &correspondences) const {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    core::Tensor R, t;
    std::tie(R, t) = pipelines::kernel::ComputeRtPointToPoint(
            source.GetPoints(), target.GetPoints(), correspondences);

    return t::pipelines::kernel::RtToTransformation(R, t);
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &correspondences) const {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    if (!target.HasPointNormals()) return 0.0;
    // TODO: Optimise using kernel.
    core::Tensor source_select =
            source.GetPoints().IndexGet({correspondences.first.Reshape({-1})});
    core::Tensor target_select =
            target.GetPoints().IndexGet({correspondences.second.Reshape({-1})});
    core::Tensor target_n_select = target.GetPointNormals().IndexGet(
            {correspondences.second.Reshape({-1})});

    core::Tensor error_t =
            (source_select - target_select).Mul_(target_n_select);
    error_t.Mul_(error_t);
    double error = static_cast<double>(error_t.Sum({0, 1}).Item<float>());
    return std::sqrt(error /
                     static_cast<double>(correspondences.second.GetLength()));
}

core::Tensor TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &correspondences) const {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    // Get pose {6} of type Float64 from correspondences indexed source and
    // target point cloud.
    core::Tensor pose = pipelines::kernel::ComputePosePointToPlane(
            source.GetPoints(), target.GetPoints(), target.GetPointNormals(),
            correspondences);

    // Get transformation {4,4} of type Float64 from pose {6}.
    return pipelines::kernel::PoseToTransformation(pose);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
