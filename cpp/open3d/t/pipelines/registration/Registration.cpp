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

#include "open3d/t/pipelines/registration/Registration.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        open3d::core::nns::NearestNeighborSearch &target_nns,
        double max_correspondence_distance,
        const core::Tensor &transformation) {
    core::Device device = source.GetDevice();

    transformation.AssertShape({4, 4});
    transformation.AssertDevice(device);

    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    bool check = target_nns.HybridIndex();
    if (!check) {
        utility::LogError(
                "[Tensor: EvaluateRegistration::"
                "GetRegistrationResultAndCorrespondences::"
                "NearestNeighborSearch::HybridSearch] "
                "Index is not set.");
        return result;
    }

    // Tensor implementation of HybridSearch takes square of max_corr_dist
    max_correspondence_distance =
            max_correspondence_distance * max_correspondence_distance;
    auto result_nns = target_nns.HybridSearch(source.GetPoints(),
                                              max_correspondence_distance, 1);
    auto corres_vec = result_nns.first.ToFlatVector<int64_t>();

    // This is unnecessary to itterate again through the entire thing just
    // to get the count of correspondence. Instead can be added in the
    // return value of SearchHybrid function.
    int corres_number = 0;
    for (size_t i = 0; i < corres_vec.size(); i++) {
        if (corres_vec[i] != -1) {
            corres_number++;
        }
    }

    // Reduction Sum of "distances"
    auto error2 = (result_nns.second.Sum({0})).Item<double_t>();

    result.correspondence_set_ = result_nns.first.Copy();
    result.fitness_ = (double)corres_number / (double)corres_vec.size();
    result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    return result;
}

RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const core::Tensor
                &transformation /* = core::Tensor::Eye(4,
                        core::Dtype::Float64, core::Device("CPU:0")))*/) {
    core::Device device = source.GetDevice();
    transformation.AssertShape({4, 4});
    transformation.AssertDevice(device);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());
    geometry::PointCloud pcd = source;
    // TODO: Check if transformation isIdentity (skip transform operation)
    pcd.Transform(transformation);

    return GetRegistrationResultAndCorrespondences(pcd, target, target_nns,
                                                   max_correspondence_distance,
                                                   transformation);
}

RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const core::Tensor &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    utility::LogError("Unimplemented");
    RegistrationResult result(
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0")));
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d