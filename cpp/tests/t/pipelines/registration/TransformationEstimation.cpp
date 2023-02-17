// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class TransformationEstimationPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TransformationEstimation,
                         TransformationEstimationPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

static std::
        tuple<t::geometry::PointCloud, t::geometry::PointCloud, core::Tensor>
        GetTestPointCloudsAndCorrespondences(const core::Dtype& dtype,
                                             const core::Device& device) {
    core::Tensor source_points =
            core::Tensor::Init<double>({{1.15495, 2.40671, 1.15061},
                                        {1.81481, 2.06281, 1.71927},
                                        {0.888322, 2.05068, 2.04879},
                                        {3.78842, 1.70788, 1.30246},
                                        {1.8437, 2.22894, 0.986237},
                                        {2.95706, 2.20180, 0.987878},
                                        {1.72644, 1.24356, 1.93486},
                                        {0.922024, 1.14872, 2.34317},
                                        {3.70293, 1.85134, 1.15357},
                                        {3.06505, 1.30386, 1.55279},
                                        {0.634826, 1.04995, 2.47046},
                                        {1.40107, 1.37469, 1.09687},
                                        {2.93002, 1.96242, 1.48532},
                                        {3.74384, 1.30258, 1.30244}},
                                       device);

    t::geometry::PointCloud source(source_points.To(device, dtype));

    core::Tensor target_points =
            core::Tensor::Init<double>({{2.41766, 2.05397, 1.74994},
                                        {1.37848, 2.19793, 1.66553},
                                        {2.24325, 2.27183, 1.33708},
                                        {3.09898, 1.98482, 1.77401},
                                        {1.81615, 1.48337, 1.49697},
                                        {3.01758, 2.20312, 1.51502},
                                        {2.38836, 1.39096, 1.74914},
                                        {1.30911, 1.4252, 1.37429},
                                        {3.16847, 1.39194, 1.90959},
                                        {1.59412, 1.53304, 1.58040},
                                        {1.34342, 2.19027, 1.30075}},
                                       device);

    core::Tensor target_normals =
            core::Tensor::Init<double>({{-0.00850160, -0.22355, -0.519574},
                                        {0.257463, -0.0738755, -0.698319},
                                        {0.0574301, -0.484248, -0.409929},
                                        {-0.0123503, -0.230172, -0.520720},
                                        {0.355904, -0.142007, -0.720467},
                                        {0.0674038, -0.418757, -0.458602},
                                        {0.226091, 0.258253, -0.874024},
                                        {0.43979, 0.122441, -0.574998},
                                        {0.109144, 0.180992, -0.762368},
                                        {0.273325, 0.292013, -0.903111},
                                        {0.385407, -0.212348, -0.277818}},
                                       device);

    t::geometry::PointCloud target(target_points.To(device, dtype));
    target.SetPointNormals(target_normals.To(device, dtype));

    core::Tensor corres = core::Tensor::Init<int64_t>(
            {10, 1, 1, 3, 2, 5, 9, 7, 5, 8, 7, 7, 5, 8}, device);

    return std::make_tuple(source, target, corres);
}

TEST_P(TransformationEstimationPermuteDevices, ComputeRMSEPointToPoint) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPoint
                estimation_p2p;
        double p2p_rmse =
                estimation_p2p.ComputeRMSE(source_pcd, target_pcd, corres);

        EXPECT_NEAR(p2p_rmse, 0.706437, 0.0001);
    }
}

TEST_P(TransformationEstimationPermuteDevices,
       ComputeTransformationPointToPoint) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPoint
                estimation_p2p;

        // Get transform.
        core::Tensor p2p_transform = estimation_p2p.ComputeTransformation(
                source_pcd, target_pcd, corres);
        // Apply transform.
        t::geometry::PointCloud source_transformed_p2p = source_pcd.Clone();
        source_transformed_p2p.Transform(p2p_transform);
        double p2p_rmse_ = estimation_p2p.ComputeRMSE(source_transformed_p2p,
                                                      target_pcd, corres);

        // Compare the new RMSE after transformation.
        EXPECT_NEAR(p2p_rmse_, 0.578255, 0.0001);
    }
}

TEST_P(TransformationEstimationPermuteDevices, ComputeRMSEPointToPlane) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPlane
                estimation_p2plane;
        double p2plane_rmse =
                estimation_p2plane.ComputeRMSE(source_pcd, target_pcd, corres);

        EXPECT_NEAR(p2plane_rmse, 0.335499, 0.0001);
    }
}

TEST_P(TransformationEstimationPermuteDevices,
       ComputeTransformationPointToPlane) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPlane
                estimation_p2plane;

        // Get transform.
        core::Tensor p2plane_transform =
                estimation_p2plane.ComputeTransformation(source_pcd, target_pcd,
                                                         corres);
        // Apply transform.
        t::geometry::PointCloud source_transformed_p2plane = source_pcd.Clone();
        source_transformed_p2plane.Transform(p2plane_transform);
        double p2plane_rmse_ = estimation_p2plane.ComputeRMSE(
                source_transformed_p2plane, target_pcd, corres);

        // Compare the new RMSE after transformation.
        EXPECT_NEAR(p2plane_rmse_, 0.601422, 0.0001);
    }
}

}  // namespace tests
}  // namespace open3d
