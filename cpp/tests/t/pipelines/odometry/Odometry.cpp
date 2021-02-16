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

#include "core/CoreTest.h"
#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class OdometryPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Odometry,
                         OdometryPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class OdometryPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Odometry,
        OdometryPermuteDevicePairs,
        testing::ValuesIn(OdometryPermuteDevicePairs::TestCases()));

TEST_P(OdometryPermuteDevices, CreateVertexMap) {
    core::Device device = GetParam();

    auto depth_legacy =
            io::CreateImageFromFile(std::string(TEST_DATA_DIR) + "/depth.png");

    t::geometry::Image depth =
            t::geometry::Image::FromLegacyImage(*depth_legacy, device);
    depth = depth.To(core::Dtype::Float32, false, 1.0);

    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    core::Tensor intrinsic_t = core::Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, core::Dtype::Float32);

    core::Tensor vertex_map = t::pipelines::odometry::CreateVertexMap(
            depth, intrinsic_t.To(device));
    vertex_map.Save("vertex_map.npy");

    io::WritePointCloud(
            "vertex_map.ply",
            t::geometry::PointCloud({{"points", vertex_map.View({-1, 3})}})
                    .ToLegacyPointCloud());
}

TEST_P(OdometryPermuteDevices, CreateNormalMap) {
    core::Device device = GetParam();

    auto depth_legacy =
            io::CreateImageFromFile(std::string(TEST_DATA_DIR) + "/depth.png");

    t::geometry::Image depth =
            t::geometry::Image::FromLegacyImage(*depth_legacy, device);
    depth = depth.To(core::Dtype::Float32, false, 1.0);

    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    core::Tensor intrinsic_t = core::Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, core::Dtype::Float32);

    core::Tensor vertex_map = t::pipelines::odometry::CreateVertexMap(
            depth, intrinsic_t.To(device));

    t::geometry::Image depth_filtered = depth.FilterBilateral(5, 50, 50);
    depth_filtered.AsTensor().Save("depth_filtered.npy");
    core::Tensor vertex_map_filtered = t::pipelines::odometry::CreateVertexMap(
            depth_filtered, intrinsic_t.To(device));
    vertex_map_filtered.Save("vertex_map_filtered.npy");
    core::Tensor normal_map =
            t::pipelines::odometry::CreateNormalMap(vertex_map_filtered);
    normal_map.Save("normal_map.npy");

    io::WritePointCloud("vertex_map_w_normal.ply",
                        t::geometry::PointCloud(
                                {{"points", vertex_map_filtered.View({-1, 3})},
                                 {"normals", normal_map}})
                                .ToLegacyPointCloud());
}

}  // namespace tests
}  // namespace open3d
