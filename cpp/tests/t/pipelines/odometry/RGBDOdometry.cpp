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

#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

#include "core/CoreTest.h"
#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"
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

core::Tensor CreateIntrisicTensor() {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    return core::Tensor::Init<float>(
            {{static_cast<float>(focal_length.first), 0,
              static_cast<float>(principal_point.first)},
             {0, static_cast<float>(focal_length.second),
              static_cast<float>(principal_point.second)},
             {0, 0, 1}});
}

TEST_P(OdometryPermuteDevices, CreateVertexMap) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    t::geometry::Image depth = *t::io::CreateImageFromFile(
            std::string(TEST_DATA_DIR) + "/depth.png");
    depth = depth.To(device).To(core::Dtype::Float32, false, 1.0);

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    core::Tensor vertex_map = t::pipelines::odometry::CreateVertexMap(
            depth, intrinsic_t.To(device));
    core::Tensor vertex_map_gt = core::Tensor::Load(fmt::format(
            "{}/open3d_downloads/RGBD/vertex_map.npy", TEST_DATA_DIR));

    // AllClose doesn't work for inf, but two vtx maps are strictly equivalent.
    int64_t sum = vertex_map.Eq(vertex_map_gt.To(device))
                          .To(core::Dtype::Int64)
                          .Sum({0, 1, 2})
                          .Item<int64_t>();
    EXPECT_EQ(sum, vertex_map.NumElements());
}

TEST_P(OdometryPermuteDevices, CreateNormalMap) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    t::geometry::Image depth = *t::io::CreateImageFromFile(
            std::string(TEST_DATA_DIR) + "/depth.png");
    depth = depth.To(device).To(core::Dtype::Float32, false, 1.0);

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    core::Tensor vertex_map = t::pipelines::odometry::CreateVertexMap(
            depth, intrinsic_t.To(device));
    core::Tensor normal_map =
            t::pipelines::odometry::CreateNormalMap(vertex_map);

    core::Tensor normal_map_gt = core::Tensor::Load(fmt::format(
            "{}/open3d_downloads/RGBD/normal_map.npy", TEST_DATA_DIR));

    // AllClose doesn't work for inf, so we ignore the 1st dimension.
    EXPECT_TRUE(normal_map.Slice(2, 1, 3).AllClose(
            normal_map_gt.Slice(2, 1, 3).To(device)));
}

TEST_P(OdometryPermuteDevices, ComputePosePointToPlane) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    float depth_factor = 1000.0;
    float depth_diff = 0.07;

    t::geometry::Image src_depth = *t::io::CreateImageFromFile(
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00000.png");
    t::geometry::Image dst_depth = *t::io::CreateImageFromFile(
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00002.png");

    src_depth = src_depth.To(device).To(core::Dtype::Float32, false, 1.0);
    dst_depth = dst_depth.To(device).To(core::Dtype::Float32, false, 1.0);

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    core::Tensor src_vertex_map = t::pipelines::odometry::CreateVertexMap(
            src_depth, intrinsic_t.To(device), depth_factor);
    core::Tensor src_normal_map =
            t::pipelines::odometry::CreateNormalMap(src_vertex_map);

    core::Tensor dst_vertex_map = t::pipelines::odometry::CreateVertexMap(
            dst_depth, intrinsic_t.To(device), depth_factor);

    core::Tensor trans =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0"));
    for (int i = 0; i < 20; ++i) {
        core::Tensor delta_src_to_dst =
                t::pipelines::odometry::ComputePosePointToPlane(
                        src_vertex_map, dst_vertex_map, src_normal_map,
                        intrinsic_t, trans.To(device), depth_diff);
        trans = delta_src_to_dst.Matmul(trans);
    }

    core::Device host("CPU:0");
    core::Tensor T0 = core::Tensor::Init<double>(
            {{-0.2739592186924325, 0.021819345900466677, -0.9614937663021573,
              -0.31057997014702826},
             {8.33962904204855e-19, -0.9997426093226981, -0.02268733357278151,
              0.5730122438481298},
             {-0.9617413095492113, -0.006215404179813816, 0.27388870414358013,
              2.1264800183565487},
             {0.0, 0.0, 0.0, 1.0}},
            host);
    core::Tensor T2 = core::Tensor::Init<double>(
            {{-0.26535185454036697, 0.04522708142999141, -0.9630902888085378,
              -0.3097373196756845},
             {1.6706953334814538e-18, -0.9988991819470762,
              -0.046908680491589354, 0.6204495589484211},
             {-0.9641516443443884, -0.012447305362484767, 0.2650597504285121,
              2.1247894438735306},
             {0.0, 0.0, 0.0, 1.0}},
            host);

    core::Tensor Tdiff = T2.Inverse().Matmul(T0).Matmul(
            trans.To(host, core::Dtype::Float64).Inverse());
    core::Tensor Ttrans = Tdiff.Slice(0, 0, 3).Slice(1, 3, 4);
    EXPECT_LE(Ttrans.T().Matmul(Ttrans).Item<double>(), 3e-4);
}

TEST_P(OdometryPermuteDevices, MultiScaleOdometry) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    float depth_factor = 1000.0;
    float depth_diff = 0.07;

    t::geometry::Image src_depth = *t::io::CreateImageFromFile(
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00000.png");
    t::geometry::Image dst_depth = *t::io::CreateImageFromFile(
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00002.png");

    t::geometry::RGBDImage src, dst;
    src.depth_ = src_depth.To(device).To(core::Dtype::Float32, false, 1.0);
    dst.depth_ = dst_depth.To(device).To(core::Dtype::Float32, false, 1.0);

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    core::Tensor trans =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0"));
    trans = t::pipelines::odometry::RGBDOdometryMultiScale(
            src, dst, intrinsic_t, trans, depth_factor, depth_diff, {10, 5, 3});

    core::Device host("CPU:0");
    core::Tensor T0 = core::Tensor::Init<double>(
            {{-0.2739592186924325, 0.021819345900466677, -0.9614937663021573,
              -0.31057997014702826},
             {8.33962904204855e-19, -0.9997426093226981, -0.02268733357278151,
              0.5730122438481298},
             {-0.9617413095492113, -0.006215404179813816, 0.27388870414358013,
              2.1264800183565487},
             {0.0, 0.0, 0.0, 1.0}},
            host);
    core::Tensor T2 = core::Tensor::Init<double>(
            {{-0.26535185454036697, 0.04522708142999141, -0.9630902888085378,
              -0.3097373196756845},
             {1.6706953334814538e-18, -0.9988991819470762,
              -0.046908680491589354, 0.6204495589484211},
             {-0.9641516443443884, -0.012447305362484767, 0.2650597504285121,
              2.1247894438735306},
             {0.0, 0.0, 0.0, 1.0}},
            host);

    core::Tensor Tdiff = T2.Inverse().Matmul(T0).Matmul(
            trans.To(host, core::Dtype::Float64).Inverse());
    core::Tensor Ttrans = Tdiff.Slice(0, 0, 3).Slice(1, 3, 4);
    EXPECT_LE(Ttrans.T().Matmul(Ttrans).Item<double>(), 5e-5);
}

}  // namespace tests
}  // namespace open3d
