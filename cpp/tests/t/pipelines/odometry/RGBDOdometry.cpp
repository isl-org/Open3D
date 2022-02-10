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

#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

#include "core/CoreTest.h"
#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class OdometryPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Odometry,
                         OdometryPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

core::Tensor CreateIntrisicTensor() {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    return core::Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});
}

TEST_P(OdometryPermuteDevices, ComputeOdometryResultPointToPlane) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    const float depth_scale = 1000.0;
    const float depth_diff = 0.07;

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image src_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0]);
    t::geometry::Image dst_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[2]);

    src_depth = src_depth.To(device);
    dst_depth = dst_depth.To(device);

    core::Tensor intrinsic_t = CreateIntrisicTensor();

    t::geometry::Image src_depth_processed =
            src_depth.ClipTransform(depth_scale, 0.0, 3.0, NAN);
    t::geometry::Image src_vertex_map =
            src_depth_processed.CreateVertexMap(intrinsic_t, NAN);
    t::geometry::Image src_normal_map = src_vertex_map.CreateNormalMap(NAN);

    t::geometry::Image dst_depth_processed =
            dst_depth.ClipTransform(depth_scale, 0.0, 3.0, NAN);
    t::geometry::Image dst_vertex_map =
            dst_depth_processed.CreateVertexMap(intrinsic_t, NAN);

    core::Tensor trans =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));
    for (int i = 0; i < 20; ++i) {
        auto result = t::pipelines::odometry::ComputeOdometryResultPointToPlane(
                src_vertex_map.AsTensor(), dst_vertex_map.AsTensor(),
                src_normal_map.AsTensor(), intrinsic_t, trans, depth_diff,
                depth_diff * 0.5);
        trans = result.transformation_.Matmul(trans).Contiguous();
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
            trans.To(host, core::Float64).Inverse());
    core::Tensor Ttrans = Tdiff.Slice(0, 0, 3).Slice(1, 3, 4);
    EXPECT_LE(Ttrans.T().Matmul(Ttrans).Item<double>(), 3e-4);
}

TEST_P(OdometryPermuteDevices, RGBDOdometryMultiScalePointToPlane) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    const float depth_scale = 1000.0;
    const float depth_max = 3.0;
    const float depth_diff = 0.07;

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image src_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0]);
    t::geometry::Image dst_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[2]);
    t::geometry::Image src_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[0]);
    t::geometry::Image dst_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[2]);

    t::geometry::RGBDImage src, dst;
    src.color_ = src_color.To(device);
    dst.color_ = dst_color.To(device);
    src.depth_ = src_depth.To(device);
    dst.depth_ = dst_depth.To(device);

    std::vector<t::pipelines::odometry::Method> methods{
            t::pipelines::odometry::Method::PointToPlane,
            t::pipelines::odometry::Method::Intensity,
            t::pipelines::odometry::Method::Hybrid};

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    core::Tensor trans =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));
    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(
            src, dst, intrinsic_t, trans, depth_scale, depth_max,
            std::vector<t::pipelines::odometry::OdometryConvergenceCriteria>{
                    10, 5, 3},
            t::pipelines::odometry::Method::PointToPlane,
            t::pipelines::odometry::OdometryLossParams(depth_diff));

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
            result.transformation_.To(host, core::Float64).Inverse());
    core::Tensor Ttrans = Tdiff.Slice(0, 0, 3).Slice(1, 3, 4);
    EXPECT_LE(Ttrans.T().Matmul(Ttrans).Item<double>(), 5e-5);
}

TEST_P(OdometryPermuteDevices, RGBDOdometryMultiScaleIntensity) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    const float depth_scale = 1000.0;
    const float depth_max = 3.0;
    const float depth_diff = 0.07;

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image src_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0]);
    t::geometry::Image dst_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[2]);
    t::geometry::Image src_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[0]);
    t::geometry::Image dst_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[2]);

    t::geometry::RGBDImage src, dst;
    src.color_ = src_color.To(device);
    dst.color_ = dst_color.To(device);
    src.depth_ = src_depth.To(device);
    dst.depth_ = dst_depth.To(device);

    std::vector<t::pipelines::odometry::Method> methods{
            t::pipelines::odometry::Method::PointToPlane,
            t::pipelines::odometry::Method::Intensity,
            t::pipelines::odometry::Method::Hybrid};

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    core::Tensor trans =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));
    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(
            src, dst, intrinsic_t, trans, depth_scale, depth_max,
            std::vector<t::pipelines::odometry::OdometryConvergenceCriteria>{
                    10, 5, 3},
            t::pipelines::odometry::Method::Intensity,
            t::pipelines::odometry::OdometryLossParams(depth_diff));

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
            result.transformation_.To(host, core::Float64).Inverse());
    core::Tensor Ttrans = Tdiff.Slice(0, 0, 3).Slice(1, 3, 4);
    EXPECT_LE(Ttrans.T().Matmul(Ttrans).Item<double>(), 5e-5);
}

TEST_P(OdometryPermuteDevices, RGBDOdometryMultiScaleHybrid) {
    core::Device device = GetParam();
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    const float depth_scale = 1000.0;
    const float depth_max = 3.0;
    const float depth_diff = 0.07;

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image src_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0]);
    t::geometry::Image dst_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[2]);
    t::geometry::Image src_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[0]);
    t::geometry::Image dst_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[2]);

    t::geometry::RGBDImage src, dst;
    src.color_ = src_color.To(device);
    dst.color_ = dst_color.To(device);
    src.depth_ = src_depth.To(device);
    dst.depth_ = dst_depth.To(device);

    std::vector<t::pipelines::odometry::Method> methods{
            t::pipelines::odometry::Method::PointToPlane,
            t::pipelines::odometry::Method::Intensity,
            t::pipelines::odometry::Method::Hybrid};

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    core::Tensor trans =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));
    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(
            src, dst, intrinsic_t, trans, depth_scale, depth_max,
            std::vector<t::pipelines::odometry::OdometryConvergenceCriteria>{
                    10, 5, 3},
            t::pipelines::odometry::Method::Hybrid,
            t::pipelines::odometry::OdometryLossParams(depth_diff));

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
            result.transformation_.To(host, core::Float64).Inverse());
    core::Tensor Ttrans = Tdiff.Slice(0, 0, 3).Slice(1, 3, 4);
    EXPECT_LE(Ttrans.T().Matmul(Ttrans).Item<double>(), 5e-5);
}
}  // namespace tests
}  // namespace open3d
