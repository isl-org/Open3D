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

#include <benchmark/benchmark.h>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/DataManager.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

static core::Tensor CreateIntrisicTensor() {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    return core::Tensor::Init<double>(
            {{(focal_length.first), 0, (principal_point.first)},
             {0, (focal_length.second), (principal_point.second)},
             {0, 0, 1}});
}

static void ComputeOdometryResultPointToPlane(benchmark::State& state,
                                              const core::Device& device) {
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {
        return;
    }

    const float depth_scale = 1000.0;
    const float depth_diff = 0.07;
    const float depth_max = 3.0;

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image src_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0]);
    t::geometry::Image dst_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[2]);

    src_depth = src_depth.To(device);
    dst_depth = dst_depth.To(device);

    core::Tensor intrinsic_t = CreateIntrisicTensor();
    t::geometry::Image src_depth_processed =
            src_depth.ClipTransform(depth_scale, 0.0, depth_max, NAN);
    t::geometry::Image src_vertex_map =
            src_depth_processed.CreateVertexMap(intrinsic_t, NAN);
    t::geometry::Image src_normal_map = src_vertex_map.CreateNormalMap(NAN);

    t::geometry::Image dst_depth_processed =
            dst_depth.ClipTransform(depth_scale, 0.0, depth_max, NAN);
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

    for (auto _ : state) {
        core::Tensor trans =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));

        for (int i = 0; i < 20; ++i) {
            auto result =
                    t::pipelines::odometry::ComputeOdometryResultPointToPlane(
                            src_vertex_map.AsTensor(),
                            dst_vertex_map.AsTensor(),
                            src_normal_map.AsTensor(), intrinsic_t, trans,
                            depth_diff, depth_diff * 0.5);
            trans = result.transformation_.Matmul(trans).Contiguous();
        }
        core::cuda::Synchronize(device);
    }
}

static void RGBDOdometryMultiScale(
        benchmark::State& state,
        const core::Device& device,
        const t::pipelines::odometry::Method& method) {
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
    t::geometry::Image src_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[0]);

    t::geometry::Image dst_depth =
            *t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[2]);

    t::geometry::Image dst_color =
            *t::io::CreateImageFromFile(redwood_data.GetColorPaths()[0]);

    t::geometry::RGBDImage source, target;
    source.color_ = src_color.To(device);
    source.depth_ = src_depth.To(device);
    target.color_ = dst_color.To(device);
    target.depth_ = dst_depth.To(device);

    core::Tensor intrinsic_t = CreateIntrisicTensor();

    // Very strict criteria to ensure running most the iterations
    t::pipelines::odometry::OdometryLossParams loss(depth_diff);
    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> criteria{
            t::pipelines::odometry::OdometryConvergenceCriteria(10, 1e-12,
                                                                1e-12),
            t::pipelines::odometry::OdometryConvergenceCriteria(5, 1e-12,
                                                                1e-12),
            t::pipelines::odometry::OdometryConvergenceCriteria(3, 1e-12,
                                                                1e-12)};

    // Warp up
    RGBDOdometryMultiScale(
            source, target, intrinsic_t,
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            depth_scale, depth_max, criteria, method, loss);

    for (auto _ : state) {
        RGBDOdometryMultiScale(
                source, target, intrinsic_t,
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
                depth_scale, depth_max, criteria, method, loss);
        core::cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(ComputeOdometryResultPointToPlane, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);
#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(ComputeOdometryResultPointToPlane,
                  CUDA,
                  core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(RGBDOdometryMultiScale,
                  Hybrid_CPU,
                  core::Device("CPU:0"),
                  t::pipelines::odometry::Method::Hybrid)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(RGBDOdometryMultiScale,
                  Intensity_CPU,
                  core::Device("CPU:0"),
                  t::pipelines::odometry::Method::Intensity)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(RGBDOdometryMultiScale,
                  PointToPlane_CPU,
                  core::Device("CPU:0"),
                  t::pipelines::odometry::Method::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(RGBDOdometryMultiScale,
                  Hybrid_CUDA,
                  core::Device("CUDA:0"),
                  t::pipelines::odometry::Method::Hybrid)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(RGBDOdometryMultiScale,
                  Intensity_CUDA,
                  core::Device("CUDA:0"),
                  t::pipelines::odometry::Method::Intensity)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(RGBDOdometryMultiScale,
                  PointToPlane_CUDA,
                  core::Device("CUDA:0"),
                  t::pipelines::odometry::Method::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif
}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
