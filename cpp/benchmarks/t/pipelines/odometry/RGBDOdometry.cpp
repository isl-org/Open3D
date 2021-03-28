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

#include <benchmark/benchmark.h>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/PointCloudIO.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

static core::Tensor CreateIntrisicTensor() {
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

static void ComputePosePointToPlane(benchmark::State& state,
                                    const core::Device& device) {
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

    for (auto _ : state) {
        core::Tensor trans = core::Tensor::Eye(4, core::Dtype::Float64,
                                               core::Device("CPU:0"));

        for (int i = 0; i < 20; ++i) {
            core::Tensor delta_src_to_dst =
                    t::pipelines::odometry::ComputePosePointToPlane(
                            src_vertex_map, dst_vertex_map, src_normal_map,
                            intrinsic_t, trans.To(device), depth_diff);
            trans = delta_src_to_dst.Matmul(trans);
        }
    }
}

BENCHMARK_CAPTURE(ComputePosePointToPlane, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(ComputePosePointToPlane, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d