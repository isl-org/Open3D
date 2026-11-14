// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "core/CoreTest.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/t/geometry/VoxelBlockGrid.h"

namespace open3d {
namespace tests {
namespace {

constexpr int kPlaneRayCount = 33 * 25;
constexpr double kPlaneBiasLimit = 0.001;
constexpr double kPlaneRmsLimit = 0.0025;
constexpr double kFlatPlaneLimit = 1e-6;

std::vector<double> IntegratePlaneErrors(const core::Device &device,
                                         double sx,
                                         double sy) {
    const core::Device cpu("CPU:0");
    constexpr int width = 128, height = 96;
    constexpr double fx = 100.0, fy = 110.0, cx = 60.25, cy = 50.75;
    const auto intrinsic =
            core::Tensor::Init<double>({{fx, 0, cx}, {0, fy, cy}, {0, 0, 1}});
    const auto extrinsic = core::Tensor::Eye(4, core::Float64, cpu);

    std::vector<float> depths(width * height);
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            depths[v * width + u] = static_cast<float>(
                    2.0 / (1.0 - sx * (u - cx) / fx - sy * (v - cy) / fy));
        }
    }
    const t::geometry::Image depth(
            core::Tensor(depths, {height, width, 1}, core::Float32, device));

    std::vector<int> keys;
    for (int z = 21; z < 29; ++z) {
        for (int y = -4; y < 4; ++y) {
            for (int x = -5; x < 5; ++x) {
                keys.insert(keys.end(), {x, y, z});
            }
        }
    }
    const core::Tensor blocks(keys, {640, 3}, core::Int32, device);
    t::geometry::VoxelBlockGrid grid(
            {"tsdf", "weight", "color"},
            {core::Float32, core::Float32, core::Float32}, {{1}, {1}, {3}},
            0.01f, 8, 640, device);
    core::Tensor block_indices, block_masks;
    grid.GetHashMap().Activate(blocks, block_indices, block_masks);
    grid.GetAttribute("tsdf").Fill(0.f);
    grid.GetAttribute("weight").Fill(0.f);
    grid.GetAttribute("color").Fill(0.f);
    grid.Integrate(blocks, depth, intrinsic, extrinsic, 1.f, 4.f, 4.f);

    const auto mesh = grid.ExtractTriangleMesh(0.5f).To(cpu);
    t::geometry::RaycastingScene scene;
    scene.AddTriangles(mesh);

    std::vector<float> rays;
    for (int y = 0; y < 25; ++y) {
        for (int x = 0; x < 33; ++x) {
            rays.insert(rays.end(),
                        {float(-0.3 + 0.6 * x / 32),
                         float(-0.22 + 0.44 * y / 24), 4.f, 0.f, 0.f, -1.f});
        }
    }
    const auto hits = scene.CastRays(core::Tensor(rays, {kPlaneRayCount, 6},
                                                  core::Float32, cpu))
                              .at("t_hit");
    const float *distance = hits.GetDataPtr<float>();
    std::vector<double> errors(kPlaneRayCount);
    const double normal_length = std::sqrt(1.0 + sx * sx + sy * sy);
    for (int i = 0; i < kPlaneRayCount; ++i) {
        errors[i] = (4.0 - distance[i] - 2.0 - sx * rays[6 * i] -
                     sy * rays[6 * i + 1]) /
                    normal_length;
    }
    return errors;
}

}  // namespace

class VoxelBlockGridPlanePermuteDevices : public PermuteDevicesWithSYCL {};
INSTANTIATE_TEST_SUITE_P(
        VoxelBlockGridPlane,
        VoxelBlockGridPlanePermuteDevices,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

TEST_P(VoxelBlockGridPlanePermuteDevices, IntegratePlaneSurface) {
    for (const auto &slope : std::vector<std::array<double, 2>>{
                 {0, 0}, {0.5, 0}, {-0.5, 0}, {0, 0.5}, {0, -0.5}}) {
        SCOPED_TRACE(testing::Message()
                     << "sx=" << slope[0] << ", sy=" << slope[1]);
        const auto errors =
                IntegratePlaneErrors(GetParam(), slope[0], slope[1]);
        ASSERT_EQ(errors.size(), kPlaneRayCount);
        ASSERT_TRUE(std::all_of(errors.begin(), errors.end(),
                                [](double e) { return std::isfinite(e); }));

        const double bias = std::accumulate(errors.begin(), errors.end(), 0.0) /
                            errors.size();
        const double rms =
                std::sqrt(std::inner_product(errors.begin(), errors.end(),
                                             errors.begin(), 0.0) /
                          errors.size());
        const bool flat = slope[0] == 0 && slope[1] == 0;

        EXPECT_LE(std::abs(bias), flat ? kFlatPlaneLimit : kPlaneBiasLimit);
        EXPECT_LE(rms, flat ? kFlatPlaneLimit : kPlaneRmsLimit);
    }
}

}  // namespace tests
}  // namespace open3d
