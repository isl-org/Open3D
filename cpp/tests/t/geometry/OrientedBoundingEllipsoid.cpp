// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class OrientedBoundingEllipsoidPermuteDevices : public PermuteDevicesWithSYCL {
};
INSTANTIATE_TEST_SUITE_P(
        OrientedBoundingEllipsoid,
        OrientedBoundingEllipsoidPermuteDevices,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

// Mirror of test_pointcloud_get_oriented_bounding_ellipsoid in
// python/test/t/geometry/test_bounding_volume_ellipsoid.py
TEST_P(OrientedBoundingEllipsoidPermuteDevices,
       PointCloudGetOrientedBoundingEllipsoid) {
    core::Device device = GetParam();

    // Six points spread along three axes — non-degenerate for Khachiyan.
    t::geometry::PointCloud pcd(core::Tensor::Init<float>({{1.0f, 0.0f, 0.0f},
                                                           {-1.0f, 0.0f, 0.0f},
                                                           {0.0f, 2.0f, 0.0f},
                                                           {0.0f, -2.0f, 0.0f},
                                                           {0.0f, 0.0f, 3.0f},
                                                           {0.0f, 0.0f, -3.0f}},
                                                          device));

    t::geometry::OrientedBoundingEllipsoid obe =
            pcd.GetOrientedBoundingEllipsoid();

    EXPECT_NEAR(obe.Volume(), 25.13274123, 1e-5);
    // Center should be near the origin for this symmetric point set.
    EXPECT_TRUE(obe.GetCenter().AllClose(
            core::Tensor::Zeros({3}, core::Float32, device),
            /*rtol=*/0.0, /*atol=*/1e-3));
}

// Mirror of test_trianglemesh_get_oriented_bounding_ellipsoid in
// python/test/t/geometry/test_bounding_volume_ellipsoid.py
TEST_P(OrientedBoundingEllipsoidPermuteDevices,
       TriangleMeshGetOrientedBoundingEllipsoid) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh = t::geometry::TriangleMesh::CreateSphere(
            /*radius=*/1.0, /*resolution=*/20,
            /*float_dtype=*/core::Float32, /*int_dtype=*/core::Int64, device);

    t::geometry::OrientedBoundingEllipsoid obe =
            mesh.GetOrientedBoundingEllipsoid();

    EXPECT_NEAR(obe.Volume(), 4.18879, 1e-5);
    // Sphere centered at origin — ellipsoid center should be near origin.
    EXPECT_TRUE(obe.GetCenter().AllClose(
            core::Tensor::Zeros({3}, core::Float32, device),
            /*rtol=*/0.0, /*atol=*/1e-3));
    // All semi-axis radii must be positive.
    EXPECT_TRUE(obe.GetRadii()
                        .Gt(0.0)
                        .All()
                        .To(core::Device("CPU:0"))
                        .Item<bool>());
}

}  // namespace tests
}  // namespace open3d
