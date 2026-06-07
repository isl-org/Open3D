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

class BoundingSpherePermuteDevices : public PermuteDevicesWithSYCL {
};
INSTANTIATE_TEST_SUITE_P(
        BoundingSphere,
        BoundingSpherePermuteDevices,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

// Mirror of test_pointcloud_get_oriented_bounding_ellipsoid in
// python/test/t/geometry/test_bounding_volume_ellipsoid.py
TEST_P(BoundingSpherePermuteDevices,
       PointCloudGetBoundingSphere) {
    core::Device device = GetParam();

    // Six points spread along three axes.
    t::geometry::PointCloud pcd(core::Tensor::Init<float>({{1.0f, 0.0f, 0.0f},
                                                           {-1.0f, 0.0f, 0.0f},
                                                           {0.0f, 2.0f, 0.0f},
                                                           {0.0f, -2.0f, 0.0f},
                                                           {0.0f, 0.0f, 3.0f},
                                                           {0.0f, 0.0f, -3.0f}},
                                                          device));

    t::geometry::BoundingSphere ebs =
            pcd.GetBoundingSphere();
    // Volume of sphere with diameter of 6.0 
    // (the largest distance between points).
    EXPECT_NEAR(ebs.Volume(), 113.09733552923255, 1e-5);
    // Center should be near the origin for this symmetric point set.
    EXPECT_TRUE(ebs.GetCenter().AllClose(
            core::Tensor::Zeros({3}, core::Float32, device),
            /*rtol=*/0.0, /*atol=*/1e-3));
}

// Mirror of test_trianglemesh_get_oriented_bounding_ellipsoid in
// python/test/t/geometry/test_bounding_volume_ellipsoid.py
TEST_P(BoundingSpherePermuteDevices,
       TriangleMeshGetBoundingSphere) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh = t::geometry::TriangleMesh::CreateSphere(
            /*radius=*/1.0, /*resolution=*/20,
            /*float_dtype=*/core::Float32, /*int_dtype=*/core::Int64, device);

    t::geometry::BoundingSphere ebs =
            mesh.GetBoundingSphere();

    EXPECT_NEAR(ebs.Volume(), 4.18879, 1e-5);
    // Sphere centered at origin — ellipsoid center should be near origin.
    EXPECT_TRUE(ebs.GetCenter().AllClose(
            core::Tensor::Zeros({3}, core::Float32, device),
            /*rtol=*/0.0, /*atol=*/1e-3));
    // Radius must be positive.
    EXPECT_TRUE(
    ebs.GetRadius()
          .To(core::Device("CPU:0"))
          .Item<double>() > 0.0);
}

TEST_P(BoundingSpherePermuteDevices,
       PointCloudGetBoundingSphereCoplanar) {
    core::Device device = GetParam();

    t::geometry::PointCloud pcd(core::Tensor::Init<float>({
            {1.0f, 0.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, -1.0f, 0.0f}},
            device));

    t::geometry::BoundingSphere ebs = pcd.GetBoundingSphere();

    EXPECT_TRUE(ebs.GetCenter().AllClose(
            core::Tensor::Zeros({3}, core::Float32, device),
            /*rtol=*/0.0, /*atol=*/1e-5));
    EXPECT_NEAR(
            ebs.GetRadius().To(core::Device("CPU:0")).Item<double>(),
            1.0, 1e-5);
}

}  // namespace tests
}  // namespace open3d