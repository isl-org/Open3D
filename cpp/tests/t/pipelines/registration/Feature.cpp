// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Feature.h"

#include "core/CoreTest.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Feature.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class FeaturePermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Feature,
                         FeaturePermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(FeaturePermuteDevices, ComputeFPFHFeature) {
    core::Device device = GetParam();

    open3d::geometry::PointCloud pcd_legacy;
    data::BunnyMesh byunny;
    open3d::io::ReadPointCloud(byunny.GetPath(), pcd_legacy);

    pcd_legacy.EstimateNormals();
    // Convert to float64 to avoid precision loss.
    const auto pcd = t::geometry::PointCloud::FromLegacy(pcd_legacy,
                                                         core::Float64, device);

    const auto fpfh = pipelines::registration::ComputeFPFHFeature(
            pcd_legacy, geometry::KDTreeSearchParamHybrid(0.01, 100));
    const auto fpfh_t =
            t::pipelines::registration::ComputeFPFHFeature(pcd, 100, 0.01);

    EXPECT_TRUE(fpfh_t.AllClose(
            core::eigen_converter::EigenMatrixToTensor(fpfh->data_)
                    .T()
                    .To(fpfh_t.GetDevice(), fpfh_t.GetDtype()),
            1e-4, 1e-4));
}

}  // namespace tests
}  // namespace open3d
