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

TEST_P(FeaturePermuteDevices, CorrespondencesFromFeatures) {
    core::Device device = GetParam();

    const float kVoxelSize = 0.05f;
    const float kFPFHRadius = kVoxelSize * 5;

    t::geometry::PointCloud source_tpcd, target_tpcd;
    data::DemoICPPointClouds pcd_fragments;
    t::io::ReadPointCloud(pcd_fragments.GetPaths()[0], source_tpcd);
    t::io::ReadPointCloud(pcd_fragments.GetPaths()[1], target_tpcd);
    source_tpcd = source_tpcd.To(device).VoxelDownSample(kVoxelSize);
    target_tpcd = target_tpcd.To(device).VoxelDownSample(kVoxelSize);

    auto t_source_fpfh = t::pipelines::registration::ComputeFPFHFeature(
            source_tpcd, 100, kFPFHRadius);
    auto t_target_fpfh = t::pipelines::registration::ComputeFPFHFeature(
            target_tpcd, 100, kFPFHRadius);

    pipelines::registration::Feature source_fpfh, target_fpfh;
    source_fpfh.data_ =
            core::eigen_converter::TensorToEigenMatrixXd(t_source_fpfh.T());
    target_fpfh.data_ =
            core::eigen_converter::TensorToEigenMatrixXd(t_target_fpfh.T());

    for (auto mutual_filter : std::vector<bool>{true, false}) {
        auto t_correspondences =
                t::pipelines::registration::CorrespondencesFromFeatures(
                        t_source_fpfh, t_target_fpfh, mutual_filter);

        auto correspondences =
                pipelines::registration::CorrespondencesFromFeatures(
                        source_fpfh, target_fpfh, mutual_filter);

        auto t_correspondence_idx =
                t_correspondences.T().GetItem(core::TensorKey::Index(1));
        auto correspondence_idx =
                core::eigen_converter::EigenVector2iVectorToTensor(
                        correspondences, core::Dtype::Int64, device)
                        .T()
                        .GetItem(core::TensorKey::Index(1));

        // TODO(wei): mask.to(float).sum() has ISPC issues. Use advanced
        // indexing instead.
        if (!mutual_filter) {
            auto mask = t_correspondence_idx.Eq(correspondence_idx);
            auto masked_idx = t_correspondence_idx.IndexGet({mask});
            float valid_ratio = float(masked_idx.GetLength()) /
                                float(t_correspondence_idx.GetLength());
            EXPECT_NEAR(valid_ratio, 1.0, 1e-2);
        } else {
            auto consistent_ratio = float(t_correspondence_idx.GetLength()) /
                                    float(correspondences.size());
            EXPECT_NEAR(consistent_ratio, 1.0, 1e-2);
        }
    }
}

}  // namespace tests
}  // namespace open3d
