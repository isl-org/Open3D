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
#include "open3d/geometry/KDTreeFlann.h"
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

TEST_P(FeaturePermuteDevices, ToyCheckCorrespondencesFromFeatures) {
    core::Device device = GetParam();

    int feat_len = 32;
    int feat_dim = 32;

    // Dummy feature at 0-th dimension
    core::Tensor arange_indices =
            core::Tensor::Arange(0, feat_len, 1, core::Int64, device);
    core::Tensor src({feat_len, feat_dim}, core::Float32, device);
    src.SetItem({core::TensorKey::Slice(core::None, core::None, core::None),
                 core::TensorKey::Index(0)},
                arange_indices.To(core::Float32));

    // Feature matching to itself, should be identity
    core::Tensor correspondences =
            t::pipelines::registration::CorrespondencesFromFeatures(src, src);

    // correspondences[:, 0] = arange(0, feat_len) [self]
    // correspondences[:, 1] = arange(0, feat_len) [1nn matched to self]
    EXPECT_TRUE(arange_indices.AllClose(correspondences.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Index(0)})));
    EXPECT_TRUE(arange_indices.AllClose(correspondences.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Index(1)})));
}

TEST_P(FeaturePermuteDevices, DemoCheckCorrespondencesFromFeatures) {
    core::Device device = GetParam();

    t::geometry::PointCloud source_tpcd, target_tpcd;
    data::DemoICPPointClouds pcd_fragments;
    t::io::ReadPointCloud(pcd_fragments.GetPaths()[0], source_tpcd);
    t::io::ReadPointCloud(pcd_fragments.GetPaths()[1], target_tpcd);

    source_tpcd = source_tpcd.To(device);
    target_tpcd = target_tpcd.To(device);

    source_tpcd.EstimateNormals();
    target_tpcd.EstimateNormals();

    const int knn = 100;
    const float radius = 0.05;

    // source fpfh
    const auto source_tfpfh = t::pipelines::registration::ComputeFPFHFeature(
            source_tpcd, knn, radius);
    const auto target_tfpfh = t::pipelines::registration::ComputeFPFHFeature(
            target_tpcd, knn, radius);

    // target pcd and fpfh
    auto source_pcd = source_tpcd.ToLegacy();
    auto target_pcd = target_tpcd.ToLegacy();

    auto source_fpfh = pipelines::registration::ComputeFPFHFeature(
            source_pcd, geometry::KDTreeSearchParamHybrid(radius, knn));
    auto target_fpfh = pipelines::registration::ComputeFPFHFeature(
            target_pcd, geometry::KDTreeSearchParamHybrid(radius, knn));

    // FPFH consistency check
    auto source_diff =
            (source_tfpfh -
             core::eigen_converter::EigenMatrixToTensor(source_fpfh->data_)
                     .T()
                     .To(source_tfpfh.GetDevice(), source_tfpfh.GetDtype()))
                    .Abs();
    auto target_diff =
            (target_tfpfh -
             core::eigen_converter::EigenMatrixToTensor(target_fpfh->data_)
                     .T()
                     .To(target_tfpfh.GetDevice(), target_tfpfh.GetDtype()))
                    .Abs();
    // At a large scale FPFH could be a bit different

    // FPFH has a large magnitude
    float source_consistency_ratio = source_diff.Sum({1})
                                             .Lt(1.0)
                                             .To(core::Float32)
                                             .Sum({0})
                                             .Item<float>() /
                                     source_diff.GetLength();
    float target_consistency_ratio = target_diff.Sum({1})
                                             .Lt(1.0)
                                             .To(core::Float32)
                                             .Sum({0})
                                             .Item<float>() /
                                     target_diff.GetLength();

    // TODO: fix ispc conversion issue (?)
    float factor = source_consistency_ratio > 1.0 ? (1.0 / 255.0) : 1.0;
    source_consistency_ratio *= factor;
    target_consistency_ratio *= factor;

    EXPECT_NEAR(source_consistency_ratio, 1.0, 1e-2);
    EXPECT_NEAR(target_consistency_ratio, 1.0, 1e-2);

    // Compute correspondences
    auto tcorrespondences =
            t::pipelines::registration::CorrespondencesFromFeatures(
                    source_tfpfh, target_tfpfh);

    // No counter part exists in legacy, copied from legacy code
    int num_src_pts = int(source_pcd.points_.size());

    geometry::KDTreeFlann kdtree_target(*target_fpfh);

    std::vector<int64_t> corres_j_vec;

    for (int i = 0; i < num_src_pts; i++) {
        std::vector<int> corres_tmp(1);
        std::vector<double> dist_tmp(1);

        kdtree_target.SearchKNN(Eigen::VectorXd(source_fpfh->data_.col(i)), 1,
                                corres_tmp, dist_tmp);
        int j = corres_tmp[0];
        corres_j_vec.push_back(j);
    }

    core::Tensor corres_j(corres_j_vec, {num_src_pts}, core::Int64, device);

    // Check consistency
    auto equivalence = corres_j.Eq(tcorrespondences.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Index(1)}));
    float consistency_ratio =
            equivalence.To(core::Float32).Sum({0}).Item<float>() / num_src_pts;
    consistency_ratio *= factor;
    utility::LogInfo("correspondence consistency_ratio: {}", consistency_ratio);
}
}  // namespace tests
}  // namespace open3d
