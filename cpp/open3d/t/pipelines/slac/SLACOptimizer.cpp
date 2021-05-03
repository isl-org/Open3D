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

#include "open3d/t/pipelines/slac/SLACOptimizer.h"

#include <fstream>
#include <set>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/ColoredICP.h"
#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

using PointCloud = open3d::geometry::PointCloud;
using TPointCloud = t::geometry::PointCloud;

/// Decoupled functions
static std::shared_ptr<PointCloud> PreprocessPointCloud(
        std::shared_ptr<PointCloud>& pcd, float voxel_size) {
    if (voxel_size > 0) {
        auto pcd_down = pcd->VoxelDownSample(voxel_size);
        pcd_down->RemoveStatisticalOutliers(20, 2.0);
        pcd_down->EstimateNormals();
        return pcd_down;
    } else {
        pcd->RemoveStatisticalOutliers(20, 2.0);
        if (!pcd->HasNormals()) {
            pcd->EstimateNormals();
        }
        return pcd;
    }
}

// The correspondences from NNS has the value as the target index
// correspondening the source point which is index of the value itself.
// Therefore, for target_indexed_correspondences = {2, 3, -1 , 4}.
// (source, target) correspondences are: {{0, 2}, {1, 3}, {3, 4}}.
//
// For convinience to access
// source and target pointcloud indexed by their correspondences, this
// function converts {N, 1} shaped target_indices correspondences to {C, 2}
// shaped CorrespondenceSet, where C is the number of correspondences such that
//
// For getting correspondence indexed pointclouds:
// source_indexed_pcd = source.GetPoints()
//                            .IndexGet({correspondence_set.T()[0]});
// target_indexed_pcd = target.GetPoints()
//                            .IndexGet({correspondence_set.T()[1]});
//
//  For getting the i-th correspondence pair:
// correspondence_pair_i = make_pair(correspondence[i][0],
//                                              correspondence[i][1]);
static core::Tensor ConvertCorrespondencesTargetIndexedToCx2Form(
        const core::Tensor& target_correspondences) {
    core::Device device = target_correspondences.GetDevice();
    int64_t N = target_correspondences.GetLength();

    core::Tensor valid_correspondences =
            target_correspondences.Ne(-1).Reshape({-1});

    // Only take valid indices.
    core::Tensor target_indices =
            target_correspondences.IndexGet({valid_correspondences});

    // Number of good correspondences (C).
    int64_t C = target_indices.GetLength();

    // correpondence_set : (i, corres[i]).
    // source[i] and target[corres[i]] is a correspondence.
    core::Tensor source_indices =
            core::Tensor::Arange(0, N, 1, core::Dtype::Int64, device)
                    .IndexGet({valid_correspondences})
                    .Reshape({C, 1});

    // Creating {C, 2} shaped tensor by horizontal stacking {source_indices,
    // target_indices}.
    core::Tensor correspondence_set({C, 2}, core::Dtype::Int64, device);
    correspondence_set.SetItem(
            {core::TensorKey::Slice(0, C, 1), core::TensorKey::Slice(0, 1, 1)},
            source_indices);
    correspondence_set.SetItem(
            {core::TensorKey::Slice(0, C, 1), core::TensorKey::Slice(1, 2, 1)},
            target_indices);

    return correspondence_set;
}

/// Aggressive pruning -- reject any suspicious pair
//
// tpcd_i is the source pointcloud, tpcd_j is the target pointcloud,
// T_i is the transformation_model_to_source,
// T_j is the transformation_model_to_target,
// T_ij is the transformation_source_to_target.
// distance_threshold is the search_distance for NNS.
// i and j are the indices of source and target pointcloud respectively.
core::Tensor GetCorrespondencesForPointCloudPair(int i,
                                                 int j,
                                                 TPointCloud& tpcd_i,
                                                 TPointCloud& tpcd_j,
                                                 const core::Tensor& T_i,
                                                 const core::Tensor& T_j,
                                                 const core::Tensor& T_ij,
                                                 float distance_threshold) {
    core::Device device = tpcd_i.GetDevice();
    core::Dtype dtype = tpcd_i.GetPoints().GetDtype();

    tpcd_j.GetPoints().AssertDevice(device);
    tpcd_j.GetPoints().AssertDtype(dtype);

    // TODO (@rishabh): AssertTransformation / IsTransformation.
    T_i.AssertShape({4, 4});
    T_j.AssertShape({4, 4});
    T_ij.AssertShape({4, 4});

    TPointCloud tpcd_i_transformed_Tij = tpcd_i.Clone();
    tpcd_i_transformed_Tij.Transform(T_ij.To(device, dtype));

    // Obtain correspondence via nns, between tpcd_i_transformed_Tij and tpcd_j.
    core::nns::NearestNeighborSearch tpcd_j_nns(tpcd_j.GetPoints());
    bool check = tpcd_j_nns.HybridIndex(distance_threshold);
    if (!check) {
        utility::LogError(
                "[NearestNeighborSearch::HybridSearch] Index is not set.");
    }
    core::Tensor target_indices, residual_distances_Tij;
    std::tie(target_indices, residual_distances_Tij) = tpcd_j_nns.HybridSearch(
            tpcd_i_transformed_Tij.GetPoints(), distance_threshold, 1);

    // Get the correspondence_set Transformed of shape {C, 2}.
    core::Tensor correspondence_set =
            ConvertCorrespondencesTargetIndexedToCx2Form(target_indices);

    // Get correspondence indexed pointcloud.
    TPointCloud tpcd_i_indexed(
            tpcd_i.GetPoints().IndexGet({correspondence_set.T()[0]}));
    TPointCloud tpcd_j_indexed(
            tpcd_j.GetPoints().IndexGet({correspondence_set.T()[1]}));

    // Inlier Ratio is calculated on pointclouds transformed by their pose in
    // model frame, to reject any suspicious pair.
    tpcd_i_indexed.Transform(T_i.To(device, dtype));
    tpcd_j_indexed.Transform(T_j.To(device, dtype));

    core::Tensor square_residual =
            (tpcd_i_indexed.GetPoints() - tpcd_j_indexed.GetPoints()).Sum({1});
    core::Tensor inliers =
            square_residual.Le(distance_threshold * distance_threshold);

    int64_t num_inliers =
            inliers.To(core::Dtype::Int64).Sum({0}).Item<int64_t>();

    float inlier_ratio = static_cast<float>(num_inliers) /
                         static_cast<float>(inliers.GetLength());

    utility::LogDebug("Tij and (Ti, Tj) compatibility ratio = {}.",
                      inlier_ratio);

    if (j != i + 1 && inlier_ratio < 0.3) {
        // Eigen::Matrix4d flip;
        // flip << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
        // auto pcd_i_corres = std::make_shared<open3d::geometry::PointCloud>(
        //         tpcd_i.Clone().Transform(T_i).ToLegacyPointCloud());
        // pcd_i_corres->PaintUniformColor({1, 0, 0});
        // pcd_i_corres->Transform(flip);
        // auto pcd_j_corres = std::make_shared<open3d::geometry::PointCloud>(
        //         tpcd_j.Clone().Transform(T_j).ToLegacyPointCloud());
        // pcd_j_corres->PaintUniformColor({0, 1, 0});
        // pcd_j_corres->Transform(flip);
        // std::vector<std::pair<int, int>> corres_lines;
        // for (size_t i = 0; i < result.correspondence_set_.size(); ++i) {
        //     corres_lines.push_back(
        //             std::make_pair(result.correspondence_set_[i](0),
        //                            result.correspondence_set_[i](1)));
        // }
        // auto lineset =
        //         open3d::geometry::LineSet::CreateFromPointCloudCorrespondences(
        //                 *pcd_i_corres, *pcd_j_corres, corres_lines);
        // lineset->PaintUniformColor({0, 0, 1});

        // visualization::DrawGeometries({pcd_i_corres, pcd_j_corres, lineset});

        return core::Tensor();
    }

    return correspondence_set;
}

/// Write point clouds after preprocessing (remove outliers, estimate normals,
/// etc).
static std::vector<std::string> PreprocessPointClouds(
        const std::vector<std::string>& fnames,
        const SLACOptimizerOption& option) {
    std::string subdir_name = option.GetSubfolderName();
    if (!subdir_name.empty()) {
        utility::filesystem::MakeDirectory(subdir_name);
    }

    std::vector<std::string> fnames_processed;

    for (auto& fname : fnames) {
        std::string fname_processed = fmt::format(
                "{}/{}", subdir_name,
                utility::filesystem::GetFileNameWithoutDirectory(fname));
        fnames_processed.emplace_back(fname_processed);
        if (utility::filesystem::FileExists(fname_processed)) continue;

        auto pcd = io::CreatePointCloudFromFile(fname);
        auto pcd_processed = PreprocessPointCloud(pcd, option.voxel_size_);

        io::WritePointCloud(fname_processed, *pcd_processed);
        utility::LogInfo("Saving processed point cloud {}", fname_processed);
    }

    return fnames_processed;
}

/// Read pose graph containing loop closures and odometry to compute
/// correspondences.
static void GetCorrespondencesForPointClouds(
        const std::vector<std::string>& fnames_processed,
        const PoseGraph& pose_graph,
        const SLACOptimizerOption& option) {
    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        utility::LogInfo("Processing {:02d} -> {:02d}", i, j);

        std::string correspondences_fname = fmt::format(
                "{}/{:03d}_{:03d}.npy", option.GetSubfolderName(), i, j);
        if (utility::filesystem::FileExists(correspondences_fname)) continue;

        auto tpcd_i = CreateTPCDFromFile(fnames_processed[i]);
        auto tpcd_j = CreateTPCDFromFile(fnames_processed[j]);

        auto pose_i = pose_graph.nodes_[i].pose_;
        auto pose_j = pose_graph.nodes_[j].pose_;
        auto pose_ij = edge.transformation_;

        core::Tensor T_i = core::eigen_converter::EigenMatrixToTensor(pose_i);
        core::Tensor T_j = core::eigen_converter::EigenMatrixToTensor(pose_j);
        core::Tensor T_ij = core::eigen_converter::EigenMatrixToTensor(pose_ij);

        // 0.008 ~ 3.0 / 512 * 1.4
        float distance_threshold =
                option.voxel_size_ < 0.001 ? 0.008 : 1.4 * option.voxel_size_;

        // Get correspondences betweem
        core::Tensor correspondence_set = GetCorrespondencesForPointCloudPair(
                i, j, tpcd_i, tpcd_j, T_i, T_j, T_ij, distance_threshold);

        if (correspondence_set.GetLength() > 0) {
            correspondence_set.Save(correspondences_fname);
            utility::LogInfo("Saving {} corres for {:02d} -> {:02d}",
                             correspondence_set.GetLength(), i, j);
        }
    }
}

static void InitializeControlGrid(ControlGrid& ctr_grid,
                                  const std::vector<std::string>& fnames,
                                  const SLACOptimizerOption& option) {
    core::Device device(option.device_);
    for (auto& fname : fnames) {
        utility::LogInfo("Initializing grid for {}", fname);

        auto tpcd = CreateTPCDFromFile(fname, device);
        ctr_grid.Touch(tpcd);
    }
    utility::LogInfo("Initialization finished.");
}

void FillInRigidAlignmentTerm(core::Tensor& AtA,
                              core::Tensor& Atb,
                              core::Tensor& residual,
                              TPointCloud& tpcd_i,
                              TPointCloud& tpcd_j,
                              core::Tensor& Ti,
                              core::Tensor& Tj,
                              core::Tensor& corres_ij,
                              int i,
                              int j,
                              float threshold,
                              core::Device& device,
                              bool visualize = false) {
    core::Dtype dtype = tpcd_i.GetPoints().GetDtype();

    TPointCloud tpcd_i_points_indexed(
            tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}).To(device, true));
    tpcd_i_points_indexed.SetPointNormals(
            tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}).To(device, true));

    TPointCloud tpcd_j_points_indexed(
            tpcd_j.GetPoints().IndexGet({corres_ij.T()[1]}).To(device, true));

    tpcd_i_points_indexed.Transform(Ti.To(device, dtype));
    tpcd_j_points_indexed.Transform(Tj.To(device, dtype));

    kernel::FillInRigidAlignmentTerm(
            AtA, Atb, residual, tpcd_i_points_indexed.GetPoints(),
            tpcd_j_points_indexed.GetPoints(),
            tpcd_i_points_indexed.GetPointNormals(), i, j, threshold);

    if (visualize) {
        // utility::LogInfo("{} -> {}", i, j);
        // Eigen::Matrix4d flip;
        // flip << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
        // auto pcd_i_corres = std::make_shared<open3d::geometry::PointCloud>(
        //         tpcd_i_points_indexed.ToLegacyPointCloud());
        // pcd_i_corres->PaintUniformColor({1, 0, 0});
        // pcd_i_corres->Transform(flip);
        // auto pcd_j_corres = std::make_shared<open3d::geometry::PointCloud>(
        //         tpcd_j_points_indexed.ToLegacyPointCloud());
        // pcd_j_corres->PaintUniformColor({0, 1, 0});
        // pcd_j_corres->Transform(flip);
        // std::vector<std::pair<int, int>> corres_lines;
        // for (size_t i = 0; i < pcd_i_corres->points_.size(); ++i) {
        //     corres_lines.push_back(std::make_pair(i, i));
        // }
        // auto lineset =
        //         open3d::geometry::LineSet::CreateFromPointCloudCorrespondences(
        //                 *pcd_i_corres, *pcd_j_corres, corres_lines);

        // visualization::DrawGeometries({pcd_i_corres, pcd_j_corres, lineset});
    }
}

void FillInRigidAlignmentTerm(core::Tensor& AtA,
                              core::Tensor& Atb,
                              core::Tensor& residual,
                              const std::vector<std::string>& fnames,
                              const PoseGraph& pose_graph,
                              const SLACOptimizerOption& option) {
    core::Device device(option.device_);

    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.npy",
                                               option.GetSubfolderName(), i, j);
        if (!utility::filesystem::FileExists(corres_fname)) {
            utility::LogWarning("Correspondence {} {} not processed!", i, j);
            continue;
        }

        auto tpcd_i = CreateTPCDFromFile(fnames[i]);
        auto tpcd_j = CreateTPCDFromFile(fnames[j]);

        auto Ti = core::eigen_converter::EigenMatrixToTensor(
                          pose_graph.nodes_[i].pose_)
                          .To(core::Dtype::Float32);
        auto Tj = core::eigen_converter::EigenMatrixToTensor(
                          pose_graph.nodes_[j].pose_)
                          .To(core::Dtype::Float32);

        auto corres_ij = core::Tensor::Load(corres_fname).To(device);

        FillInRigidAlignmentTerm(
                AtA, Atb, residual, tpcd_i, tpcd_j, Ti, Tj, corres_ij, i, j,
                option.threshold_, device,
                option.debug_enabled_ && i >= option.debug_start_idx_);
    }

    AtA.Save(fmt::format("{}/hessian.npy", option.GetSubfolderName()));
    Atb.Save(fmt::format("{}/residual.npy", option.GetSubfolderName()));
}

void FillInSLACAlignmentTerm(core::Tensor& AtA,
                             core::Tensor& Atb,
                             core::Tensor& residual,
                             ControlGrid& ctr_grid,
                             TPointCloud& tpcd_i,
                             TPointCloud& tpcd_j,
                             core::Tensor& Ti,
                             core::Tensor& Tj,
                             core::Tensor& Tij_icp,
                             core::Tensor& corres_ij,
                             int i,
                             int j,
                             int n_fragments,
                             float threshold,
                             core::Device& device,
                             bool visualize = false) {
    TPointCloud tpcd_i_points_indexed(device);
    tpcd_i_points_indexed.SetPoints(
            tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}).To(device));
    tpcd_i_points_indexed.SetPointNormals(
            tpcd_i.GetPointNormals().IndexGet({corres_ij.T()[0]}).To(device));

    TPointCloud tpcd_j_points_indexed(device);
    tpcd_j_points_indexed.SetPoints(
            tpcd_j.GetPoints().IndexGet({corres_ij.T()[1]}).To(device));
    tpcd_j_points_indexed.SetPointNormals(
            tpcd_j.GetPointNormals().IndexGet({corres_ij.T()[1]}).To(device));

    //     Parameterize points in the control grid
    auto tpcd_param_i = ctr_grid.Parameterize(tpcd_i_points_indexed);
    auto tpcd_param_j = ctr_grid.Parameterize(tpcd_j_points_indexed);

    // Parameterize: setup point cloud -> cgrid correspondences
    auto cgrid_index_ps =
            tpcd_param_i.GetPointAttr(ControlGrid::kAttrNbGridIdx).To(device);
    auto cgrid_ratio_ps =
            tpcd_param_i.GetPointAttr(ControlGrid::kAttrNbGridPointInterp)
                    .To(device);

    auto cgrid_index_qs =
            tpcd_param_j.GetPointAttr(ControlGrid::kAttrNbGridIdx).To(device);
    auto cgrid_ratio_qs =
            tpcd_param_j.GetPointAttr(ControlGrid::kAttrNbGridPointInterp)
                    .To(device);

    // Warp with control grids
    auto tpcd_nonrigid_i = ctr_grid.Warp(tpcd_param_i);
    auto tpcd_nonrigid_j = ctr_grid.Warp(tpcd_param_j);

    // TODO: Put the following pre-processing inside the FillInSlACAlignmentTerm
    // kernel, if possible.
    auto Cps = tpcd_nonrigid_i.GetPoints();
    auto Cqs = tpcd_nonrigid_j.GetPoints();
    auto Cnormal_ps = tpcd_nonrigid_i.GetPointNormals();

    auto Ri = Ti.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
    auto ti = Ti.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

    auto Rj = Tj.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
    auto tj = Tj.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

    // Transform for required entries
    auto Ti_Cps = (Ri.Matmul(Cps.T())).Add_(ti).T().Contiguous();
    auto Tj_Cqs = (Rj.Matmul(Cqs.T())).Add_(tj).T().Contiguous();
    auto Ri_Cnormal_ps = (Ri.Matmul(Cnormal_ps.T())).T().Contiguous();
    auto RjT_Ri_Cnormal_ps =
            (Rj.T().Matmul(Ri_Cnormal_ps.T())).T().Contiguous();

    kernel::FillInSLACAlignmentTerm(
            AtA, Atb, residual, Ti_Cps, Tj_Cqs, Cnormal_ps, Ri_Cnormal_ps,
            RjT_Ri_Cnormal_ps, cgrid_index_ps, cgrid_index_qs, cgrid_ratio_ps,
            cgrid_ratio_qs, i, j, n_fragments, threshold);

    if (visualize) {
        utility::LogInfo("edge {} -> {}", i, j);
        VisualizePCDCorres(tpcd_i, tpcd_j, tpcd_param_i, tpcd_param_j, Tij_icp);
        VisualizePCDCorres(tpcd_i, tpcd_j, tpcd_nonrigid_i, tpcd_nonrigid_j,
                           Tij_icp);

        VisualizeWarp(tpcd_param_i, ctr_grid);
        VisualizeWarp(tpcd_param_j, ctr_grid);

        VisualizePCDGridCorres(tpcd_param_i, ctr_grid);
        VisualizePCDGridCorres(tpcd_param_j, ctr_grid);
    }
}

void FillInSLACAlignmentTerm(core::Tensor& AtA,
                             core::Tensor& Atb,
                             core::Tensor& residual,
                             ControlGrid& ctr_grid,
                             const std::vector<std::string>& fnames,
                             const PoseGraph& pose_graph,
                             const SLACOptimizerOption& option) {
    core::Device device(option.device_);
    int n_frags = pose_graph.nodes_.size();

    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        // Load poses
        auto Ti = core::eigen_converter::EigenMatrixToTensor(
                          pose_graph.nodes_[i].pose_)
                          .To(device, core::Dtype::Float32);
        auto Tj = core::eigen_converter::EigenMatrixToTensor(
                          pose_graph.nodes_[j].pose_)
                          .To(device, core::Dtype::Float32);
        auto Tij =
                core::eigen_converter::EigenMatrixToTensor(edge.transformation_)
                        .To(device, core::Dtype::Float32);

        // Load point clouds
        auto tpcd_i = CreateTPCDFromFile(fnames[i], device);
        auto tpcd_j = CreateTPCDFromFile(fnames[j], device);

        // Load correspondences and select points and normals
        std::string correspondences_fname = fmt::format(
                "{}/{:03d}_{:03d}.npy", option.GetSubfolderName(), i, j);
        if (!utility::filesystem::FileExists(correspondences_fname)) {
            utility::LogWarning("Correspondence {} {} not processed!", i, j);
            continue;
        }
        core::Tensor correspondences_ij =
                core::Tensor::Load(correspondences_fname).To(device);

        // Fill In
        FillInSLACAlignmentTerm(
                AtA, Atb, residual, ctr_grid, tpcd_i, tpcd_j, Ti, Tj, Tij,
                correspondences_ij, i, j, n_frags, option.threshold_, device,
                option.debug_enabled_ && i >= option.debug_start_idx_);
    }
    AtA.Save(fmt::format("{}/hessian.npy", option.GetSubfolderName()));
    Atb.Save(fmt::format("{}/residual.npy", option.GetSubfolderName()));
}

void FillInSLACRegularizerTerm(core::Tensor& AtA,
                               core::Tensor& Atb,
                               core::Tensor& residual,
                               ControlGrid& ctr_grid,
                               int n_frags,
                               const SLACOptimizerOption& option) {
    core::Tensor active_addrs, nb_addrs, nb_masks;
    std::tie(active_addrs, nb_addrs, nb_masks) = ctr_grid.GetNeighborGridMap();

    core::Tensor positions_init = ctr_grid.GetInitPositions();
    core::Tensor positions_curr = ctr_grid.GetCurrPositions();
    kernel::FillInSLACRegularizerTerm(
            AtA, Atb, residual, active_addrs, nb_addrs, nb_masks,
            positions_init, positions_curr, n_frags * option.regularizor_coeff_,
            n_frags, ctr_grid.anchor_idx_);
    AtA.Save(fmt::format("{}/hessian_regularized.npy",
                         option.GetSubfolderName()));
}

void UpdatePoses(PoseGraph& fragment_pose_graph,
                 core::Tensor& delta,
                 const SLACOptimizerOption& option) {
    core::Tensor delta_poses = delta.View({-1, 6}).To(core::Device("CPU:0"));

    if (delta_poses.GetLength() != int64_t(fragment_pose_graph.nodes_.size())) {
        utility::LogError("Dimension Mismatch");
    }
    for (int64_t i = 0; i < delta_poses.GetLength(); ++i) {
        core::Tensor pose_delta = kernel::PoseToTransformation(delta_poses[i]);
        core::Tensor pose_tensor =
                pose_delta.Matmul(core::eigen_converter::EigenMatrixToTensor(
                                          fragment_pose_graph.nodes_[i].pose_)
                                          .To(core::Dtype::Float32));

        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> pose_eigen =
                core::eigen_converter::TensorToEigenMatrixXf(pose_tensor);
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> pose_eigen_d =
                pose_eigen.cast<double>().eval();
        Eigen::Ref<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> pose_eigen_ref(
                pose_eigen_d);
        fragment_pose_graph.nodes_[i].pose_ = pose_eigen_ref;
    }
}

void UpdateControlGrid(ControlGrid& ctr_grid,
                       core::Tensor& delta,
                       const SLACOptimizerOption& option) {
    core::Tensor delta_cgrids = delta.View({-1, 3});
    if (delta_cgrids.GetLength() != int64_t(ctr_grid.Size())) {
        utility::LogError("Dimension Mismatch");
    }

    ctr_grid.GetCurrPositions().Slice(0, 0, ctr_grid.Size()) += delta_cgrids;
}

std::pair<PoseGraph, ControlGrid> RunSLACOptimizerForFragments(
        const std::vector<std::string>& fnames,
        const PoseGraph& pose_graph,
        SLACOptimizerOption& option) {
    core::Device device(option.device_);
    if (!option.buffer_folder_.empty()) {
        utility::filesystem::MakeDirectory(option.buffer_folder_);
    }

    // First preprocess the point cloud with downsampling and normal
    // estimation.
    auto fnames_down = PreprocessPointClouds(fnames, option);
    // Then obtain the correspondences given the pose graph
    GetCorrespondencesForPointClouds(fnames_down, pose_graph, option);

    // First initialize ctr_grid
    ControlGrid ctr_grid(3.0 / 8, 8000, device);
    InitializeControlGrid(ctr_grid, fnames_down, option);
    utility::LogInfo("Compactifying");
    ctr_grid.Compactify();
    utility::LogInfo("Compactifying finished");

    // Fill-in
    // fragments x 6 (se3) + control_grids x 3 (R^3)
    int64_t num_params = fnames_down.size() * 6 + ctr_grid.Size() * 3;
    utility::LogInfo("Initializing {}^2 matrices", num_params);

    PoseGraph pose_graph_update(pose_graph);
    for (int itr = 0; itr < option.max_iterations_; ++itr) {
        option.debug_enabled_ = option.debug_ && itr >= option.debug_start_itr_;

        utility::LogInfo("Iteration {}", itr);
        core::Tensor AtA = core::Tensor::Zeros({num_params, num_params},
                                               core::Dtype::Float32, device);
        core::Tensor Atb = core::Tensor::Zeros({num_params, 1},
                                               core::Dtype::Float32, device);

        core::Tensor indices_eye0 =
                core::Tensor::Arange(0, 6, 1, core::Dtype::Int64, device);
        AtA.IndexSet({indices_eye0, indices_eye0},
                     core::Tensor::Ones({}, core::Dtype::Float32, device));

        core::Tensor residual_data =
                core::Tensor::Zeros({1}, core::Dtype::Float32, device);
        FillInSLACAlignmentTerm(AtA, Atb, residual_data, ctr_grid, fnames_down,
                                pose_graph_update, option);
        utility::LogInfo("Residual Data = {}", residual_data[0].Item<float>());

        core::Tensor residual_reg =
                core::Tensor::Zeros({1}, core::Dtype::Float32, device);
        FillInSLACRegularizerTerm(AtA, Atb, residual_reg, ctr_grid,
                                  pose_graph_update.nodes_.size(), option);
        utility::LogInfo("Residual Reg = {}", residual_reg[0].Item<float>());

        core::Tensor delta = AtA.Solve(Atb.Neg());

        core::Tensor delta_poses =
                delta.Slice(0, 0, 6 * pose_graph_update.nodes_.size());
        core::Tensor delta_cgrids = delta.Slice(
                0, 6 * pose_graph_update.nodes_.size(), delta.GetLength());

        UpdatePoses(pose_graph_update, delta_poses, option);
        UpdateControlGrid(ctr_grid, delta_cgrids, option);
        VisualizeRegularizor(ctr_grid);
    }
    return std::make_pair(pose_graph_update, ctr_grid);
}

PoseGraph RunRigidOptimizerForFragments(const std::vector<std::string>& fnames,
                                        const PoseGraph& pose_graph,
                                        SLACOptimizerOption& option) {
    core::Device device(option.device_);
    if (!option.buffer_folder_.empty()) {
        utility::filesystem::MakeDirectory(option.buffer_folder_);
    }

    // First preprocess the point cloud with downsampling and normal
    // estimation.
    std::vector<std::string> fnames_down =
            PreprocessPointClouds(fnames, option);

    // Then obtain the correspondences given the pose graph and save them on
    // disk.
    GetCorrespondencesForPointClouds(fnames_down, pose_graph, option);

    // Fill-in
    // fragments x 6 (se3)
    int64_t num_params = fnames_down.size() * 6;
    utility::LogInfo("Initializing {}^2 Hessian matrix", num_params);

    PoseGraph pose_graph_update(pose_graph);
    for (int itr = 0; itr < option.max_iterations_; ++itr) {
        option.debug_enabled_ = option.debug_ && itr >= option.debug_start_itr_;

        utility::LogInfo("Iteration {}", itr);
        core::Tensor AtA = core::Tensor::Zeros({num_params, num_params},
                                               core::Dtype::Float32, device);
        core::Tensor Atb = core::Tensor::Zeros({num_params, 1},
                                               core::Dtype::Float32, device);
        core::Tensor residual =
                core::Tensor::Zeros({1}, core::Dtype::Float32, device);

        // Fix pose 0
        core::Tensor indices_eye0 = core::Tensor::Arange(0, 6, 1);
        AtA.IndexSet(
                {indices_eye0, indices_eye0},
                1e5 * core::Tensor::Ones({}, core::Dtype::Float32, device));

        FillInRigidAlignmentTerm(AtA, Atb, residual, fnames_down,
                                 pose_graph_update, option);
        utility::LogInfo("Residual = {}", residual[0].Item<float>());

        core::Tensor delta = AtA.Solve(Atb.Neg());
        UpdatePoses(pose_graph_update, delta, option);
    }

    return pose_graph_update;
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
