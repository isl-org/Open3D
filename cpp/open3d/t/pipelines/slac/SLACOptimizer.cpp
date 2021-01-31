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
#include "open3d/geometry/LineSet.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/Draw.h"
#include "open3d/visualization/utility/DrawGeometry.h"
namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

Eigen::Vector3d Jet(double v, double vmin, double vmax) {
    Eigen::Vector3d c(1, 1, 1);
    double dv;

    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c(0) = 0;
        c(1) = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c(0) = 0;
        c(2) = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c(0) = 4 * (v - vmin - 0.5 * dv) / dv;
        c(2) = 0;
    } else {
        c(1) = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c(2) = 0;
    }

    return c;
}

t::geometry::PointCloud CreateTPCDFromFile(
        const std::string& fname,
        const core::Device& device = core::Device("CPU:0")) {
    auto pcd = io::CreatePointCloudFromFile(fname);
    return t::geometry::PointCloud::FromLegacyPointCloud(
            *pcd, core::Dtype::Float32, device);
}

void VisualizePCDCorres(t::geometry::PointCloud& tpcd_i,
                        t::geometry::PointCloud& tpcd_j,
                        core::Tensor& corres,
                        const Eigen::Matrix4d& pose_ij) {
    auto pcd_i = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_i.ToLegacyPointCloud());
    auto pcd_j = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_j.ToLegacyPointCloud());
    pcd_i->Transform(pose_ij);

    std::vector<std::pair<int, int>> corres_lines;
    for (int64_t i = 0; i < corres.GetLength(); ++i) {
        std::pair<int, int> pair = {corres[i][0].Item<int64_t>(),
                                    corres[i][1].Item<int64_t>()};
        corres_lines.push_back(pair);
    }
    auto lineset =
            open3d::geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *pcd_i, *pcd_j, corres_lines);
    lineset->PaintUniformColor({0, 1, 0});
    visualization::DrawGeometries({pcd_i, pcd_j, lineset});
}

void VisualizePCDGridCorres(t::geometry::PointCloud& tpcd,
                            t::geometry::PointCloud& tpcd_param,
                            ControlGrid& ctr_grid) {
    // Prepare all ctr grid point cloud for lineset
    auto pcd = std::make_shared<open3d::geometry::PointCloud>(
            tpcd.ToLegacyPointCloud());

    t::geometry::PointCloud tpcd_grid(ctr_grid.GetPositions());
    auto pcd_grid = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_grid.ToLegacyPointCloud());

    // Prepare n x 8 corres for visualization
    core::Tensor corres = tpcd_param.GetPointAttr(ControlGrid::kAttrNbGridIdx)
                                  .To(core::Device("CPU:0"));
    std::vector<std::pair<int, int>> corres_lines;
    for (int64_t i = 0; i < corres.GetLength(); ++i) {
        for (int k = 0; k < 8; ++k) {
            std::pair<int, int> pair = {i, corres[i][k].Item<int>()};
            corres_lines.push_back(pair);
        }
    }
    auto lineset =
            open3d::geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *pcd, *pcd_grid, corres_lines);

    core::Tensor corres_interp =
            tpcd_param.GetPointAttr(ControlGrid::kAttrNbGridPointInterp)
                    .To(core::Device("CPU:0"));
    for (int64_t i = 0; i < corres.GetLength(); ++i) {
        for (int k = 0; k < 8; ++k) {
            float ratio = corres_interp[i][k].Item<float>();
            Eigen::Vector3d color = Jet(ratio, 0, 0.5);
            lineset->colors_.push_back(color);
        }
    }

    // Prepare nb point cloud for visualization
    t::geometry::PointCloud tpcd_grid_nb(tpcd_grid.GetPoints().IndexGet(
            {corres.View({-1}).To(core::Dtype::Int64)}));
    auto pcd_grid_nb = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_grid_nb.ToLegacyPointCloud());
    pcd_grid_nb->PaintUniformColor({1, 0, 0});

    visualization::DrawGeometries({lineset, pcd, pcd_grid_nb});
    visualization::DrawGeometries({pcd});
}

void VisualizeWarp(const geometry::PointCloud& tpcd_param,
                   ControlGrid& ctr_grid) {
    auto pcd = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_param.ToLegacyPointCloud());
    pcd->PaintUniformColor({1, 0, 0});

    auto tpcd_warped = ctr_grid.Warp(tpcd_param);
    auto pcd_warped = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_warped.ToLegacyPointCloud());
    pcd_warped->PaintUniformColor({0, 1, 0});
    visualization::DrawGeometries({pcd, pcd_warped});
}

/// Write point clouds after downsampling and normal estimation for
/// correspondence check.
std::vector<std::string> PreprocessPointClouds(
        const std::vector<std::string>& fragment_fnames,
        const SLACOptimizerOption& option) {
    std::string subdir_name = option.GetSubfolderName();
    if (!subdir_name.empty()) {
        utility::filesystem::MakeDirectory(subdir_name);
    }

    std::vector<std::string> fnames_down;
    for (auto& fname : fragment_fnames) {
        std::string fname_down = fmt::format(
                "{}/{}", subdir_name,
                utility::filesystem::GetFileNameWithoutDirectory(fname));
        fnames_down.emplace_back(fname_down);

        if (utility::filesystem::FileExists(fname_down)) continue;
        utility::LogInfo("Processing {}", fname_down);

        auto pcd = io::CreatePointCloudFromFile(fname);
        if (option.voxel_size_ > 0) {
            auto pcd_down = pcd->VoxelDownSample(option.voxel_size_);
            pcd_down->EstimateNormals();
            io::WritePointCloud(fname_down, *pcd_down);
        } else if (!pcd->HasNormals()) {
            pcd->EstimateNormals();
            io::WritePointCloud(fname_down, *pcd);
        }
    }

    return fnames_down;
}

/// Read pose graph containing loop closures and odometry to compute
/// correspondences.
void GetCorrespondencesForPointClouds(
        const std::vector<std::string>& fragment_down_fnames,
        const PoseGraph& pose_graph,
        const SLACOptimizerOption& option) {
    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.npy",
                                               option.GetSubfolderName(), i, j);
        if (utility::filesystem::FileExists(corres_fname)) continue;

        auto tpcd_i = CreateTPCDFromFile(fragment_down_fnames[i]);
        auto tpcd_j = CreateTPCDFromFile(fragment_down_fnames[j]);

        auto pose_i = pose_graph.nodes_[i].pose_;
        auto pose_j = pose_graph.nodes_[j].pose_;
        auto pose_ij = (pose_j.inverse() * pose_i).eval();

        // Obtain correspondence via nns
        auto result = t::pipelines::registration::EvaluateRegistration(
                tpcd_i, tpcd_j,
                option.voxel_size_ < 0 ? 0.05 : option.voxel_size_,
                core::eigen_converter::EigenMatrixToTensor(pose_ij).To(
                        core::Dtype::Float32));

        // Make correspondence indices (N x 2)
        core::Tensor corres =
                core::Tensor({2, result.correspondence_set_.GetLength()},
                             core::Dtype::Int64);
        core::Tensor indices = core::Tensor::Arange(
                0, result.correspondence_select_bool_.GetLength(), 1,
                core::Dtype::Int64);
        corres.SetItem({core::TensorKey::Index(0)},
                       indices.IndexGet({result.correspondence_select_bool_}));
        corres.SetItem({core::TensorKey::Index(1)}, result.correspondence_set_);
        corres = corres.T();
        corres.Save(corres_fname);
        utility::LogInfo("Saving {} corres for {:02d} -> {:02d}",
                         corres.GetLength(), i, j);
    }
}

void InitializeControlGrid(ControlGrid& ctr_grid,
                           const std::vector<std::string>& fnames,
                           const SLACOptimizerOption& option) {
    core::Device device(option.device_);
    for (auto& fname : fnames) {
        utility::LogInfo("Initializing grid for {}", fname);

        auto tpcd = CreateTPCDFromFile(fname, device);
        ctr_grid.Touch(tpcd);
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

        utility::LogInfo("edge {} -> {}", i, j);
        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.npy",
                                               option.GetSubfolderName(), i, j);
        if (!utility::filesystem::FileExists(corres_fname)) {
            utility::LogError("Correspondence not processed");
        }

        // Load poses
        auto Ti_e = pose_graph.nodes_[i].pose_;
        auto Tj_e = pose_graph.nodes_[j].pose_;
        auto Tij_e = (Tj_e.inverse() * Ti_e).eval();
        auto Ti = core::eigen_converter::EigenMatrixToTensor(Ti_e).To(
                device, core::Dtype::Float32);
        auto Tj = core::eigen_converter::EigenMatrixToTensor(Tj_e).To(
                device, core::Dtype::Float32);

        auto Ri = Ti.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
        auto ti = Ti.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

        auto Rj = Tj.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
        auto tj = Tj.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

        // Load point clouds
        auto tpcd_i = CreateTPCDFromFile(fnames[i], device);
        auto tpcd_j = CreateTPCDFromFile(fnames[j], device);

        // Load correspondences and select points and normals
        utility::LogInfo("Load correspondences");
        core::Tensor corres_ij = core::Tensor::Load(corres_fname).To(device);
        auto ps = tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}).To(device);
        auto qs = tpcd_j.GetPoints().IndexGet({corres_ij.T()[1]}).To(device);
        auto normal_ps = tpcd_i.GetPointNormals()
                                 .IndexGet({corres_ij.T()[0]})
                                 .To(device);
        auto normal_qs = tpcd_j.GetPointNormals()
                                 .IndexGet({corres_ij.T()[1]})
                                 .To(device);

        // Parameterize points in the control grid
        utility::LogInfo("Parameterize");
        auto tpcd_param_i = ctr_grid.Parameterize(t::geometry::PointCloud(
                {{"points", ps}, {"normals", normal_ps}}));
        auto tpcd_param_j = ctr_grid.Parameterize(t::geometry::PointCloud(
                {{"points", qs}, {"normals", normal_qs}}));

        // Parameterize: setup point cloud -> cgrid correspondences
        utility::LogInfo("Obtain nn info");
        auto cgrid_index_ps =
                tpcd_param_i.GetPointAttr(ControlGrid::kAttrNbGridIdx)
                        .To(device);
        auto cgrid_ratio_ps =
                tpcd_param_i.GetPointAttr(ControlGrid::kAttrNbGridPointInterp)
                        .To(device);

        auto cgrid_index_qs =
                tpcd_param_j.GetPointAttr(ControlGrid::kAttrNbGridIdx)
                        .To(device);
        auto cgrid_ratio_qs =
                tpcd_param_j.GetPointAttr(ControlGrid::kAttrNbGridPointInterp)
                        .To(device);

        // Warp with control grids
        utility::LogInfo("Warp");
        auto tpcd_nonrigid_i = ctr_grid.Warp(tpcd_param_i);
        auto tpcd_nonrigid_j = ctr_grid.Warp(tpcd_param_j);

        utility::LogInfo("Warped attributes");
        auto Cps = tpcd_nonrigid_i.GetPoints();
        auto Cqs = tpcd_nonrigid_j.GetPoints();
        auto Cnormal_ps = tpcd_nonrigid_i.GetPointNormals();

        // Transform for required entries
        utility::LogInfo("Preprocess");
        auto Ti_Cps = (Ri.Matmul(Cps.T())).Add_(ti).T().Contiguous();
        auto Tj_Cqs = (Rj.Matmul(Cqs.T())).Add_(tj).T().Contiguous();
        auto Ri_Cnormal_ps = (Ri.Matmul(Cnormal_ps.T())).T().Contiguous();
        auto RjT_Ri_Cnormal_ps =
                (Rj.T().Matmul(Ri_Cnormal_ps.T())).T().Contiguous();

        utility::LogInfo("Fill in");
        kernel::FillInSLACAlignmentTerm(
                AtA, Atb, residual, Ti_Cps, Tj_Cqs, Cnormal_ps, Ri_Cnormal_ps,
                RjT_Ri_Cnormal_ps, cgrid_index_ps, cgrid_index_qs,
                cgrid_ratio_ps, cgrid_ratio_qs, i, j, n_frags);

        if (option.grid_debug_) {
            VisualizeWarp(tpcd_param_i, ctr_grid);
            VisualizeWarp(tpcd_param_j, ctr_grid);

            VisualizePCDCorres(tpcd_i, tpcd_j, corres_ij, Tij_e);
            VisualizePCDGridCorres(tpcd_i, tpcd_param_i, ctr_grid);
            VisualizePCDGridCorres(tpcd_j, tpcd_param_j, ctr_grid);
        }

        // TODO: use parameterization to update normals and points per grid
    }
    AtA.Save(fmt::format("{}/hessian.npy", option.GetSubfolderName()));
    Atb.Save(fmt::format("{}/residual.npy", option.GetSubfolderName()));
}

void FillInSLACRegularizerTerm(core::Tensor& AtA,
                               core::Tensor& Atb,
                               ControlGrid& ctr_grid,
                               const SLACOptimizerOption& option) {
    utility::LogError("Unimplemented.");
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
            utility::LogError("Correspondence not processed!");
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

        auto ps = tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}).To(device);
        auto qs = tpcd_j.GetPoints().IndexGet({corres_ij.T()[1]}).To(device);

        auto normal_ps = tpcd_i.GetPointNormals()
                                 .IndexGet({corres_ij.T()[0]})
                                 .To(device);

        auto Ri = Ti.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
        auto ti = Ti.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

        auto Rj = Tj.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
        auto tj = Tj.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

        auto Ti_ps = (Ri.Matmul(ps.T())).Add_(ti).T().Contiguous();
        auto Tj_qs = (Rj.Matmul(qs.T())).Add_(tj).T().Contiguous();
        auto Ri_normal_ps = (Ri.Matmul(normal_ps.T())).T().Contiguous();

        kernel::FillInRigidAlignmentTerm(AtA, Atb, residual, Ti_ps, Tj_qs,
                                         Ri_normal_ps, i, j);

        // For debug
        if (option.grid_debug_) {
            t::geometry::PointCloud tpcd_i_corres(Ti_ps);
            t::geometry::PointCloud tpcd_j_corres(Tj_qs);
            auto pcd_i_corres = std::make_shared<open3d::geometry::PointCloud>(
                    tpcd_i_corres.ToLegacyPointCloud());
            pcd_i_corres->PaintUniformColor({1, 0, 0});
            auto pcd_j_corres = std::make_shared<open3d::geometry::PointCloud>(
                    tpcd_j_corres.ToLegacyPointCloud());
            pcd_j_corres->PaintUniformColor({0, 1, 0});

            visualization::DrawGeometries({pcd_i_corres, pcd_j_corres});
        }
    }

    AtA.Save(fmt::format("{}/hessian.npy", option.GetSubfolderName()));
    Atb.Save(fmt::format("{}/residual.npy", option.GetSubfolderName()));
}

core::Tensor Solve(core::Tensor& AtA,
                   core::Tensor& Atb,
                   const SLACOptimizerOption& option) {
    core::Tensor Atb_neg = -1 * Atb;
    return AtA.Solve(Atb_neg);
}

void UpdatePoses(PoseGraph& fragment_pose_graph,
                 core::Tensor& delta,
                 const SLACOptimizerOption& option) {
    core::Tensor delta_poses = delta.View({-1, 6}).To(core::Device("CPU:0"));

    if (delta_poses.GetLength() != int64_t(fragment_pose_graph.nodes_.size())) {
        utility::LogError("Dimension Mismatch");
    }
    for (int64_t i = 0; i < delta_poses.GetLength(); ++i) {
        // std::cout << i << "\n";
        core::Tensor pose_tensor =
                kernel::PoseToTransformation(delta_poses[i])
                        .Matmul(core::eigen_converter::EigenMatrixToTensor(
                                        fragment_pose_graph.nodes_[i].pose_)
                                        .To(core::Dtype::Float32));
        Eigen::Matrix4d pose_eigen;
        pose_eigen << pose_tensor[0][0].Item<float>(),
                pose_tensor[0][1].Item<float>(),
                pose_tensor[0][2].Item<float>(),
                pose_tensor[0][3].Item<float>(),
                pose_tensor[1][0].Item<float>(),
                pose_tensor[1][1].Item<float>(),
                pose_tensor[1][2].Item<float>(),
                pose_tensor[1][3].Item<float>(),
                pose_tensor[2][0].Item<float>(),
                pose_tensor[2][1].Item<float>(),
                pose_tensor[2][2].Item<float>(),
                pose_tensor[2][3].Item<float>(),
                pose_tensor[3][0].Item<float>(),
                pose_tensor[3][1].Item<float>(),
                pose_tensor[3][2].Item<float>(),
                pose_tensor[3][3].Item<float>();
        fragment_pose_graph.nodes_[i].pose_ = pose_eigen;
    }
}

void UpdateControlGrid(ControlGrid& ctr_grid,
                       core::Tensor& result,
                       const SLACOptimizerOption& option) {
    utility::LogError("Unimplemented.");
}

std::pair<PoseGraph, ControlGrid> RunSLACOptimizerForFragments(
        const std::vector<std::string>& fnames,
        const PoseGraph& pose_graph,
        const SLACOptimizerOption& option) {
    core::Device device(option.device_);
    if (!option.buffer_folder_.empty()) {
        utility::filesystem::MakeDirectory(option.buffer_folder_);
    }

    // First preprocess the point cloud with downsampling and normal estimation.
    auto fnames_down = PreprocessPointClouds(fnames, option);
    // Then obtain the correspondences given the pose graph
    GetCorrespondencesForPointClouds(fnames_down, pose_graph, option);

    // First initialize ctr_grid
    ControlGrid ctr_grid(3.0 / 8, 1000, device);
    InitializeControlGrid(ctr_grid, fnames_down, option);

    ctr_grid.GenerateIndexLookupTable();

    // Fill-in
    // fragments x 6 (se3) + control_grids x 3 (R^3)
    int64_t num_params = fnames_down.size() * 6 + ctr_grid.Size() * 3;
    utility::LogInfo("Initializing {}^2 matrices", num_params);

    PoseGraph pose_graph_update(pose_graph);
    for (int itr = 0; itr < option.max_iterations_; ++itr) {
        core::Tensor AtA = core::Tensor::Zeros({num_params, num_params},
                                               core::Dtype::Float32, device);
        core::Tensor Atb = core::Tensor::Zeros({num_params, 1},
                                               core::Dtype::Float32, device);
        core::Tensor residual =
                core::Tensor::Zeros({1}, core::Dtype::Float32, device);

        core::Tensor indices_eye0 =
                core::Tensor::Arange(0, 6, 1, core::Dtype::Int64, device);
        AtA.IndexSet(
                {indices_eye0, indices_eye0},
                1e5 * core::Tensor::Ones({}, core::Dtype::Float32, device));

        utility::LogInfo("Iteration {}", itr);
        FillInSLACAlignmentTerm(AtA, Atb, residual, ctr_grid, fnames_down,
                                pose_graph_update, option);
        FillInSLACRegularizerTerm(AtA, Atb, ctr_grid, option);

        core::Tensor delta = Solve(AtA, Atb, option);

        UpdatePoses(pose_graph_update, delta, option);
        UpdateControlGrid(ctr_grid, delta, option);

        utility::LogError("Unimplemented!");
    }
    return std::make_pair(pose_graph_update, ctr_grid);
}

PoseGraph RunRigidOptimizerForFragments(const std::vector<std::string>& fnames,
                                        const PoseGraph& pose_graph,
                                        const SLACOptimizerOption& option) {
    core::Device device(option.device_);
    if (!option.buffer_folder_.empty()) {
        utility::filesystem::MakeDirectory(option.buffer_folder_);
    }

    // First preprocess the point cloud with downsampling and normal estimation.
    std::vector<std::string> fnames_down =
            PreprocessPointClouds(fnames, option);
    // Then obtain the correspondences given the pose graph
    GetCorrespondencesForPointClouds(fnames_down, pose_graph, option);

    // Fill-in
    // fragments x 6 (se3)
    int64_t num_params = fnames_down.size() * 6;
    utility::LogInfo("Initializing {}^2 Hessian matrix", num_params);

    PoseGraph pose_graph_update(pose_graph);
    for (int itr = 0; itr < option.max_iterations_; ++itr) {
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

        utility::LogInfo("Iteration {}", itr);
        FillInRigidAlignmentTerm(AtA, Atb, residual, fnames_down,
                                 pose_graph_update, option);
        utility::LogInfo("Residual = {}", residual[0].Item<float>());

        core::Tensor delta = Solve(AtA, Atb, option);
        UpdatePoses(pose_graph_update, delta, option);
    }

    return pose_graph_update;
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
