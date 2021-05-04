// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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

#pragma once

#include <fstream>

#include "open3d/core/EigenConverter.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/pipelines/registration/ColoredICP.h"
#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

using TPointCloud = t::geometry::PointCloud;

static void FillInRigidAlignmentTerm(core::Tensor& AtA,
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

        TPointCloud tpcd_i = CreateTPCDFromFile(fnames[i]);
        TPointCloud tpcd_j = CreateTPCDFromFile(fnames[j]);

        core::Tensor Ti = core::eigen_converter::EigenMatrixToTensor(
                                  pose_graph.nodes_[i].pose_)
                                  .To(core::Dtype::Float32);
        core::Tensor Tj = core::eigen_converter::EigenMatrixToTensor(
                                  pose_graph.nodes_[j].pose_)
                                  .To(core::Dtype::Float32);

        core::Tensor corres_ij = core::Tensor::Load(corres_fname).To(device);

        FillInRigidAlignmentTerm(
                AtA, Atb, residual, tpcd_i, tpcd_j, Ti, Tj, corres_ij, i, j,
                option.threshold_, device,
                option.debug_enabled_ && i >= option.debug_start_idx_);
    }

    AtA.Save(fmt::format("{}/hessian.npy", option.GetSubfolderName()));
    Atb.Save(fmt::format("{}/residual.npy", option.GetSubfolderName()));
}

static void FillInSLACAlignmentTerm(core::Tensor& AtA,
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
    TPointCloud tpcd_i_points_indexed(
            tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}).To(device));
    tpcd_i_points_indexed.SetPointNormals(
            tpcd_i.GetPointNormals().IndexGet({corres_ij.T()[0]}).To(device));

    TPointCloud tpcd_j_points_indexed(
            tpcd_j.GetPoints().IndexGet({corres_ij.T()[1]}).To(device));
    tpcd_j_points_indexed.SetPointNormals(
            tpcd_j.GetPointNormals().IndexGet({corres_ij.T()[1]}).To(device));

    // Parameterize points in the control grid.
    TPointCloud tpcd_param_i = ctr_grid.Parameterize(tpcd_i_points_indexed);
    TPointCloud tpcd_param_j = ctr_grid.Parameterize(tpcd_j_points_indexed);

    // Parameterize: setup point cloud -> cgrid correspondences
    core::Tensor cgrid_index_ps =
            tpcd_param_i.GetPointAttr(ControlGrid::kGrid8NbIndices).To(device);
    core::Tensor cgrid_ratio_ps =
            tpcd_param_i.GetPointAttr(ControlGrid::kGrid8NbVertexInterpRatios)
                    .To(device);

    core::Tensor cgrid_index_qs =
            tpcd_param_j.GetPointAttr(ControlGrid::kGrid8NbIndices).To(device);
    core::Tensor cgrid_ratio_qs =
            tpcd_param_j.GetPointAttr(ControlGrid::kGrid8NbVertexInterpRatios)
                    .To(device);

    // Deform with control grids
    TPointCloud tpcd_nonrigid_i = ctr_grid.Deform(tpcd_param_i);
    TPointCloud tpcd_nonrigid_j = ctr_grid.Deform(tpcd_param_j);

    // TODO: Put the following pre-processing inside the FillInSlACAlignmentTerm
    // kernel, if possible.
    core::Tensor Cps = tpcd_nonrigid_i.GetPoints();
    core::Tensor Cqs = tpcd_nonrigid_j.GetPoints();
    core::Tensor Cnormal_ps = tpcd_nonrigid_i.GetPointNormals();

    core::Tensor Ri = Ti.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
    core::Tensor ti = Ti.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

    core::Tensor Rj = Tj.Slice(0, 0, 3).Slice(1, 0, 3).To(device);
    core::Tensor tj = Tj.Slice(0, 0, 3).Slice(1, 3, 4).To(device);

    // Transform for required entries
    core::Tensor Ti_Cps = (Ri.Matmul(Cps.T())).Add_(ti).T().Contiguous();
    core::Tensor Tj_Cqs = (Rj.Matmul(Cqs.T())).Add_(tj).T().Contiguous();
    core::Tensor Ri_Cnormal_ps = (Ri.Matmul(Cnormal_ps.T())).T().Contiguous();
    core::Tensor RjT_Ri_Cnormal_ps =
            (Rj.T().Matmul(Ri_Cnormal_ps.T())).T().Contiguous();

    kernel::FillInSLACAlignmentTerm(
            AtA, Atb, residual, Ti_Cps, Tj_Cqs, Cnormal_ps, Ri_Cnormal_ps,
            RjT_Ri_Cnormal_ps, cgrid_index_ps, cgrid_index_qs, cgrid_ratio_ps,
            cgrid_ratio_qs, i, j, n_fragments, threshold);

    if (visualize) {
        utility::LogInfo("edge {} -> {}", i, j);
        VisualizePCDCorres(tpcd_i, tpcd_j, tpcd_param_i, tpcd_param_j, Tij_icp);
        // VisualizePCDCorres(tpcd_i, tpcd_j, tpcd_nonrigid_i, tpcd_nonrigid_j,
        //                    Tij_icp);

        // VisualizeDeform(tpcd_param_i, ctr_grid);
        // VisualizeDeform(tpcd_param_j, ctr_grid);

        // VisualizePCDGridCorres(tpcd_param_i, ctr_grid);
        // VisualizePCDGridCorres(tpcd_param_j, ctr_grid);
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

    // Enumerate pose graph edges.
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        // Load poses.
        auto Ti = core::eigen_converter::EigenMatrixToTensor(
                          pose_graph.nodes_[i].pose_)
                          .To(device, core::Dtype::Float32);
        auto Tj = core::eigen_converter::EigenMatrixToTensor(
                          pose_graph.nodes_[j].pose_)
                          .To(device, core::Dtype::Float32);
        auto Tij =
                core::eigen_converter::EigenMatrixToTensor(edge.transformation_)
                        .To(device, core::Dtype::Float32);

        // Load point clouds.
        TPointCloud tpcd_i = CreateTPCDFromFile(fnames[i], device);
        TPointCloud tpcd_j = CreateTPCDFromFile(fnames[j], device);

        // Load correspondences.
        std::string correspondences_fname = fmt::format(
                "{}/{:03d}_{:03d}.npy", option.GetSubfolderName(), i, j);
        if (!utility::filesystem::FileExists(correspondences_fname)) {
            utility::LogWarning("Correspondence {} {} not processed!", i, j);
            continue;
        }
        core::Tensor correspondences_ij =
                core::Tensor::Load(correspondences_fname).To(device);

        // Fill In.
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

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
