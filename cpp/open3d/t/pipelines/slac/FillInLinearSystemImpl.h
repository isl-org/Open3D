// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <fstream>

#include "open3d/core/EigenConverter.h"
#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"
#include "open3d/t/pipelines/slac/SLACOptimizer.h"
#include "open3d/utility/FileSystem.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

using namespace open3d::core::eigen_converter;
using core::Tensor;
using t::geometry::PointCloud;

// Reads pointcloud from filename, and loads on the device as
// Tensor PointCloud of Float32 dtype.
static PointCloud CreateTPCDFromFile(
        const std::string& fname,
        const core::Device& device = core::Device("CPU:0")) {
    std::shared_ptr<open3d::geometry::PointCloud> pcd =
            open3d::io::CreatePointCloudFromFile(fname);
    return PointCloud::FromLegacy(*pcd, core::Float32, device);
}

static void FillInRigidAlignmentTerm(Tensor& AtA,
                                     Tensor& Atb,
                                     Tensor& residual,
                                     PointCloud& tpcd_i,
                                     PointCloud& tpcd_j,
                                     const Tensor& Ti,
                                     const Tensor& Tj,
                                     const int i,
                                     const int j,
                                     const float threshold) {
    tpcd_i.Transform(Ti);
    tpcd_j.Transform(Tj);

    kernel::FillInRigidAlignmentTerm(AtA, Atb, residual,
                                     tpcd_i.GetPointPositions(),
                                     tpcd_j.GetPointPositions(),
                                     tpcd_i.GetPointNormals(), i, j, threshold);
}

void FillInRigidAlignmentTerm(Tensor& AtA,
                              Tensor& Atb,
                              Tensor& residual,
                              const std::vector<std::string>& fnames,
                              const PoseGraph& pose_graph,
                              const SLACOptimizerParams& params,
                              const SLACDebugOption& debug_option) {
    core::Device device(params.device_);

    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.npy",
                                               params.GetSubfolderName(), i, j);
        if (!utility::filesystem::FileExists(corres_fname)) {
            utility::LogWarning("Correspondence {} {} skipped!", i, j);
            continue;
        }
        Tensor corres_ij = Tensor::Load(corres_fname).To(device);
        PointCloud tpcd_i = CreateTPCDFromFile(fnames[i], device);
        PointCloud tpcd_j = CreateTPCDFromFile(fnames[j], device);

        PointCloud tpcd_i_indexed(
                tpcd_i.GetPointPositions().IndexGet({corres_ij.T()[0]}));
        tpcd_i_indexed.SetPointNormals(
                tpcd_i.GetPointNormals().IndexGet({corres_ij.T()[0]}));
        PointCloud tpcd_j_indexed(
                tpcd_j.GetPointPositions().IndexGet({corres_ij.T()[1]}));

        Tensor Ti = EigenMatrixToTensor(pose_graph.nodes_[i].pose_)
                            .To(device, core::Float32);
        Tensor Tj = EigenMatrixToTensor(pose_graph.nodes_[j].pose_)
                            .To(device, core::Float32);

        FillInRigidAlignmentTerm(AtA, Atb, residual, tpcd_i_indexed,
                                 tpcd_j_indexed, Ti, Tj, i, j,
                                 params.distance_threshold_);

        if (debug_option.debug_ && i >= debug_option.debug_start_node_idx_) {
            VisualizePointCloudCorrespondences(tpcd_i, tpcd_j, corres_ij,
                                               Tj.Inverse().Matmul(Ti));
        }
    }
}

static void FillInSLACAlignmentTerm(Tensor& AtA,
                                    Tensor& Atb,
                                    Tensor& residual,
                                    ControlGrid& ctr_grid,
                                    const PointCloud& tpcd_param_i,
                                    const PointCloud& tpcd_param_j,
                                    const Tensor& Ti,
                                    const Tensor& Tj,
                                    const int i,
                                    const int j,
                                    const int n_fragments,
                                    const float threshold) {
    // Parameterize: setup point cloud -> cgrid correspondences
    Tensor cgrid_index_ps =
            tpcd_param_i.GetPointAttr(ControlGrid::kGrid8NbIndices);
    Tensor cgrid_ratio_ps =
            tpcd_param_i.GetPointAttr(ControlGrid::kGrid8NbVertexInterpRatios);

    Tensor cgrid_index_qs =
            tpcd_param_j.GetPointAttr(ControlGrid::kGrid8NbIndices);
    Tensor cgrid_ratio_qs =
            tpcd_param_j.GetPointAttr(ControlGrid::kGrid8NbVertexInterpRatios);

    // Deform with control grids
    PointCloud tpcd_nonrigid_i = ctr_grid.Deform(tpcd_param_i);
    PointCloud tpcd_nonrigid_j = ctr_grid.Deform(tpcd_param_j);

    Tensor Cps = tpcd_nonrigid_i.GetPointPositions();
    Tensor Cqs = tpcd_nonrigid_j.GetPointPositions();
    Tensor Cnormal_ps = tpcd_nonrigid_i.GetPointNormals();

    Tensor Ri = Ti.Slice(0, 0, 3).Slice(1, 0, 3);
    Tensor ti = Ti.Slice(0, 0, 3).Slice(1, 3, 4);

    Tensor Rj = Tj.Slice(0, 0, 3).Slice(1, 0, 3);
    Tensor tj = Tj.Slice(0, 0, 3).Slice(1, 3, 4);

    // Transform for required entries
    Tensor Ti_Cps = (Ri.Matmul(Cps.T())).Add_(ti).T().Contiguous();
    Tensor Tj_Cqs = (Rj.Matmul(Cqs.T())).Add_(tj).T().Contiguous();
    Tensor Ri_Cnormal_ps = (Ri.Matmul(Cnormal_ps.T())).T().Contiguous();
    Tensor RjT_Ri_Cnormal_ps =
            (Rj.T().Matmul(Ri_Cnormal_ps.T())).T().Contiguous();

    kernel::FillInSLACAlignmentTerm(
            AtA, Atb, residual, Ti_Cps, Tj_Cqs, Cnormal_ps, Ri_Cnormal_ps,
            RjT_Ri_Cnormal_ps, cgrid_index_ps, cgrid_index_qs, cgrid_ratio_ps,
            cgrid_ratio_qs, i, j, n_fragments, threshold);
}

void FillInSLACAlignmentTerm(Tensor& AtA,
                             Tensor& Atb,
                             Tensor& residual,
                             ControlGrid& ctr_grid,
                             const std::vector<std::string>& fnames,
                             const PoseGraph& pose_graph,
                             const SLACOptimizerParams& params,
                             const SLACDebugOption& debug_option) {
    core::Device device(params.device_);
    int n_frags = pose_graph.nodes_.size();

    // Enumerate pose graph edges.
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.npy",
                                               params.GetSubfolderName(), i, j);
        if (!utility::filesystem::FileExists(corres_fname)) {
            utility::LogWarning("Correspondence {} {} skipped!", i, j);
            continue;
        }
        Tensor corres_ij = Tensor::Load(corres_fname).To(device);

        PointCloud tpcd_i = CreateTPCDFromFile(fnames[i], device);
        PointCloud tpcd_j = CreateTPCDFromFile(fnames[j], device);

        PointCloud tpcd_i_indexed(
                tpcd_i.GetPointPositions().IndexGet({corres_ij.T()[0]}));
        tpcd_i_indexed.SetPointNormals(
                tpcd_i.GetPointNormals().IndexGet({corres_ij.T()[0]}));

        PointCloud tpcd_j_indexed(
                tpcd_j.GetPointPositions().IndexGet({corres_ij.T()[1]}));
        tpcd_j_indexed.SetPointNormals(
                tpcd_j.GetPointNormals().IndexGet({corres_ij.T()[1]}));

        // Parameterize points in the control grid.
        PointCloud tpcd_param_i = ctr_grid.Parameterize(tpcd_i_indexed);
        PointCloud tpcd_param_j = ctr_grid.Parameterize(tpcd_j_indexed);

        // Load poses.
        auto Ti = EigenMatrixToTensor(pose_graph.nodes_[i].pose_)
                          .To(device, core::Float32);
        auto Tj = EigenMatrixToTensor(pose_graph.nodes_[j].pose_)
                          .To(device, core::Float32);
        auto Tij = EigenMatrixToTensor(edge.transformation_)
                           .To(device, core::Float32);

        // Fill In.
        FillInSLACAlignmentTerm(AtA, Atb, residual, ctr_grid, tpcd_param_i,
                                tpcd_param_j, Ti, Tj, i, j, n_frags,
                                params.distance_threshold_);

        if (debug_option.debug_ && i >= debug_option.debug_start_node_idx_) {
            VisualizePointCloudCorrespondences(tpcd_i, tpcd_j, corres_ij,
                                               Tj.Inverse().Matmul(Ti));
            VisualizePointCloudEmbedding(tpcd_param_i, ctr_grid);
            VisualizePointCloudDeformation(tpcd_param_i, ctr_grid);
        }
    }
}

void FillInSLACRegularizerTerm(Tensor& AtA,
                               Tensor& Atb,
                               Tensor& residual,
                               ControlGrid& ctr_grid,
                               int n_frags,
                               const SLACOptimizerParams& params,
                               const SLACDebugOption& debug_option) {
    Tensor active_buf_indices, nb_buf_indices, nb_masks;
    std::tie(active_buf_indices, nb_buf_indices, nb_masks) =
            ctr_grid.GetNeighborGridMap();

    Tensor positions_init = ctr_grid.GetInitPositions();
    Tensor positions_curr = ctr_grid.GetCurrPositions();
    kernel::FillInSLACRegularizerTerm(AtA, Atb, residual, active_buf_indices,
                                      nb_buf_indices, nb_masks, positions_init,
                                      positions_curr,
                                      n_frags * params.regularizer_weight_,
                                      n_frags, ctr_grid.GetAnchorIdx());
    if (debug_option.debug_) {
        VisualizeGridDeformation(ctr_grid);
    }
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
