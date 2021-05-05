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

using namespace open3d::core::eigen_converter;
using TPointCloud = t::geometry::PointCloud;
using core::Tensor;

static void FillInRigidAlignmentTerm(Tensor& AtA,
                                     Tensor& Atb,
                                     Tensor& residual,
                                     TPointCloud& tpcd_i,
                                     TPointCloud& tpcd_j,
                                     const Tensor& Ti,
                                     const Tensor& Tj,
                                     const int i,
                                     const int j,
                                     const float threshold) {
    tpcd_i.Transform(Ti);
    tpcd_j.Transform(Tj);

    kernel::FillInRigidAlignmentTerm(AtA, Atb, residual, tpcd_i.GetPoints(),
                                     tpcd_j.GetPoints(),
                                     tpcd_i.GetPointNormals(), i, j, threshold);
}

void FillInRigidAlignmentTerm(Tensor& AtA,
                              Tensor& Atb,
                              Tensor& residual,
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
        Tensor corres_ij = Tensor::Load(corres_fname).To(device);
        TPointCloud tpcd_i = CreateTPCDFromFile(fnames[i], device);
        TPointCloud tpcd_j = CreateTPCDFromFile(fnames[j], device);

        TPointCloud tpcd_i_indexed(
                tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}));
        tpcd_i_indexed.SetPointNormals(
                tpcd_i.GetPointNormals().IndexGet({corres_ij.T()[0]}));
        TPointCloud tpcd_j_indexed(
                tpcd_j.GetPoints().IndexGet({corres_ij.T()[1]}));

        Tensor Ti = EigenMatrixToTensor(pose_graph.nodes_[i].pose_)
                            .To(device, core::Dtype::Float32);
        Tensor Tj = EigenMatrixToTensor(pose_graph.nodes_[j].pose_)
                            .To(device, core::Dtype::Float32);

        FillInRigidAlignmentTerm(AtA, Atb, residual, tpcd_i_indexed,
                                 tpcd_j_indexed, Ti, Tj, i, j,
                                 option.threshold_);

        // if (option.debug_enabled_ && i >= option.debug_start_idx_) {
        VisualizePointCloudCorrespondences(tpcd_i, tpcd_j, corres_ij,
                                           Tj.Inverse().Matmul(Ti));
        //}
    }
}

static void FillInSLACAlignmentTerm(Tensor& AtA,
                                    Tensor& Atb,
                                    Tensor& residual,
                                    ControlGrid& ctr_grid,
                                    const TPointCloud& tpcd_param_i,
                                    const TPointCloud& tpcd_param_j,
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
    TPointCloud tpcd_nonrigid_i = ctr_grid.Deform(tpcd_param_i);
    TPointCloud tpcd_nonrigid_j = ctr_grid.Deform(tpcd_param_j);

    Tensor Cps = tpcd_nonrigid_i.GetPoints();
    Tensor Cqs = tpcd_nonrigid_j.GetPoints();
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
                             const SLACOptimizerOption& option) {
    core::Device device(option.device_);
    int n_frags = pose_graph.nodes_.size();

    // Enumerate pose graph edges.
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        std::string correspondences_fname = fmt::format(
                "{}/{:03d}_{:03d}.npy", option.GetSubfolderName(), i, j);
        if (!utility::filesystem::FileExists(correspondences_fname)) {
            utility::LogWarning("Correspondence {} {} not processed!", i, j);
            continue;
        }
        Tensor corres_ij = Tensor::Load(correspondences_fname).To(device);

        TPointCloud tpcd_i = CreateTPCDFromFile(fnames[i], device);
        TPointCloud tpcd_j = CreateTPCDFromFile(fnames[j], device);

        TPointCloud tpcd_i_indexed(
                tpcd_i.GetPoints().IndexGet({corres_ij.T()[0]}));
        tpcd_i_indexed.SetPointNormals(
                tpcd_i.GetPointNormals().IndexGet({corres_ij.T()[0]}));

        TPointCloud tpcd_j_indexed(
                tpcd_j.GetPoints().IndexGet({corres_ij.T()[1]}));
        tpcd_j_indexed.SetPointNormals(
                tpcd_j.GetPointNormals().IndexGet({corres_ij.T()[1]}));

        // Parameterize points in the control grid.
        TPointCloud tpcd_param_i = ctr_grid.Parameterize(tpcd_i_indexed);
        TPointCloud tpcd_param_j = ctr_grid.Parameterize(tpcd_j_indexed);

        // Load poses.
        auto Ti = EigenMatrixToTensor(pose_graph.nodes_[i].pose_)
                          .To(device, core::Dtype::Float32);
        auto Tj = EigenMatrixToTensor(pose_graph.nodes_[j].pose_)
                          .To(device, core::Dtype::Float32);
        auto Tij = EigenMatrixToTensor(edge.transformation_)
                           .To(device, core::Dtype::Float32);

        // Fill In.
        FillInSLACAlignmentTerm(AtA, Atb, residual, ctr_grid, tpcd_param_i,
                                tpcd_param_j, Ti, Tj, i, j, n_frags,
                                option.threshold_);

        if (i == 0 && j == 1) {
            VisualizePointCloudCorrespondences(tpcd_i, tpcd_j, corres_ij,
                                               Tj.Inverse().Matmul(Ti));
            VisualizePointCloudEmbedding(tpcd_param_i, ctr_grid);
            VisualizePointCloudDeformation(tpcd_param_i, ctr_grid);
        }
    }
    VisualizeGridDeformation(ctr_grid);
}

void FillInSLACRegularizerTerm(Tensor& AtA,
                               Tensor& Atb,
                               Tensor& residual,
                               ControlGrid& ctr_grid,
                               int n_frags,
                               const SLACOptimizerOption& option) {
    Tensor active_addrs, nb_addrs, nb_masks;
    std::tie(active_addrs, nb_addrs, nb_masks) = ctr_grid.GetNeighborGridMap();

    Tensor positions_init = ctr_grid.GetInitPositions();
    Tensor positions_curr = ctr_grid.GetCurrPositions();
    kernel::FillInSLACRegularizerTerm(
            AtA, Atb, residual, active_addrs, nb_addrs, nb_masks,
            positions_init, positions_curr, n_frags * option.regularizor_coeff_,
            n_frags, ctr_grid.anchor_idx_);
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
