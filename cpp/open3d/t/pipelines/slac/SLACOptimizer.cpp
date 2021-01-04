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
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/utility/FileSystem.h"
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

struct SLACPairwiseCorrespondence {
    SLACPairwiseCorrespondence() {}
    SLACPairwiseCorrespondence(int i, int j, const core::Tensor& corres)
        : i_(i), j_(j), correspondence_(corres) {}

    // Source index
    int i_;
    // Target index
    int j_;

    // N x 2 for point clouds, storing corresponding point indices;
    // N x 4 for RGBD images, storing corresponding uv coordinates.
    core::Tensor correspondence_;

    static SLACPairwiseCorrespondence ReadFromFile(const std::string& fname) {
        int i, j;
        int64_t len;

        std::ifstream fin(fname, std::ifstream::binary);
        fin.read(reinterpret_cast<char*>(&i), sizeof(int));
        fin.read(reinterpret_cast<char*>(&j), sizeof(int));
        fin.read(reinterpret_cast<char*>(&len), sizeof(int64_t));

        core::Tensor corres({len, 2}, core::Dtype::Int64);
        fin.read(static_cast<char*>(corres.GetDataPtr()),
                 corres.NumElements() * core::Dtype::Int64.ByteSize());

        return SLACPairwiseCorrespondence(i, j, corres);
    }

    void Write(const std::string& fname) {
        std::ofstream fout(fname, std::ofstream::binary);
        int64_t len = correspondence_.GetLength();
        fout.write(reinterpret_cast<const char*>(&i_), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&j_), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&len), sizeof(int64_t));
        fout.write(
                static_cast<const char*>(correspondence_.GetDataPtr()),
                correspondence_.NumElements() * core::Dtype::Int64.ByteSize());
    }
};

/// Write point clouds after downsampling and normal estimation for
/// correspondence check.
std::vector<std::string> PreprocessPointClouds(
        const std::vector<std::string>& fragment_fnames,
        const SLACOptimizerOption& option) {
    std::vector<std::string> fnames_down;
    for (auto& fname : fragment_fnames) {
        std::string fname_down = fmt::format(
                "{}/{}", option.buffer_folder_,
                utility::filesystem::GetFileNameWithoutDirectory(fname));
        fnames_down.emplace_back(fname_down);

        if (utility::filesystem::FileExists(fname_down)) continue;
        utility::LogInfo("Processing {}", fname_down);

        auto pcd = io::CreatePointCloudFromFile(fname);
        auto pcd_down = pcd->VoxelDownSample(option.voxel_size_);
        pcd_down->EstimateNormals();
        io::WritePointCloud(fname_down, *pcd_down);
    }

    return fnames_down;
}

/// Read pose graph containing loop closures and odometry to compute
/// correspondences.
void GetCorrespondencesForPointClouds(
        const std::vector<std::string>& fragment_down_fnames,
        const open3d::pipelines::registration::PoseGraph& pose_graph,
        const SLACOptimizerOption& option) {
    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;
        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.corres",
                                               option.buffer_folder_, i, j);
        if (utility::filesystem::FileExists(corres_fname)) continue;

        auto pcd_i = io::CreatePointCloudFromFile(fragment_down_fnames[i]);
        t::geometry::PointCloud tpcd_i =
                t::geometry::PointCloud::FromLegacyPointCloud(
                        *pcd_i, core::Dtype::Float32);

        auto pcd_j = io::CreatePointCloudFromFile(fragment_down_fnames[j]);
        t::geometry::PointCloud tpcd_j =
                t::geometry::PointCloud::FromLegacyPointCloud(
                        *pcd_j, core::Dtype::Float32);

        auto pose_i = pose_graph.nodes_[i].pose_;
        auto pose_j = pose_graph.nodes_[j].pose_;
        auto pose_ij = (pose_j.inverse() * pose_i).eval();

        // Obtain correspondence
        auto result = t::pipelines::registration::EvaluateRegistration(
                tpcd_i, tpcd_j, option.voxel_size_,
                core::eigen_converter::EigenMatrixToTensor(pose_ij).To(
                        core::Dtype::Float32));
        core::Tensor corres =
                core::Tensor({result.correspondence_set_.GetLength(), 2},
                             core::Dtype::Int64);

        // Make correspondence indices
        std::vector<int64_t> arange(
                result.correspondence_select_bool_.GetLength());
        std::iota(arange.begin(), arange.end(), 0);
        core::Tensor indices(arange,
                             {result.correspondence_select_bool_.GetLength()},
                             core::Dtype::Int64);
        corres.SetItem(
                {core::TensorKey::Slice(core::None, core::None, core::None),
                 core::TensorKey::Index(0)},
                indices.IndexGet({result.correspondence_select_bool_}));
        corres.SetItem(
                {core::TensorKey::Slice(core::None, core::None, core::None),
                 core::TensorKey::Index(1)},
                result.correspondence_set_);

        auto pair_corres = SLACPairwiseCorrespondence(i, j, corres);
        pair_corres.Write(corres_fname);
        utility::LogInfo("Edge: {:02d} -> {:02d}, corres {}", i, j,
                         corres.GetLength());

        // For IO debug
        // auto corres_read =
        //         SLACPairwiseCorrespondence::ReadFromFile(corres_fname);
        // utility::LogInfo("written = {}", corres.GetShape());
        // utility::LogInfo("read = {}",
        // corres_read.correspondence_.GetShape()); if
        // (!corres.AllClose(corres_read.correspondence_)) {
        //     utility::LogError("IO of correspondences mismatch");
        // }

        // For visual debug
        if (option.visual_debug_) {
            pcd_i->Transform(pose_ij);
            visualization::DrawGeometries({pcd_i, pcd_j});
            pcd_i->Transform(pose_ij.inverse());

            std::vector<std::pair<int, int>> corres_lines;
            for (int64_t i = 0; i < corres.GetLength(); ++i) {
                std::pair<int, int> pair = {corres[i][0].Item<int64_t>(),
                                            corres[i][1].Item<int64_t>()};
                corres_lines.push_back(pair);
            }
            auto lineset = open3d::geometry::LineSet::
                    CreateFromPointCloudCorrespondences(*pcd_i, *pcd_j,
                                                        corres_lines);
            lineset->PaintUniformColor({0, 1, 0});
            visualization::DrawGeometries({pcd_i, pcd_j, lineset});
        }
    }
}

void InitializeControlGrid(ControlGrid& ctr_grid,
                           const std::vector<std::string>& fnames) {
    for (auto& fname : fnames) {
        utility::LogInfo("Initializing grid for {}", fname);

        auto pcd = io::CreatePointCloudFromFile(fname);
        auto tpcd = t::geometry::PointCloud::FromLegacyPointCloud(
                *pcd, core::Dtype::Float32);
        ctr_grid.Touch(tpcd);
    }
}

std::pair<core::Tensor, core::Tensor> FillInControlGrid(
        ControlGrid& ctr_grid,
        const std::vector<std::string>& fnames,
        const open3d::pipelines::registration::PoseGraph& rgbd_pose_graph,
        const SLACOptimizerOption& option) {
    core::Tensor AtA, Atb;

    for (auto& fname : fnames) {
        utility::LogInfo("Parameterizing {}", fname);

        auto pcd = io::CreatePointCloudFromFile(fname);
        auto tpcd = t::geometry::PointCloud::FromLegacyPointCloud(
                *pcd, core::Dtype::Float32);
        auto tpcd_param = ctr_grid.Parameterize(tpcd);

        // Prepare all ctr grid point cloud for lineset
        t::geometry::PointCloud tpcd_grid(
                tpcd_param.GetPointAttr("ctr_grid_positions"));
        auto pcd_grid = std::make_shared<open3d::geometry::PointCloud>(
                tpcd_grid.ToLegacyPointCloud());

        // Prepare n x 8 corres for visualization
        core::Tensor corres = tpcd_param.GetPointAttr("ctr_grid_nb_idx");
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
                tpcd_param.GetPointAttr("ctr_grid_nb_ratio");
        for (int64_t i = 0; i < corres.GetLength(); ++i) {
            for (int k = 0; k < 8; ++k) {
                float ratio = corres_interp[i][k].Item<float>();
                Eigen::Vector3d color = Jet(ratio, 0, 0.5);
                utility::LogInfo("{}: ({} {} {})", ratio, color(0), color(1),
                                 color(2));
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

    return std::make_pair(AtA, Atb);
}

std::vector<SLACPairwiseCorrespondence> GetCorrespondencesForRGBDImages(
        const std::vector<std::string>& rgbd_fnames,
        const open3d::pipelines::registration::PoseGraph& rgbd_pose_graph) {
    utility::LogError("Unimplemented.");
}

void FillInAlignmentTerm(core::Tensor& tensor) {
    utility::LogError("Unimplemented.");
}

void FillInRegularizer(core::Tensor& tensor) {
    utility::LogError("Unimplemented.");
}

ControlGrid RunSLACOptimizerForFragments(
        const std::vector<std::string>& fragment_fnames,
        const open3d::pipelines::registration::PoseGraph& fragment_pose_graph,
        const SLACOptimizerOption& option) {
    if (!option.buffer_folder_.empty()) {
        utility::filesystem::MakeDirectory(option.buffer_folder_);
    }

    // First preprocess the point cloud with downsampling and normal estimation.
    auto fragment_down_fnames = PreprocessPointClouds(fragment_fnames, option);
    // Then obtain the correspondences given the pose graph
    GetCorrespondencesForPointClouds(fragment_down_fnames, fragment_pose_graph,
                                     option);

    // First initialize ctr_grid
    ControlGrid ctr_grid(3.0 / 8);
    InitializeControlGrid(ctr_grid, fragment_down_fnames);

    // Fill-in using
    core::Tensor AtA, Atb;
    std::tie(AtA, Atb) = FillInControlGrid(ctr_grid, fragment_down_fnames,
                                           fragment_pose_graph, option);

    // // Then allocate the Hessian matrix.
    // // TODO: write a sparse matrix representation / helper.
    // int64_t n = 6 * num_fragments + control_grids.count);
    // core::Tensor AtA({n, n}, core::Dtype::Float32);
    // core::Tensor Atb({n}, core::Dtype::Float32);

    // // Core: iterative optimization
    // for (int itr = 0; itr < max_itr; ++itr) {
    //     // First: alignment term from correspondences
    //     for (auto pair : pairs) {
    //         auto pcd_i = io::ReadPointCloud(pair.i_);
    //         auto pcd_j = io::ReadPointCloud(pair.j_);
    //         auto corres_ij = pair.correspondences_;
    //         pcd_i = ctr_grid.Warp(pcd_i);
    //         pcd_j = ctr_grid.Warp(pcd_j);

    //         // Parallel fill-in
    //         FillInAlignmentTerm(AtA, pcd_i, pcd_j, corres_ij, ctr_grid);
    //     }

    //     // Next: regularization term from neighbors in the control grid
    //     FillInRegularizer(AtA, ctr_grid);

    //     // Solve the linear system
    //     std::tie(poses, ctr_grid) = Solve(AtA, Atb);

    //     // Update
    //     UpdateRotationsAndNormals();
    // }

    utility::LogError("Unimplemented!");
}

ControlGrid RunSLACOptimizerForRGBDImages(
        const std::vector<std::string>& rgbd_fnames,
        const open3d::pipelines::registration::PoseGraph& rgbd_pose_graph,
        const SLACOptimizerOption& option) {
    utility::LogError("Unimplemented!");
}
}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
