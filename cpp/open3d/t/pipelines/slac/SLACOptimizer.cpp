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

void VisualizePCDGridCorres(std::shared_ptr<open3d::geometry::PointCloud>& pcd,
                            t::geometry::PointCloud& tpcd_param) {
    // Prepare all ctr grid point cloud for lineset
    t::geometry::PointCloud tpcd_grid(
            tpcd_param.GetPointAttr("ctr_grid_positions"));
    auto pcd_grid = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_grid.ToLegacyPointCloud());

    // Prepare n x 8 corres for visualization
    core::Tensor corres = tpcd_param.GetPointAttr("ctr_grid_nb_idx")
                                  .Copy(core::Device("CPU:0"));
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

    core::Tensor corres_interp = tpcd_param.GetPointAttr("ctr_grid_nb_ratio")
                                         .Copy(core::Device("CPU:0"));
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
        const PoseGraph& pose_graph,
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

        // For visual debug
        if (option.correspondence_debug_) {
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
                           const std::vector<std::string>& fnames,
                           const SLACOptimizerOption& option) {
    core::Device device(option.device_);
    for (auto& fname : fnames) {
        utility::LogInfo("Initializing grid for {}", fname);

        auto pcd = io::CreatePointCloudFromFile(fname);
        auto tpcd = t::geometry::PointCloud::FromLegacyPointCloud(
                *pcd, core::Dtype::Float32, device);
        ctr_grid.Touch(tpcd);
    }
}

void FillInSLACAlignmentTerm(core::Tensor& AtA,
                             core::Tensor& Atb,
                             ControlGrid& ctr_grid,
                             const std::vector<std::string>& fnames,
                             const PoseGraph& pose_graph,
                             const SLACOptimizerOption& option) {
    core::Device device(option.device_);

    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;
        utility::LogInfo("edge {} -> {}", i, j);
        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.corres",
                                               option.buffer_folder_, i, j);
        if (!utility::filesystem::FileExists(corres_fname)) {
            utility::LogError("Correspondence not processed");
        }

        utility::LogInfo("pcd {}", i);
        auto pcd_i = io::CreatePointCloudFromFile(fnames[i]);
        auto tpcd_i = t::geometry::PointCloud::FromLegacyPointCloud(
                *pcd_i, core::Dtype::Float32, device);
        auto tpcd_param_i = ctr_grid.Parameterize(tpcd_i);

        utility::LogInfo("pcd {}", j);
        auto pcd_j = io::CreatePointCloudFromFile(fnames[j]);
        auto tpcd_j = t::geometry::PointCloud::FromLegacyPointCloud(
                *pcd_j, core::Dtype::Float32, device);
        auto tpcd_param_j = ctr_grid.Parameterize(tpcd_j);

        // auto pose_i = pose_graph.nodes_[i].pose_;
        // auto pose_j = pose_graph.nodes_[j].pose_;
        utility::LogInfo("corres {}{}", i, j);
        auto corres_ij = SLACPairwiseCorrespondence::ReadFromFile(corres_fname);

        if (option.grid_debug_) {
            utility::LogInfo("visualizing", j);
            VisualizePCDGridCorres(pcd_i, tpcd_param_i);
            VisualizePCDGridCorres(pcd_j, tpcd_param_j);
        }

        // TODO: use parameterization to update normals and points per grid
    }
}

void FillInSLACRegularizer(core::Tensor& AtA,
                           core::Tensor& Atb,
                           ControlGrid& ctr_grid,
                           const SLACOptimizerOption& option) {
    utility::LogError("Unimplemented.");
}

void FillInRigidAlignmentTerm(core::Tensor& AtA,
                              core::Tensor& Atb,
                              const std::vector<std::string>& fnames,
                              const PoseGraph& pose_graph,
                              const SLACOptimizerOption& option) {
    core::Device device(option.device_);

    // Enumerate pose graph edges
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;
        utility::LogInfo("edge {} -> {}", i, j);
        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.corres",
                                               option.buffer_folder_, i, j);
        if (!utility::filesystem::FileExists(corres_fname)) {
            utility::LogError("Correspondence not processed");
        }

        utility::LogInfo("pcd {}", i);
        auto pcd_i = io::CreatePointCloudFromFile(fnames[i]);
        auto tpcd_i = t::geometry::PointCloud::FromLegacyPointCloud(
                *pcd_i, core::Dtype::Float32, device);

        utility::LogInfo("pcd {}", j);
        auto pcd_j = io::CreatePointCloudFromFile(fnames[j]);
        auto tpcd_j = t::geometry::PointCloud::FromLegacyPointCloud(
                *pcd_j, core::Dtype::Float32, device);

        // auto pose_i = pose_graph.nodes_[i].pose_;
        // auto pose_j = pose_graph.nodes_[j].pose_;
        utility::LogInfo("corres {}{}", i, j);
        auto corres_ij = SLACPairwiseCorrespondence::ReadFromFile(corres_fname);

        // TODO: use parameterization to update normals and points per grid
    }
}

core::Tensor Solve(core::Tensor& AtA,
                   core::Tensor& Atb,
                   const SLACOptimizerOption& option) {
    utility::LogError("Unimplemented.");
    return Atb;
}

void UpdatePoses(const PoseGraph& fragment_pose_graph,
                 core::Tensor& result,
                 const SLACOptimizerOption& option) {
    utility::LogError("Unimplemented.");
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

    PoseGraph updated_pose_graph;

    // Fill-in
    // fragments x 6 (se3) + control_grids x 3 (R^3)
    int64_t num_params = fnames_down.size() * 6 + ctr_grid.Size() * 3;
    utility::LogInfo("Initializing {}^2 matrices", num_params);
    core::Tensor AtA({num_params, num_params}, core::Dtype::Float32, device);
    core::Tensor Atb({num_params, 1}, core::Dtype::Float32, device);
    for (int itr = 0; itr < option.max_iterations_; ++itr) {
        utility::LogInfo("Iteration {}", itr);
        FillInSLACAlignmentTerm(AtA, Atb, ctr_grid, fnames_down, pose_graph,
                                option);
        FillInSLACRegularizer(AtA, Atb, ctr_grid, option);

        core::Tensor delta = Solve(AtA, Atb, option);

        UpdatePoses(pose_graph, delta, option);
        UpdateControlGrid(ctr_grid, delta, option);

        utility::LogError("Unimplemented!");
    }
    return std::make_pair(updated_pose_graph, ctr_grid);
}

PoseGraph RunRigidOptimizerForFragments(const std::vector<std::string>& fnames,
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

    // Fill-in
    // fragments x 6 (se3)
    int64_t num_params = fnames_down.size() * 6;
    utility::LogInfo("Initializing {}^2 matrices", num_params);
    core::Tensor AtA({num_params, num_params}, core::Dtype::Float32, device);
    core::Tensor Atb({num_params, 1}, core::Dtype::Float32, device);
    for (int itr = 0; itr < option.max_iterations_; ++itr) {
        utility::LogInfo("Iteration {}", itr);
        FillInRigidAlignmentTerm(AtA, Atb, fnames_down, pose_graph, option);

        core::Tensor delta = Solve(AtA, Atb, option);
        UpdatePoses(pose_graph, delta, option);

        utility::LogError("Unimplemented!");
    }
    return pose_graph;
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
