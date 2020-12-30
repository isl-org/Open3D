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
#include "open3d/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

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
        if (utility::filesystem::FileExists(fname_down)) continue;
        utility::LogInfo("{}", fname_down);

        auto pcd = io::CreatePointCloudFromFile(fname);
        auto pcd_down = pcd->VoxelDownSample(option.voxel_size_);
        pcd_down->EstimateNormals();

        io::WritePointCloud(fname_down, *pcd_down);
        fnames_down.emplace_back(fname_down);
    }

    return fnames_down;
}

/// Read pose graph containing loop closures and odometry to compute
/// correspondences.
void GetCorrespondencesForPointClouds(
        const std::vector<std::string>& fragment_down_fnames,
        const open3d::pipelines::registration::PoseGraph& pose_graph,
        const SLACOptimizerOption& option) {
    std::vector<SLACPairwiseCorrespondence> pair_corres;

    // Enumerate pose graph edges
    std::set<int> processed_pcd;

    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

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

        pair_corres.emplace_back(i, j, corres);
        std::string corres_fname = fmt::format("{}/{:03d}_{:03d}.corres",
                                               option.buffer_folder_, i, j);
        pair_corres.back().Write(corres_fname);
        utility::LogInfo("Edge: {:02d} -> {:02d}, corres {}", i, j,
                         corres.GetLength());

        // For IO debug
        auto corres_read =
                SLACPairwiseCorrespondence::ReadFromFile(corres_fname);
        utility::LogInfo("written = {}", corres.GetShape());
        utility::LogInfo("read = {}", corres_read.correspondence_.GetShape());
        if (!corres.AllClose(corres_read.correspondence_)) {
            utility::LogError("IO of correspondences mismatch");
        }

        // For visual debug
        pcd_i->Transform(pose_ij);
        visualization::DrawGeometries({pcd_i, pcd_j});
        pcd_i->Transform(pose_ij.inverse());
    }
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

    // Then obtain the correspondences given the pose graph
    auto fragment_down_fnames = PreprocessPointClouds(fragment_fnames, option);
    GetCorrespondencesForPointClouds(fragment_down_fnames, fragment_pose_graph,
                                     option);

    // // First initialize ctr_grid
    // ControlGrid ctr_grid;
    // for (auto fname : fragment_fnames) {
    //     auto pcd = io::ReadPointCloud(fname);
    //     ctr_grid.Touch(pcd);
    // }

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
