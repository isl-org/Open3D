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

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

struct SLACPairwiseCorrespondence {
    // Source index
    int i_;
    // Target index
    int j_;

    // N x 2 for point clouds, storing corresponding point indices;
    // N x 4 for RGBD images, storing corresponding uv coordinates.
    core::Tensor correspondence_;
};

/// Read pose graph containing loop closures and odometry to compute
/// correspondences.
std::vector<SLACPairwiseCorrespondence> GetCorrespondencesForPointClouds(
        const std::vector<std::string>& fragment_fnames,
        const open3d::pipelines::registration::PoseGraph& fragment_pose_graph) {
    // Enumerate pose graph edges
    // Read fragments
    // Use EvaluateRegistration to obtain correspondences
    utility::LogError("Unimplemented.");
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
    // Then obtain the correspondences given the pose graph
    auto pairs = GetCorrespondencesForPointClouds(fragment_fnames,
                                                  fragment_pose_graph);

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
