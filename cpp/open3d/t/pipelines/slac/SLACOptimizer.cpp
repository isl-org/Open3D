// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/EigenConverter.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/t/pipelines/slac/FillInLinearSystemImpl.h"
#include "open3d/utility/FileSystem.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {
using t::geometry::PointCloud;

// Write point clouds to disk after preprocessing (remove outliers,
// estimate normals, etc).
static std::vector<std::string> PreprocessPointClouds(
        const std::vector<std::string>& fnames,
        const SLACOptimizerParams& params) {
    std::string subdir_name = params.GetSubfolderName();
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
        if (pcd == nullptr) {
            utility::LogError("Internal error: pcd is nullptr.");
        }

        // Pre-processing input pointcloud.
        if (params.voxel_size_ > 0) {
            pcd = pcd->VoxelDownSample(params.voxel_size_);
            pcd->RemoveStatisticalOutliers(20, 2.0);
            pcd->EstimateNormals();
        } else {
            pcd->RemoveStatisticalOutliers(20, 2.0);
            if (!pcd->HasNormals()) {
                pcd->EstimateNormals();
            }
        }

        io::WritePointCloud(fname_processed, *pcd);
        utility::LogInfo("Saving processed point cloud {}", fname_processed);
    }

    return fnames_processed;
}

// The correspondences from NNS has the value as the target index
// correspondening the source point which is index of the value itself.
// Therefore, for target_indexed_correspondences = {2, 3, -1 , 4}.
// (source, target) correspondences are: {{0, 2}, {1, 3}, {3, 4}}.
//
// For convenience to access
// source and target pointcloud indexed by their correspondences, this
// function converts {N, 1} shaped target_indices correspondences to {C, 2}
// shaped CorrespondenceSet, where C is the number of correspondences such that
//
// For getting correspondence indexed pointclouds:
//  source_indexed_pcd = source.GetPointPositions()
//                            .IndexGet({correspondence_set.T()[0]});
//  target_indexed_pcd = target.GetPointPositions()
//                            .IndexGet({correspondence_set.T()[1]});
//
// For getting the i-th correspondence pair:
//  correspondence_pair_i = make_pair(correspondence[i][0],
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
            core::Tensor::Arange(0, N, 1, core::Int64, device)
                    .IndexGet({valid_correspondences})
                    .Reshape({C, 1});

    // Creating {C, 2} shaped tensor by horizontal stacking {source_indices,
    // target_indices}.
    core::Tensor correspondence_set({C, 2}, core::Int64, device);
    correspondence_set.SetItem(
            {core::TensorKey::Slice(0, C, 1), core::TensorKey::Slice(0, 1, 1)},
            source_indices);
    correspondence_set.SetItem(
            {core::TensorKey::Slice(0, C, 1), core::TensorKey::Slice(1, 2, 1)},
            target_indices);

    return correspondence_set;
}

// Aggressive pruning -- reject any suspicious pair
//
// tpcd_i is the source pointcloud, tpcd_j is the target pointcloud,
// T_i is the transformation_source_to_world,
// T_j is the transformation_target_to_world,
// T_ij is the transformation_source_to_target.
// distance_threshold is the search_distance for NNS.
// i and j are the indices of source and target pointcloud respectively.
static core::Tensor GetCorrespondenceSetForPointCloudPair(
        int i,
        int j,
        PointCloud& tpcd_i,
        PointCloud& tpcd_j,
        const core::Tensor& T_i,
        const core::Tensor& T_j,
        const core::Tensor& T_ij,
        float distance_threshold,
        float fitness_threshold,
        bool debug) {
    core::Device device = tpcd_i.GetDevice();
    core::Dtype dtype = tpcd_i.GetPointPositions().GetDtype();

    core::AssertTensorDevice(tpcd_j.GetPointPositions(), device);
    core::AssertTensorDtype(tpcd_j.GetPointPositions(), dtype);

    core::AssertTensorShape(T_i, {4, 4});
    core::AssertTensorShape(T_j, {4, 4});
    core::AssertTensorShape(T_ij, {4, 4});

    PointCloud tpcd_i_transformed_Tij = tpcd_i.Clone();
    tpcd_i_transformed_Tij.Transform(T_ij);

    // Obtain correspondence via nns, between tpcd_i_transformed_Tij and tpcd_j.
    core::nns::NearestNeighborSearch tpcd_j_nns(tpcd_j.GetPointPositions());
    bool check = tpcd_j_nns.HybridIndex(distance_threshold);
    if (!check) {
        utility::LogError(
                "[NearestNeighborSearch::HybridSearch] Index is not set.");
    }
    core::Tensor target_indices, residual_distances_Tij, neighbour_counts;
    std::tie(target_indices, residual_distances_Tij, neighbour_counts) =
            tpcd_j_nns.HybridSearch(tpcd_i_transformed_Tij.GetPointPositions(),
                                    distance_threshold, 1);

    target_indices = target_indices.To(core::Int64);

    // Get the correspondence_set Transformed of shape {C, 2}.
    core::Tensor correspondence_set =
            ConvertCorrespondencesTargetIndexedToCx2Form(target_indices);

    // Get correspondence indexed pointcloud.
    PointCloud tpcd_i_indexed(
            tpcd_i.GetPointPositions().IndexGet({correspondence_set.T()[0]}));
    PointCloud tpcd_j_indexed(
            tpcd_j.GetPointPositions().IndexGet({correspondence_set.T()[1]}));

    // Inlier Ratio is calculated on pointclouds transformed by their pose in
    // model frame, to reject any suspicious pair.
    tpcd_i_indexed.Transform(T_i);
    tpcd_j_indexed.Transform(T_j);

    core::Tensor residual = (tpcd_i_indexed.GetPointPositions() -
                             tpcd_j_indexed.GetPointPositions());
    core::Tensor square_residual = (residual * residual).Sum({1});
    core::Tensor inliers =
            square_residual.Le(distance_threshold * distance_threshold);

    int64_t num_inliers = inliers.To(core::Int64).Sum({0}).Item<int64_t>();

    float inlier_ratio = static_cast<float>(num_inliers) /
                         static_cast<float>(inliers.GetLength());

    utility::LogDebug("Tij and (Ti, Tj) compatibility ratio = {}.",
                      inlier_ratio);

    if (j != i + 1 && inlier_ratio < fitness_threshold) {
        if (debug) {
            VisualizePointCloudCorrespondences(
                    tpcd_i, tpcd_j, correspondence_set,
                    T_j.Inverse().Matmul(T_i).To(device, dtype));
        }
        return core::Tensor();
    }

    return correspondence_set;
}

// Read pose graph containing loop closures and odometry to compute
// correspondences.
void SaveCorrespondencesForPointClouds(
        const std::vector<std::string>& fnames_processed,
        const PoseGraph& pose_graph,
        const SLACOptimizerParams& params,
        const SLACDebugOption& debug_option) {
    // Enumerate pose graph edges.
    for (auto& edge : pose_graph.edges_) {
        int i = edge.source_node_id_;
        int j = edge.target_node_id_;

        utility::LogInfo("Processing {:02d} -> {:02d}", i, j);

        std::string correspondences_fname = fmt::format(
                "{}/{:03d}_{:03d}.npy", params.GetSubfolderName(), i, j);
        if (utility::filesystem::FileExists(correspondences_fname)) continue;

        PointCloud tpcd_i =
                CreateTPCDFromFile(fnames_processed[i], params.device_);
        PointCloud tpcd_j =
                CreateTPCDFromFile(fnames_processed[j], params.device_);

        // pose of i in model frame.
        core::Tensor T_i = core::eigen_converter::EigenMatrixToTensor(
                pose_graph.nodes_[i].pose_);
        // pose of j in model frame.
        core::Tensor T_j = core::eigen_converter::EigenMatrixToTensor(
                pose_graph.nodes_[j].pose_);
        // transformation of i to j.
        core::Tensor T_ij = core::eigen_converter::EigenMatrixToTensor(
                edge.transformation_);

        // Get correspondences.
        core::Tensor correspondence_set = GetCorrespondenceSetForPointCloudPair(
                i, j, tpcd_i, tpcd_j, T_i, T_j, T_ij,
                params.distance_threshold_, params.fitness_threshold_,
                debug_option.debug_);

        if (correspondence_set.GetLength() > 0) {
            correspondence_set.Save(correspondences_fname);
            utility::LogInfo("Saving {} corres for {:02d} -> {:02d}",
                             correspondence_set.GetLength(), i, j);
        }
    }
}

static void InitializeControlGrid(ControlGrid& ctr_grid,
                                  const std::vector<std::string>& fnames) {
    core::Device device(ctr_grid.GetDevice());
    for (auto& fname : fnames) {
        utility::LogInfo("Initializing grid for {}", fname);

        auto tpcd = CreateTPCDFromFile(fname, device);
        ctr_grid.Touch(tpcd);
    }
    utility::LogInfo("Initialization finished.");
}

static void UpdatePoses(PoseGraph& fragment_pose_graph, core::Tensor& delta) {
    core::Tensor delta_poses = delta.View({-1, 6}).To(core::Device("CPU:0"));

    if (delta_poses.GetLength() != int64_t(fragment_pose_graph.nodes_.size())) {
        utility::LogError("Dimension Mismatch");
    }
    for (int64_t i = 0; i < delta_poses.GetLength(); ++i) {
        core::Tensor pose_delta = kernel::PoseToTransformation(delta_poses[i]);
        core::Tensor pose_tensor =
                pose_delta.Matmul(core::eigen_converter::EigenMatrixToTensor(
                                          fragment_pose_graph.nodes_[i].pose_)
                                          .To(core::Float32));

        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> pose_eigen =
                core::eigen_converter::TensorToEigenMatrixXf(pose_tensor);
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> pose_eigen_d =
                pose_eigen.cast<double>().eval();
        Eigen::Ref<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> pose_eigen_ref(
                pose_eigen_d);
        fragment_pose_graph.nodes_[i].pose_ = pose_eigen_ref;
    }
}

static void UpdateControlGrid(ControlGrid& ctr_grid, core::Tensor& delta) {
    core::Tensor delta_cgrids = delta.View({-1, 3});
    if (delta_cgrids.GetLength() != int64_t(ctr_grid.Size())) {
        utility::LogError("Dimension Mismatch");
    }

    ctr_grid.GetCurrPositions().Slice(0, 0, ctr_grid.Size()) += delta_cgrids;
}

std::pair<PoseGraph, ControlGrid> RunSLACOptimizerForFragments(
        const std::vector<std::string>& fnames,
        const PoseGraph& pose_graph,
        const SLACOptimizerParams& params,
        const SLACDebugOption& debug_option) {
    core::Device device(params.device_);
    if (!params.slac_folder_.empty()) {
        utility::filesystem::MakeDirectory(params.slac_folder_);
    }

    // First preprocess the point cloud with downsampling and normal
    // estimation.
    auto fnames_down = PreprocessPointClouds(fnames, params);
    // Then obtain the correspondences given the pose graph
    SaveCorrespondencesForPointClouds(fnames_down, pose_graph, params,
                                      debug_option);

    // First initialize the ctr_grid.
    // grid size = 3.0 / 8: recommended by the original implementation
    // https://github.com/qianyizh/ElasticReconstruction
    // grid count = 8000: empirical value, will be increased dynamically if
    // exceeded.
    ControlGrid ctr_grid(3.0 / 8, 8000, device);
    InitializeControlGrid(ctr_grid, fnames_down);
    ctr_grid.Compactify();

    // Fill-in
    // fragments x 6 (se3) + control_grids x 3 (R^3)
    int64_t num_params = fnames_down.size() * 6 + ctr_grid.Size() * 3;
    utility::LogInfo("Initializing the {}^2 Hessian matrix", num_params);

    PoseGraph pose_graph_update(pose_graph);
    for (int itr = 0; itr < params.max_iterations_; ++itr) {
        utility::LogInfo("Iteration {}", itr);
        core::Tensor AtA = core::Tensor::Zeros({num_params, num_params},
                                               core::Float32, device);
        core::Tensor Atb =
                core::Tensor::Zeros({num_params, 1}, core::Float32, device);

        core::Tensor indices_eye0 =
                core::Tensor::Arange(0, 6, 1, core::Int64, device);
        AtA.IndexSet({indices_eye0, indices_eye0},
                     core::Tensor::Ones({}, core::Float32, device));

        core::Tensor residual_data =
                core::Tensor::Zeros({1}, core::Float32, device);
        FillInSLACAlignmentTerm(AtA, Atb, residual_data, ctr_grid, fnames_down,
                                pose_graph_update, params, debug_option);

        utility::LogInfo("Alignment loss = {}", residual_data[0].Item<float>());

        core::Tensor residual_reg =
                core::Tensor::Zeros({1}, core::Float32, device);
        FillInSLACRegularizerTerm(AtA, Atb, residual_reg, ctr_grid,
                                  pose_graph_update.nodes_.size(), params,
                                  debug_option);
        utility::LogInfo("Regularizer loss = {}",
                         residual_reg[0].Item<float>());

        core::Tensor delta = AtA.Solve(Atb.Neg());

        core::Tensor delta_poses =
                delta.Slice(0, 0, 6 * pose_graph_update.nodes_.size());
        core::Tensor delta_cgrids = delta.Slice(
                0, 6 * pose_graph_update.nodes_.size(), delta.GetLength());

        UpdatePoses(pose_graph_update, delta_poses);
        UpdateControlGrid(ctr_grid, delta_cgrids);
    }
    return std::make_pair(pose_graph_update, ctr_grid);
}

PoseGraph RunRigidOptimizerForFragments(const std::vector<std::string>& fnames,
                                        const PoseGraph& pose_graph,
                                        const SLACOptimizerParams& params,
                                        const SLACDebugOption& debug_option) {
    core::Device device(params.device_);
    if (!params.slac_folder_.empty()) {
        utility::filesystem::MakeDirectory(params.slac_folder_);
    }

    // First preprocess the point cloud with downsampling and normal
    // estimation.
    std::vector<std::string> fnames_down =
            PreprocessPointClouds(fnames, params);
    // Then obtain the correspondences given the pose graph
    SaveCorrespondencesForPointClouds(fnames_down, pose_graph, params,
                                      debug_option);

    // Fill-in
    // fragments x 6 (se3)
    int64_t num_params = fnames_down.size() * 6;
    utility::LogInfo("Initializing the {}^2 Hessian matrix.", num_params);

    PoseGraph pose_graph_update(pose_graph);
    for (int itr = 0; itr < params.max_iterations_; ++itr) {
        utility::LogInfo("Iteration {}", itr);
        core::Tensor AtA = core::Tensor::Zeros({num_params, num_params},
                                               core::Float32, device);
        core::Tensor Atb =
                core::Tensor::Zeros({num_params, 1}, core::Float32, device);
        core::Tensor residual = core::Tensor::Zeros({1}, core::Float32, device);

        // Fix pose 0
        core::Tensor indices_eye0 = core::Tensor::Arange(0, 6, 1);
        AtA.IndexSet({indices_eye0, indices_eye0},
                     1e5 * core::Tensor::Ones({}, core::Float32, device));

        FillInRigidAlignmentTerm(AtA, Atb, residual, fnames_down,
                                 pose_graph_update, params, debug_option);
        utility::LogInfo("Loss = {}", residual[0].Item<float>());

        core::Tensor delta = AtA.Solve(Atb.Neg());
        UpdatePoses(pose_graph_update, delta);
    }

    return pose_graph_update;
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
