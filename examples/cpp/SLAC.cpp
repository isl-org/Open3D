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
#include "open3d/Open3D.h"
#include "open3d/t/pipelines/slac/ControlGrid.h"

using namespace open3d;
using namespace open3d::core;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    SLAC [dataset_folder] [options]");
    utility::LogInfo("--method [default: rigid, optional: slac]");
    utility::LogInfo("--voxel_size [default: 0.05]");
    utility::LogInfo("--device [default: CPU:0]");
    utility::LogInfo("--debug");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char** argv) {
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc < 2) {
        PrintHelp();
        return 1;
    }

    // Color and depth
    std::string dataset_folder = std::string(argv[1]);
    std::string fragment_folder = dataset_folder + "/fragments";
    std::string scene_folder = dataset_folder + "/scene";
    std::string slac_folder = dataset_folder + "/slac";

    std::vector<std::string> fragment_fnames;
    utility::filesystem::ListFilesInDirectoryWithExtension(
            fragment_folder, "ply", fragment_fnames);
    std::sort(fragment_fnames.begin(), fragment_fnames.end());

    std::string pose_graph_fname =
            scene_folder + "/refined_registration_optimized.json";
    auto pose_graph = io::CreatePoseGraphFromFile(pose_graph_fname);

    auto option = t::pipelines::slac::SLACOptimizerOption();
    option.buffer_folder_ = slac_folder;
    option.device_ =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CPU:0");
    option.voxel_size_ =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", 0.05);
    option.correspondence_debug_ =
            utility::ProgramOptionExists(argc, argv, "--debug");
    option.grid_debug_ = utility::ProgramOptionExists(argc, argv, "--debug");

    std::string method =
            utility::GetProgramOptionAsString(argc, argv, "--method", "rigid");

    pipelines::registration::PoseGraph pose_graph_updated;
    if ("rigid" == method) {
        pose_graph_updated = t::pipelines::slac::RunRigidOptimizerForFragments(
                fragment_fnames, *pose_graph, option);
    } else if ("slac" == method) {
        t::pipelines::slac::ControlGrid control_grid;
        std::tie(pose_graph_updated, control_grid) =
                t::pipelines::slac::RunSLACOptimizerForFragments(
                        fragment_fnames, *pose_graph, option);

        auto hashmap = control_grid.GetHashmap();
        core::Tensor active_addrs;
        hashmap->GetActiveIndices(active_addrs);
        hashmap->GetKeyTensor()
                .IndexGet({active_addrs.To(core::Dtype::Int64)})
                .Save(option.GetSubfolderName() + "/ctr_grid_keys.npy");
        hashmap->GetValueTensor()
                .IndexGet({active_addrs.To(core::Dtype::Int64)})
                .Save(option.GetSubfolderName() + "/ctr_grid_values.npy");
    }

    // Write pose graph
    io::WritePoseGraph(option.GetSubfolderName() + "/optimized_posegraph.json",
                       pose_graph_updated);

    camera::PinholeCameraTrajectory trajectory;
    for (size_t i = 0; i < pose_graph_updated.nodes_.size(); ++i) {
        auto fragment_pose_graph = io::CreatePoseGraphFromFile(fmt::format(
                "{}/fragment_optimized_{:03d}.json", fragment_folder, i));
        for (auto node : fragment_pose_graph->nodes_) {
            auto pose = pose_graph_updated.nodes_[i].pose_ * node.pose_;
            camera::PinholeCameraParameters param;
            param.extrinsic_ = pose.inverse().eval();
            trajectory.parameters_.push_back(param);
        }
    }
    io::WritePinholeCameraTrajectory(
            option.GetSubfolderName() + "/optimized_trajectory.log",
            trajectory);

    return 0;
}
