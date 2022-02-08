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

#include "open3d/Open3D.h"
#include "open3d/t/pipelines/slac/ControlGrid.h"

using namespace open3d;
using namespace open3d::core;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > SLACIntegrate [dataset_folder] [slac_folder] [options]");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --color_subfolder [default: color, rgb, image]");
    utility::LogInfo("    --depth_subfolder [default: depth]");
    utility::LogInfo("    --voxel_size [=0.0058 (m)]");
    utility::LogInfo("    --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("    --block_count [=40000]");
    utility::LogInfo("    --depth_scale [=1000.0]");
    utility::LogInfo("    --max_depth [=3.0]");
    utility::LogInfo("    --sdf_trunc [=0.04]");
    utility::LogInfo("    --device [CPU:0]");
    utility::LogInfo("    --mesh");
    utility::LogInfo("    --pointcloud");
    utility::LogInfo("    --debug");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string color_subfolder = utility::GetProgramOptionAsString(
            argc, argv, "--color_subfolder", "color");
    std::string depth_subfolder = utility::GetProgramOptionAsString(
            argc, argv, "--depth_subfolder", "depth");

    // Color and depth
    std::string dataset_folder = std::string(argv[1]);
    std::string color_folder = dataset_folder + "/" + color_subfolder;
    std::string depth_folder = dataset_folder + "/" + depth_subfolder;
    std::string fragment_folder = dataset_folder + "/fragments";
    std::vector<std::string> color_filenames;
    utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<std::string> depth_filenames;
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    if (color_filenames.size() != depth_filenames.size()) {
        utility::LogError("Number of color and depth files mismatch: {} vs {}.",
                          color_filenames.size(), depth_filenames.size());
    }

    // Optimized fragment pose graph
    std::string slac_folder = std::string(argv[2]);
    std::string posegraph_path =
            std::string(slac_folder + "/optimized_posegraph_slac.json");
    auto posegraph = io::CreatePoseGraphFromFile(posegraph_path);
    if (posegraph == nullptr) {
        utility::LogError(
                "Unable to open {}, please run SLAC before SLACIntegrate.",
                posegraph_path);
    }

    // Intrinsics
    std::string intrinsic_path = utility::GetProgramOptionAsString(
            argc, argv, "--intrinsic_path", "");
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    if (intrinsic_path.empty()) {
        utility::LogWarning("Using default Primesense intrinsics");
    } else if (!io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
        utility::LogError("Unable to convert json to intrinsics.");
    }
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t = Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});

    // Device
    std::string device_code = "CPU:0";
    if (utility::ProgramOptionExists(argc, argv, "--device")) {
        device_code = utility::GetProgramOptionAsString(argc, argv, "--device");
    }
    core::Device device(device_code);
    utility::LogInfo("Using device: {}", device.ToString());

    // Voxelgrid options
    int block_count =
            utility::GetProgramOptionAsInt(argc, argv, "--block_count", 40000);
    float voxel_size = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--voxel_size", 3.f / 512.f));
    float depth_scale = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float max_depth = static_cast<float>(
            utility::GetProgramOptionAsDouble(argc, argv, "--max_depth", 3.f));
    // float sdf_trunc = static_cast<float>(utility::GetProgramOptionAsDouble(
    //         argc, argv, "--sdf_trunc", 0.04f));
    t::geometry::VoxelBlockGrid voxel_grid(
            {"tsdf", "weight", "color"},
            {core::Dtype::Float32, core::Dtype::Float32, core::Dtype::Float32},
            {{1}, {1}, {3}}, voxel_size, 16, block_count, device);

    // Load control grid
    core::Tensor ctr_grid_keys =
            core::Tensor::Load(slac_folder + "/ctr_grid_keys.npy");
    core::Tensor ctr_grid_values =
            core::Tensor::Load(slac_folder + "/ctr_grid_values.npy");
    t::pipelines::slac::ControlGrid ctr_grid(3.0 / 8, ctr_grid_keys.To(device),
                                             ctr_grid_values.To(device),
                                             device);

    int k = 0;
    for (size_t i = 0; i < posegraph->nodes_.size(); ++i) {
        utility::LogInfo("Fragment: {}", i);
        auto fragment_pose_graph = *io::CreatePoseGraphFromFile(fmt::format(
                "{}/fragment_optimized_{:03d}.json", fragment_folder, i));
        for (auto node : fragment_pose_graph.nodes_) {
            Eigen::Matrix4d pose_local = node.pose_;
            Tensor extrinsic_local_t =
                    core::eigen_converter::EigenMatrixToTensor(
                            pose_local.inverse().eval());

            Eigen::Matrix4d pose = posegraph->nodes_[i].pose_ * node.pose_;
            Tensor extrinsic_t = core::eigen_converter::EigenMatrixToTensor(
                    pose.inverse().eval());

            auto depth =
                    t::io::CreateImageFromFile(depth_filenames[k])->To(device);
            auto color =
                    t::io::CreateImageFromFile(color_filenames[k])->To(device);
            t::geometry::RGBDImage rgbd(color, depth);

            utility::Timer timer;
            timer.Start();

            t::geometry::RGBDImage rgbd_projected =
                    ctr_grid.Deform(rgbd, intrinsic_t, extrinsic_local_t,
                                    depth_scale, max_depth);

            core::Tensor frustum_block_coords =
                    voxel_grid.GetUniqueBlockCoordinates(
                            rgbd_projected.depth_, intrinsic_t, extrinsic_t,
                            depth_scale, max_depth);

            voxel_grid.Integrate(frustum_block_coords, rgbd_projected.depth_,
                                 rgbd_projected.color_, intrinsic_t,
                                 extrinsic_t, depth_scale, max_depth);
            timer.Stop();

            ++k;
            utility::LogInfo("{}: Deformation + Integration takes {}", k,
                             timer.GetDuration());
        }
    }

    if (utility::ProgramOptionExists(argc, argv, "--mesh")) {
        auto mesh = voxel_grid.ExtractTriangleMesh();
        auto mesh_legacy =
                std::make_shared<geometry::TriangleMesh>(mesh.ToLegacy());
        open3d::io::WriteTriangleMesh("mesh_" + device.ToString() + ".ply",
                                      *mesh_legacy);
    }

    if (utility::ProgramOptionExists(argc, argv, "--pointcloud")) {
        auto pcd = voxel_grid.ExtractPointCloud();
        auto pcd_legacy =
                std::make_shared<open3d::geometry::PointCloud>(pcd.ToLegacy());
        open3d::io::WritePointCloud("pcd_" + device.ToString() + ".ply",
                                    *pcd_legacy);
    }
}
