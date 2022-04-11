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

#include <chrono>
#include <memory>
#include <string>

#include "OnlineSLAMUtil.h"
#include "open3d/Open3D.h"

std::pair<std::vector<std::string>, std::vector<std::string>> LoadFilenames(
        const std::string dataset_path) {
    using namespace open3d;

    std::vector<std::string> rgb_candidates{"color", "image", "rgb"};
    std::vector<std::string> rgb_files;

    // Load rgb
    for (auto rgb_candidate : rgb_candidates) {
        const std::string rgb_dir = dataset_path + "/" + rgb_candidate;
        utility::filesystem::ListFilesInDirectoryWithExtension(rgb_dir, "jpg",
                                                               rgb_files);
        if (rgb_files.size() != 0) break;
        utility::filesystem::ListFilesInDirectoryWithExtension(rgb_dir, "png",
                                                               rgb_files);
        if (rgb_files.size() != 0) break;
    }
    if (rgb_files.size() == 0) {
        utility::LogError(
                "RGB images not found! Please ensure a folder named color, "
                "image, or rgb is in {}",
                dataset_path);
    }

    const std::string depth_dir = dataset_path + "/depth";
    std::vector<std::string> depth_files;
    utility::filesystem::ListFilesInDirectoryWithExtension(depth_dir, "png",
                                                           depth_files);
    if (depth_files.size() == 0) {
        utility::LogError(
                "Depth images not found! Please ensure a folder named "
                "depth is in {}",
                dataset_path);
    }

    if (depth_files.size() != rgb_files.size()) {
        utility::LogError(
                "Number of depth images ({}) and color image ({}) "
                "mismatch!",
                dataset_path);
    }

    std::sort(rgb_files.begin(), rgb_files.end());
    std::sort(depth_files.begin(), depth_files.end());
    return std::make_pair(rgb_files, depth_files);
}

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();

    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RGBDDenseSLAM [dataset_path] [options]");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    [-V]");
    utility::LogInfo("    [--intrinsic_path camera_intrinsic.json]");
    utility::LogInfo("    [--align]");
    utility::LogInfo("    [--device CUDA:0]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;
    using namespace open3d::visualization;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string dataset_path = argv[1];
    if (!utility::filesystem::DirectoryExists(dataset_path)) {
        utility::LogWarning(
                "Expected an existing directory, but {} does not exist.",
                dataset_path);
        return -1;
    }

    if (utility::ProgramOptionExists(argc, argv, "-V")) {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    } else {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
    }

    std::string intrinsic_path = utility::GetProgramOptionAsString(
            argc, argv, "--intrinsics_path", "");

    bool align_streams = false;
    if (utility::ProgramOptionExists(argc, argv, "--align")) {
        align_streams = true;
    }

    std::string device_code =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CUDA:0");
    core::Device device(device_code);
    utility::LogInfo("Using device {}.", device_code);

    // Load files
    std::vector<std::string> rgb_files, depth_files;
    std::tie(rgb_files, depth_files) = LoadFilenames(dataset_path);

    // Load intrinsics (if provided)
    // Default
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    if (intrinsic_path.empty()) {
        utility::LogInfo("Using Primesense default intrinsics.");
    } else if (!io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
        utility::LogWarning(
                "Failed to load {}, using Primesense default intrinsics.",
                intrinsic_path);
    } else {
        utility::LogInfo("Loaded intrinsics from {}.", intrinsic_path);
    }
    core::Tensor intrinsic_t = core::eigen_converter::EigenMatrixToTensor(
            intrinsic.intrinsic_matrix_);

    const size_t max_idx = depth_files.size();
    auto get_rgbd_image_input = [&](const size_t idx) {
        if (idx < max_idx) {
            t::geometry::Image depth =
                    *t::io::CreateImageFromFile(depth_files[idx]);
            t::geometry::Image color =
                    *t::io::CreateImageFromFile(rgb_files[idx]);
            t::geometry::RGBDImage rgbd_im(color, depth, align_streams);
            return rgbd_im;
        } else {
            // Return empty image to indicate EOF.
            return t::geometry::RGBDImage();
        }
    };

    std::unordered_map<std::string, double> default_params = {
            {"depth_scale", 1000}};

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, const_cast<const char**>(argv));
    auto mono =
            app.AddFont(gui::FontDescription(gui::FontDescription::MONOSPACE));
    app.AddWindow(std::make_shared<examples::online_slam::ReconstructionWindow>(
            get_rgbd_image_input, intrinsic_t, default_params, device, mono));
    app.Run();

    return 0;
}
