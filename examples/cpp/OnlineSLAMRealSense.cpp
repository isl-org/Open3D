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

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();

    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > OnlineSLAMRealSense [options]");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    [-V]");
    utility::LogInfo("    [--use_bag_file /path/to/realsense_video_file.bag] If not provided, it will look for realsense sensor.");
    utility::LogInfo("    [-l|--list-devices]");
    utility::LogInfo("    [--align]");
    utility::LogInfo("    [--record rgbd_video_file.bag]");
    utility::LogInfo("    [-c|--config rs-config.json]");
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

    std::string bag_file;
    bool use_bag_file = false;
    if (utility::ProgramOptionExists(argc, argv, "--use_bag_file")) {
        bag_file =
                utility::GetProgramOptionAsString(argc, argv, "--use_bag_file");
    }
    if (!bag_file.empty()) {
        use_bag_file = true;
    }

    if (utility::ProgramOptionExists(argc, argv, "--list-devices") ||
        utility::ProgramOptionExists(argc, argv, "-l")) {
        t::io::RealSenseSensor::ListDevices();
        return 0;
    }
    if (utility::ProgramOptionExists(argc, argv, "-V")) {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    } else {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
    }
    bool align_streams = false;
    std::string config_file, record_to_bag_file;

    if (utility::ProgramOptionExists(argc, argv, "-c")) {
        config_file = utility::GetProgramOptionAsString(argc, argv, "-c");
    } else if (utility::ProgramOptionExists(argc, argv, "--config")) {
        config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    }
    if (utility::ProgramOptionExists(argc, argv, "--align")) {
        align_streams = true;
    }
    if (utility::ProgramOptionExists(argc, argv, "--record")) {
        record_to_bag_file =
                utility::GetProgramOptionAsString(argc, argv, "--record");
    }

    std::string device_code =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CUDA:0");
    core::Device device(device_code);
    utility::LogInfo("Using device {}.", device_code);

    std::function<t::geometry::RGBDImage(const int)> get_rgbd_image_input;
    core::Tensor intrinsic_t;
    std::unordered_map<std::string, double> default_params;
    t::io::RealSenseSensor rs;
    t::io::RSBagReader bag_reader;

    if (!use_bag_file) {
        // Read in camera configuration.
        t::io::RealSenseSensorConfig rs_cfg;
        if (!config_file.empty())
            open3d::io::ReadIJsonConvertible(config_file, rs_cfg);

        // Initialize camera.
        rs.ListDevices();
        rs.InitSensor(rs_cfg, 0, record_to_bag_file);
        utility::LogInfo("{}", rs.GetMetadata().ToString());
        rs.StartCapture();

        get_rgbd_image_input = [&](const size_t idx) {
            return rs.CaptureFrame(true, align_streams);
        };
        intrinsic_t = core::eigen_converter::EigenMatrixToTensor(
                rs.GetMetadata().intrinsics_.intrinsic_matrix_);
        default_params = {{"depth_scale", rs.GetMetadata().depth_scale_}};
    } else {
        // Use Bag File.
        bag_reader.Open(bag_file);

        if (!bag_reader.IsOpened()) {
            utility::LogError("Unable to open {}", bag_file);
            return 1;
        }
        const auto bag_metadata = bag_reader.GetMetadata();
        utility::LogInfo("{}", bag_metadata.ToString());

        auto next_frame = bag_reader.NextFrame();
        get_rgbd_image_input = [&](const int idx) {
            if (!bag_reader.IsEOF()) {
                return bag_reader.NextFrame();
            } else {
                // Return empty image.
                return open3d::t::geometry::RGBDImage();
            }
        };

        intrinsic_t = core::eigen_converter::EigenMatrixToTensor(
                bag_metadata.intrinsics_.intrinsic_matrix_);
        default_params = {{"depth_scale", bag_metadata.depth_scale_}};
    }

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, const_cast<const char**>(argv));
    auto mono =
            app.AddFont(gui::FontDescription(gui::FontDescription::MONOSPACE));
    app.AddWindow(std::make_shared<examples::online_slam::ReconstructionWindow>(
            get_rgbd_image_input, intrinsic_t, default_params, device, mono));
    app.Run();

    if (!use_bag_file) {
        rs.StopCapture();
    } else {
        bag_reader.Close();
    }

    return 0;
}
