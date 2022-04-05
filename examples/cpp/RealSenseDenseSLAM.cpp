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

#include "OnlineSLAMGUI.h"
#include "open3d/Open3D.h"

using namespace open3d;

//------------------------------------------------------------------------------
void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();

    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RealSenseDenseSLAMGUI [options]");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    [-V]");
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

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
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
    std::string config_file, bag_file;

    if (utility::ProgramOptionExists(argc, argv, "-c")) {
        config_file = utility::GetProgramOptionAsString(argc, argv, "-c");
    } else if (utility::ProgramOptionExists(argc, argv, "--config")) {
        config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    }
    if (utility::ProgramOptionExists(argc, argv, "--align")) {
        align_streams = true;
    }
    if (utility::ProgramOptionExists(argc, argv, "--record")) {
        bag_file = utility::GetProgramOptionAsString(argc, argv, "--record");
    }

    std::string device_code =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CUDA:0");
    if (device_code != "CPU:0" && device_code != "CUDA:0") {
        utility::LogWarning(
                "Unrecognized device {}. Expecting CPU:0 or CUDA:0.",
                device_code);
        return -1;
    }
    utility::LogInfo("Using device {}.", device_code);

    // Read in camera configuration.
    t::io::RealSenseSensorConfig rs_cfg;
    if (!config_file.empty())
        open3d::io::ReadIJsonConvertible(config_file, rs_cfg);

    // Initialize camera.
    t::io::RealSenseSensor rs;
    rs.ListDevices();
    rs.InitSensor(rs_cfg, 0, bag_file);
    utility::LogInfo("{}", rs.GetMetadata().ToString());
    rs.StartCapture();

    auto get_rgbd_image_input = [&](const int idx) {
        return rs.CaptureFrame(true, align_streams);
    };
    core::Tensor intrinsic_t = core::eigen_converter::EigenMatrixToTensor(
            rs.GetMetadata().intrinsics_.intrinsic_matrix_);
    std::unordered_map<std::string, double> default_params = {
            {"depth_scale", rs.GetMetadata().depth_scale_}};

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, const_cast<const char**>(argv));
    auto mono =
            app.AddFont(gui::FontDescription(gui::FontDescription::MONOSPACE));
    app.AddWindow(std::make_shared<ReconstructionWindow>(
            get_rgbd_image_input, intrinsic_t, default_params, device_code,
            mono));
    app.Run();

    rs.StopCapture();

    return 0;
}
