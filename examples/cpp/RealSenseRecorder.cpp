// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <chrono>
#include <memory>
#include <string>

#include "open3d/Open3D.h"

using namespace open3d;
namespace tio = open3d::t::io;
namespace sc = std::chrono;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RealSenseRecorder [-V] [-l|--list-devices] [--align] [--record rgbd_video_file.bag] [-c|--config rs-config.json]");
    utility::LogInfo(
            "Open a RealSense camera and display live color and depth streams. You can set\n"
            "frame sizes and frame rates for each stream and the depth stream can be\n"
            "optionally aligned to the color stream. NOTE: An error of 'UNKNOWN: Couldn't\n"
            "resolve requests' implies  unsupported stream format settings.");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc <= 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    if (utility::ProgramOptionExists(argc, argv, "--list-devices") ||
        utility::ProgramOptionExists(argc, argv, "-l")) {
        tio::RealSenseSensor::ListDevices();
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

    // Read in camera configuration.
    tio::RealSenseSensorConfig rs_cfg;
    if (!config_file.empty()) {
        open3d::io::ReadIJsonConvertible(config_file, rs_cfg);
    }

    // Initialize camera.
    tio::RealSenseSensor rs;
    rs.ListDevices();
    rs.InitSensor(rs_cfg, 0, bag_file);
    utility::LogInfo("{}", rs.GetMetadata().ToString());

    // Create windows to show depth and color streams.
    bool flag_start = false, flag_record = flag_start, flag_exit = false;
    visualization::VisualizerWithKeyCallback depth_vis, color_vis;
    auto callback_exit = [&](visualization::Visualizer *vis) {
        flag_exit = true;
        if (flag_start) {
            utility::LogInfo("Recording finished.");
        } else {
            utility::LogInfo("Nothing has been recorded.");
        }
        return false;
    };
    depth_vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);
    color_vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);
    auto callback_toggle_record = [&](visualization::Visualizer *vis) {
        if (flag_record) {
            rs.PauseRecord();
            utility::LogInfo(
                    "Recording paused. "
                    "Press [SPACE] to continue. "
                    "Press [ESC] to save and exit.");
            flag_record = false;
        } else {
            rs.ResumeRecord();
            flag_record = true;
            if (!flag_start) {
                utility::LogInfo(
                        "Recording started. "
                        "Press [SPACE] to pause. "
                        "Press [ESC] to save and exit.");
                flag_start = true;
            } else {
                utility::LogInfo(
                        "Recording resumed, video may be discontinuous. "
                        "Press [SPACE] to pause. "
                        "Press [ESC] to save and exit.");
            }
        }
        return false;
    };
    if (!bag_file.empty()) {
        depth_vis.RegisterKeyCallback(GLFW_KEY_SPACE, callback_toggle_record);
        color_vis.RegisterKeyCallback(GLFW_KEY_SPACE, callback_toggle_record);
        utility::LogInfo(
                "In the visulizer window, "
                "press [SPACE] to start recording, "
                "press [ESC] to exit.");
    } else {
        utility::LogInfo("In the visulizer window, press [ESC] to exit.");
    }

    using legacyRGBDImage = open3d::geometry::RGBDImage;
    using legacyImage = open3d::geometry::Image;
    std::shared_ptr<legacyImage> depth_image_ptr, color_image_ptr;

    // Loop over frames from device
    legacyRGBDImage im_rgbd;
    bool is_geometry_added = false;
    size_t frame_id = 0;
    rs.StartCapture(flag_start);
    do {
        im_rgbd = rs.CaptureFrame(true, align_streams).ToLegacy();

        // Improve depth visualization by scaling
        /* im_rgbd.depth_.LinearTransform(0.25); */
        depth_image_ptr = std::shared_ptr<open3d::geometry::Image>(
                &im_rgbd.depth_, [](open3d::geometry::Image *) {});
        color_image_ptr = std::shared_ptr<open3d::geometry::Image>(
                &im_rgbd.color_, [](open3d::geometry::Image *) {});

        if (!is_geometry_added) {
            if (!depth_vis.CreateVisualizerWindow(
                        "Open3D || RealSense || Depth", depth_image_ptr->width_,
                        depth_image_ptr->height_, 15, 50) ||
                !depth_vis.AddGeometry(depth_image_ptr) ||
                !color_vis.CreateVisualizerWindow(
                        "Open3D || RealSense || Color", color_image_ptr->width_,
                        color_image_ptr->height_, 675, 50) ||
                !color_vis.AddGeometry(color_image_ptr)) {
                utility::LogError("Window creation failed!");
                return 0;
            }
            is_geometry_added = true;
        }

        depth_vis.UpdateGeometry();
        color_vis.UpdateGeometry();
        depth_vis.PollEvents();
        color_vis.PollEvents();
        depth_vis.UpdateRender();
        color_vis.UpdateRender();

        if (frame_id++ % 30 == 0) {
            utility::LogInfo("Time: {}s, Frame {}",
                             static_cast<double>(rs.GetTimestamp()) * 1e-6,
                             frame_id - 1);
        }
    } while (!flag_exit);

    rs.StopCapture();
    return 0;
}
