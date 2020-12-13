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

#include <librealsense2/rs.hpp>
#include <memory>
#include <string>

#include "open3d/Open3D.h"
#include "open3d/t/io/sensor/realsense/RealSenseSensor.h"

using namespace open3d;
using namespace open3d::t::io;

void PrintUsage() {
    PrintOpen3DVersion();
    utility::LogInfo(
            "Open a RealSense camera and display live color and depth "
            "streams.\n"
            "You can set frame sizes and frame rates for each stream and the\n"
            "depth stream can be optionally aligned to the color stream.\n"
            "NOTE: An error of 'Couldn't resolve requests' implies "
            "unsupported\n"
            "stream format settings.");
    utility::LogInfo("Usage:");
    utility::LogInfo(
            "RealSenseRecorder [-h|--help] [-l|--list-devices] [--align]\n"
            "[--record rgbd_video_file.bag] [-c|--config rs-config.json]");
}

int main(int argc, char **argv) {
    // Parse command line arguments
    if (utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintUsage();
        return 0;
    }
    if (utility::ProgramOptionExists(argc, argv, "--list-devices") ||
        utility::ProgramOptionExists(argc, argv, "-l")) {
        RealSenseSensor::ListDevices();
        return 0;
    }
    bool align_streams = false;
    /* visualization::gui::Size color_size(0, 0), depth_size(0, 0); */
    /* int color_fps = 0, depth_fps = 0; */
    std::string config_file, bag_file;

    if (utility::ProgramOptionExists(argc, argv, "-c"))
        config_file = utility::GetProgramOptionAsString(argc, argv, "-c");
    else if (utility::ProgramOptionExists(argc, argv, "--config"))
        config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    else {
        utility::LogError("config json file required.");
        PrintUsage();
        return 1;
    }
    if (utility::ProgramOptionExists(argc, argv, "--align")) {
        align_streams = true;
    }
    if (utility::ProgramOptionExists(argc, argv, "--record"))
        bag_file = utility::GetProgramOptionAsString(argc, argv, "--record");

    RealSenseSensorConfig rs_cfg;
    open3d::io::ReadIJsonConvertible(config_file, rs_cfg);

    RealSenseSensor rs;
    rs.InitSensor(rs_cfg, 0, bag_file);

    /* // Get device details */
    /* utility::LogInfo("Using device 0, an {}", */
    /*                  rs_device.get_info(RS2_CAMERA_INFO_NAME)); */
    /* utility::LogInfo("    Serial number: {}", */
    /*                  rs_device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)); */
    /* utility::LogInfo("    Firmware version: {}", */
    /*                  rs_device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION));
     */
    /* utility::LogInfo(""); */

    /* // Get stream configuration */
    /* const auto rs_depth = profile.get_stream(RS2_STREAM_DEPTH) */
    /*                               .as<rs2::video_stream_profile>(); */
    /* const auto rs_color = profile.get_stream(RS2_STREAM_COLOR) */
    /*                               .as<rs2::video_stream_profile>(); */
    /* rs2_extrinsics extr_depth2color = rs_depth.get_extrinsics_to(rs_color);
     */
    /* utility::LogInfo("depth->color extrinsics: Rotation"); */
    /* for (int i = 0; i < 9; i++) { */
    /*     utility::LogInfo("{:.6f} ", extr_depth2color.rotation[i]); */
    /* } */
    /* utility::LogInfo("depth->color extrinsics: Translation"); */
    /* for (int i = 0; i < 3; i++) { */
    /*     utility::LogInfo("{:.6f} ", extr_depth2color.translation[i]); */
    /* } */
    /* utility::LogInfo(""); */

    /* for (const auto &rs_stream : {rs_depth, rs_color}) { */
    /*     rs2_intrinsics intr = rs_stream.get_intrinsics(); */
    /*     if (rs_stream.unique_id() == rs_depth.unique_id()) */
    /*         depth_size = {intr.width, intr.height}; */
    /*     else if (rs_stream.unique_id() == rs_color.unique_id()) */
    /*         color_size = {intr.width, intr.height}; */
    /*     utility::LogInfo("Instrinsics for stream {}",
     * rs_stream.stream_name()); */
    /*     utility::LogInfo("{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}", intr.width,
     */
    /*                      intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy);
     */
    /*     for (int i = 0; i < 5; i++) { */
    /*         utility::LogInfo("{:.6f} ", intr.coeffs[i]); */
    /*     } */
    /*     utility::LogInfo(""); */
    /* } */

    // Create windows to show depth and color streams
    bool flag_record = false, flag_start = false, flag_exit = false;
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
    } else
        utility::LogInfo("In the visulizer window, press [ESC] to exit.");

    using legacyRGBDImage = open3d::geometry::RGBDImage;
    using legacyImage = open3d::geometry::Image;
    std::shared_ptr<legacyImage> depth_image_ptr, color_image_ptr;

    // Loop over frames from device
    legacyRGBDImage im_rgbd;
    bool is_geometry_added = false;
    size_t frame_id = 0;
    rs.StartCapture(false);
    do {
        im_rgbd = rs.CaptureFrame(true, align_streams).ToLegacyRGBDImage();

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

        if (frame_id++ % 30 == 0) utility::LogInfo("Frame {}", frame_id - 1);

    } while (!flag_exit);

    rs.StopCapture();
    return 0;
}
