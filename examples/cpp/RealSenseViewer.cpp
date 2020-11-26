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

// TODO: Add support for recording

#include <librealsense2/rs.hpp>

#include "open3d/Open3D.h"

using namespace open3d;

void PrintUsage() {
    PrintOpen3DVersion();
    utility::LogInfo(
            "Open a RealSense camera and display live color and depth streams. "
            "You can set frame sizes and frame rates for each stream and the "
            "depth stream can be optionally aligned to the color stream."
            "NOTE: An error of 'Couldn't resolve requests' implies unsupported "
            "stream format settings.");
    utility::LogInfo("Usage:");
    utility::LogInfo(
            "RealSenseBagReader [-h|--help] [--align] "
            "[--depth-stream (WIDTH,HEIGHT,FPS)] "
            "[--color-stream (WIDTH,HEIGHT,FPS)]");
}

int main(int argc, char **argv) {
    // Parse command line arguments
    if (utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintUsage();
        return 0;
    }
    bool align_streams = false;
    visualization::gui::Size color_size(0, 0), depth_size(0, 0);
    int color_fps = 0, depth_fps = 0;

    if (utility::ProgramOptionExists(argc, argv, "--align")) {
        align_streams = true;
    }
    if (utility::ProgramOptionExists(argc, argv, "--depth-stream")) {
        auto depth_stream_options = utility::GetProgramOptionAsEigenVectorXd(
                argc, argv, "--depth-stream", Eigen::VectorXd::Zero(3, 1));
        depth_size = {static_cast<int>(depth_stream_options[0]),
                      static_cast<int>(depth_stream_options[1])};
        depth_fps = static_cast<int>(depth_stream_options[2]);
    }
    if (utility::ProgramOptionExists(argc, argv, "--color-stream")) {
        auto color_stream_options = utility::GetProgramOptionAsEigenVectorXd(
                argc, argv, "--color-stream", Eigen::VectorXd::Zero(3, 1));
        color_size = {static_cast<int>(color_stream_options[0]),
                      static_cast<int>(color_stream_options[1])};
        color_fps = static_cast<int>(color_stream_options[2]);
    }

    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    rs2::config cfg;
    // Select stream type, stream frame size, pixel format and frame rate.
    cfg.enable_stream(RS2_STREAM_DEPTH, depth_size.width, depth_size.height,
                      RS2_FORMAT_Z16, depth_fps);
    cfg.enable_stream(RS2_STREAM_COLOR, color_size.width, color_size.height,
                      RS2_FORMAT_RGB8, color_fps);
    rs2::pipeline_profile profile;
    try {
        profile = pipe.start(cfg);
    } catch (const rs2::error &e) {
        utility::LogError(
                "Could not start capture from RealSense camera!\n"
                "Reason: {}: {}\n",
                e.get_type(), e.what());
    }

    // Get device details
    const auto rs_device = profile.get_device();
    utility::LogInfo("Using device 0, an {}",
                     rs_device.get_info(RS2_CAMERA_INFO_NAME));
    utility::LogInfo("    Serial number: {}",
                     rs_device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    utility::LogInfo("    Firmware version: {}",
                     rs_device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION));
    utility::LogInfo("");

    // Get stream configuration
    const auto rs_depth = profile.get_stream(RS2_STREAM_DEPTH)
                                  .as<rs2::video_stream_profile>();
    const auto rs_color = profile.get_stream(RS2_STREAM_COLOR)
                                  .as<rs2::video_stream_profile>();
    rs2_extrinsics extr_depth2color = rs_depth.get_extrinsics_to(rs_color);
    utility::LogInfo("depth->color extrinsics: Rotation");
    for (int i = 0; i < 9; i++) {
        utility::LogInfo("{:.6f} ", extr_depth2color.rotation[i]);
    }
    utility::LogInfo("depth->color extrinsics: Translation");
    for (int i = 0; i < 3; i++) {
        utility::LogInfo("{:.6f} ", extr_depth2color.translation[i]);
    }
    utility::LogInfo("");

    for (const auto &rs_stream : {rs_depth, rs_color}) {
        rs2_intrinsics intr = rs_stream.get_intrinsics();
        if (rs_stream.unique_id() == rs_depth.unique_id())
            depth_size = {intr.width, intr.height};
        else if (rs_stream.unique_id() == rs_color.unique_id())
            color_size = {intr.width, intr.height};
        utility::LogInfo("Instrinsics for stream {}", rs_stream.stream_name());
        utility::LogInfo("{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}", intr.width,
                         intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy);
        for (int i = 0; i < 5; i++) {
            utility::LogInfo("{:.6f} ", intr.coeffs[i]);
        }
        utility::LogInfo("");
    }

    // Allocate memory for frames
    auto depth_image_ptr = std::make_shared<geometry::Image>();
    if (align_streams)
        depth_image_ptr->Prepare(color_size.width, color_size.height, 1, 2);
    else
        depth_image_ptr->Prepare(depth_size.width, depth_size.height, 1, 2);
    auto color_image_ptr = std::make_shared<geometry::Image>();
    color_image_ptr->Prepare(color_size.width, color_size.height, 3, 1);

    // Create windows to show depth and color streams
    visualization::Visualizer depth_vis, color_vis;
    if (!depth_vis.CreateVisualizerWindow("Open3D || RealSense || Depth",
                                          depth_size.width, depth_size.height,
                                          15, 50) ||
        !depth_vis.AddGeometry(depth_image_ptr) ||
        !color_vis.CreateVisualizerWindow("Open3D || RealSense || Color",
                                          color_size.width, color_size.height,
                                          675, 50) ||
        !color_vis.AddGeometry(color_image_ptr)) {
        return 0;
    }

    // Create filter to align the depth image to the color image
    rs2::align align_to_color = rs2::align(RS2_STREAM_COLOR);
    // Declare rates printer for showing streaming rates of the enabled
    // streams.
    rs2::rates_printer printer;

    // Loop over frames from device
    while (depth_vis.PollEvents() && color_vis.PollEvents()) {
        rs2::frameset frames =
                pipe.wait_for_frames()  // Wait for next set of frames from
                                        // the camera
                        .apply_filter(printer);  // Print each enabled
                                                 // stream frame rate

        if (align_streams)  // Align depth to color
            frames = frames.apply_filter(align_to_color);
        const auto &depth_frame = frames.get_depth_frame();
        memcpy(depth_image_ptr->data_.data(), depth_frame.get_data(),
               depth_image_ptr->data_.size());
        const auto &color_frame = frames.get_color_frame();
        memcpy(color_image_ptr->data_.data(), color_frame.get_data(),
               color_image_ptr->data_.size());
        depth_vis.UpdateGeometry();
        color_vis.UpdateGeometry();
    }

    return 0;
}
