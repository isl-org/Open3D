// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <k4a/k4a.h>

#include "assert.h"

#include <math.h>
#include <atomic>
#include <csignal>
#include <ctime>
#include <iostream>

#include "Open3D/Open3D.h"

using namespace open3d;

void PrintUsage() {
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Options: \n");
    utility::LogInfo("-c        Set the color sensor mode (default: 720p)\n");
    utility::LogInfo("          Available options:\n");
    utility::LogInfo("          3072p, 2160p, 1536p, 1440p, 1080p, 720p, OFF\n");
    utility::LogInfo("-d        Set the depth sensor mode (default: WFOV_2X2BINNED)\n");
    utility::LogInfo("          Available options:\n");
    utility::LogInfo("          NFOV_2X2BINNED, NFOV_UNBINNED, WFOV_2X2BINNED, WFOV_UNBINNED, OFF\n");
    utility::LogInfo("-r        Set the camera frame rate in Frames per Second (default: 30)\n");
    utility::LogInfo("          Available options: 30, 15, 5\n"),
    utility::LogInfo("-a        Align depth with color image (default: disabled)\n");
    utility::LogInfo("--list    List the currently connected K4A devices\n");
    utility::LogInfo("--device  Specify the device index to use (default: 0)\n");
    // clang-format on
}

int ParseArgs(int argc,
              char **argv,
              io::AzureKinectSensorConfig &sensor_config,
              int &sensor_index,
              bool &enable_align_depth_to_color,
              std::string &recording_filename) {
    k4a_image_format_t recording_color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    k4a_color_resolution_t recording_color_resolution =
            K4A_COLOR_RESOLUTION_720P;
    k4a_depth_mode_t recording_depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    k4a_fps_t recording_rate = K4A_FRAMES_PER_SECOND_30;
    enable_align_depth_to_color = false;

    if (utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintUsage();
        exit(0);
    }

    if (utility::ProgramOptionExists(argc, argv, "--list")) {
        io::AzureKinectSensor::ListDevices();
        exit(0);
    }

    sensor_index = utility::GetProgramOptionAsInt(argc, argv, "--device", 0);
    if (sensor_index < 0 || sensor_index > 255) {
        std::ostringstream str;
        str << "Sensor index must between [0, 255]: " << sensor_index;
        throw std::runtime_error(str.str());
    }
    enable_align_depth_to_color =
            utility::ProgramOptionExists(argc, argv, "-a");

    auto color_mode =
            utility::GetProgramOptionAsString(argc, argv, "-c", "720p");
    if (color_mode == "3072p") {
        recording_color_resolution = K4A_COLOR_RESOLUTION_3072P;
    } else if (color_mode == "2160p") {
        recording_color_resolution = K4A_COLOR_RESOLUTION_2160P;
    } else if (color_mode == "1536p") {
        recording_color_resolution = K4A_COLOR_RESOLUTION_1536P;
    } else if (color_mode == "1440p") {
        recording_color_resolution = K4A_COLOR_RESOLUTION_1440P;
    } else if (color_mode == "1080p") {
        recording_color_resolution = K4A_COLOR_RESOLUTION_1080P;
    } else if (color_mode == "720p") {
        recording_color_resolution = K4A_COLOR_RESOLUTION_720P;
    } else {
        recording_color_resolution = K4A_COLOR_RESOLUTION_OFF;
        std::ostringstream str;
        str << "Unsupported color mode specified: " << color_mode;
        throw std::runtime_error(str.str());
    }

    auto depth_mode = utility::GetProgramOptionAsString(argc, argv, "-d",
                                                        "WFOV_2X2BINNED");
    if (depth_mode == "NFOV_2X2BINNED") {
        recording_depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    } else if (depth_mode == "NFOV_UNBINNED") {
        recording_depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    } else if (depth_mode == "WFOV_2X2BINNED") {
        recording_depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    } else if (depth_mode == "WFOV_UNBINNED") {
        recording_depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
    } else {
        recording_depth_mode = K4A_DEPTH_MODE_OFF;
        std::ostringstream str;
        str << "Unsupported depth mode specified: " << depth_mode;
        throw std::runtime_error(str.str());
    }

    auto fps = utility::GetProgramOptionAsInt(argc, argv, "-r", 30);
    if (fps == 30) {
        recording_rate = K4A_FRAMES_PER_SECOND_30;
    } else if (fps == 15) {
        recording_rate = K4A_FRAMES_PER_SECOND_15;
    } else if (fps == 5) {
        recording_rate = K4A_FRAMES_PER_SECOND_5;
    } else {
        std::ostringstream str;
        str << "Unsupported frame rate specified: " << fps;
        throw std::runtime_error(str.str());
    }

    if (recording_rate == K4A_FRAMES_PER_SECOND_30 &&
        (recording_depth_mode == K4A_DEPTH_MODE_WFOV_UNBINNED ||
         recording_color_resolution == K4A_COLOR_RESOLUTION_3072P)) {
        utility::LogWarning(
                "Warning: 30 Frames per second is not supported by this "
                "camera mode, fallback to 15 Frames per second.\n");
        recording_rate = K4A_FRAMES_PER_SECOND_15;
    }

    k4a_device_configuration_t device_config =
            K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    device_config.color_format = recording_color_format;
    device_config.color_resolution = recording_color_resolution;
    device_config.depth_mode = recording_depth_mode;
    device_config.camera_fps = recording_rate;
    sensor_config.ConvertFromNativeConfig(device_config);

    return 0;
}

int main(int argc, char **argv) {
    int sensor_index = 0;
    std::string recording_filename;
    io::AzureKinectSensorConfig sensor_config;
    bool enable_align_depth_to_color;
    if (ParseArgs(argc, argv, sensor_config, sensor_index,
                  enable_align_depth_to_color, recording_filename) != 0) {
        utility::LogError("Parse args error\n");
    }

    io::AzureKinectSensor sensor(sensor_config);
    sensor.Connect(sensor_index);

    bool loop_finished = false;
    bool is_geometry_added = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.CreateVisualizerWindow("Open3D Azure Kinect Recorder", 1920, 540);
    vis.GetRenderOption().image_stretch_option_ =
            visualization::RenderOption::ImageStretchOption::StretchKeepRatio;

    // Finish callback
    vis.RegisterKeyCallback(GLFW_KEY_ESCAPE,
                            [&](visualization::Visualizer *vis) {
                                loop_finished = true;
                                return false;
                            });

    do {
        auto im_rgbd = sensor.CaptureFrame(enable_align_depth_to_color);
        if (im_rgbd == nullptr) {
            utility::LogInfo("Invalid capture, skipping this frame\n");
            continue;
        }

        if (!is_geometry_added) {
            vis.AddGeometry(im_rgbd);
            is_geometry_added = true;
        }

        // Update visualizer
        vis.UpdateGeometry();
        vis.PollEvents();
        vis.UpdateRender();

    } while (!loop_finished);
}
