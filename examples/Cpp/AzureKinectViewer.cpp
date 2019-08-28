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
    utility::LogInfo("--config  Config .json file (default: none)\n");
    utility::LogInfo("--list    List the currently connected K4A devices\n");
    utility::LogInfo("--device  Specify the device index to use (default: 0)\n");
    utility::LogInfo("-a        Align depth with color image (default: disabled)\n");
    utility::LogInfo("-h        Print this helper\n");
    // clang-format on
}

int main(int argc, char **argv) {
    // Parse arguments
    if (utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintUsage();
        return 0;
    }

    if (utility::ProgramOptionExists(argc, argv, "--list")) {
        io::AzureKinectSensor::ListDevices();
        return 0;
    }

    io::AzureKinectSensorConfig sensor_config;
    if (utility::ProgramOptionExists(argc, argv, "--config")) {
        auto config_filename =
                utility::GetProgramOptionAsString(argc, argv, "--config", "");
        if (!io::ReadIJsonConvertibleFromJSON(config_filename, sensor_config)) {
            utility::LogInfo("Invalid sensor config\n");
            return 1;
        }
    } else {
        utility::LogInfo("Use default sensor config\n");
    }

    int sensor_index =
            utility::GetProgramOptionAsInt(argc, argv, "--device", 0);
    if (sensor_index < 0 || sensor_index > 255) {
        utility::LogError("Sensor index must between [0, 255]: {}\n",
                          sensor_index);
        return 1;
    }

    bool enable_align_depth_to_color =
            utility::ProgramOptionExists(argc, argv, "-a");

    // Init sensor
    io::AzureKinectSensor sensor(sensor_config);
    if (!sensor.Connect(sensor_index)) {
        utility::LogError("Failed to connect to sensor, abort.\n");
        return 1;
    }

    // Start viewing
    bool flag_exit = false;
    bool is_geometry_added = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.RegisterKeyCallback(GLFW_KEY_ESCAPE,
                            [&](visualization::Visualizer *vis) {
                                flag_exit = true;
                                return false;
                            });

    vis.CreateVisualizerWindow("Open3D Azure Kinect Recorder", 1920, 540);
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

    } while (!flag_exit);

    return 0;
}
