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
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <k4a/k4a.h>
#include <math.h>

#include <atomic>
#include <csignal>
#include <ctime>
#include <iostream>

#include "open3d/Open3D.h"

using namespace open3d;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > AzureKinectRecord [options]");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --config                  : Config .json file (default: none)");
    utility::LogInfo("    --list                    : List the currently connected K4A devices");
    utility::LogInfo("    --device                  : Specify the device index to use (default: 0)");
    utility::LogInfo("    --output                  : Output mkv file name (default: current_time.mkv)");
    utility::LogInfo("    -a                        : Align depth with color image (default: disabled)");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char **argv) {
    if (argc < 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    if (utility::ProgramOptionExists(argc, argv, "--list")) {
        io::AzureKinectSensor::ListDevices();
        return 0;
    }

    io::AzureKinectSensorConfig sensor_config;
    if (utility::ProgramOptionExists(argc, argv, "--config")) {
        auto config_filename =
                utility::GetProgramOptionAsString(argc, argv, "--config", "");
        bool success = io::ReadIJsonConvertibleFromJSON(config_filename,
                                                        sensor_config);
        if (!success) {
            utility::LogInfo("Invalid sensor config");
            return 1;
        }
    } else {
        utility::LogInfo("Use default sensor config");
    }

    int sensor_index =
            utility::GetProgramOptionAsInt(argc, argv, "--device", 0);
    if (sensor_index < 0 || sensor_index > 255) {
        utility::LogWarning("Sensor index must between [0, 255]: {}",
                            sensor_index);
        return 1;
    }

    bool enable_align_depth_to_color =
            utility::ProgramOptionExists(argc, argv, "-a");

    std::string recording_filename = utility::GetProgramOptionAsString(
            argc, argv, "--output", utility::GetCurrentTimeStamp() + ".mkv");
    utility::LogInfo("Prepare writing to {}", recording_filename);

    // Init recorder
    io::AzureKinectRecorder recorder(sensor_config, sensor_index);
    if (!recorder.InitSensor()) {
        utility::LogWarning("Failed to connect to sensor, abort.");
        return 1;
    }

    bool flag_record = false;
    bool flag_exit = false;
    bool is_geometry_added = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.RegisterKeyCallback(
            GLFW_KEY_SPACE, [&](visualization::Visualizer *vis) {
                if (flag_record) {
                    utility::LogInfo(
                            "Recording paused. "
                            "Press [SPACE] to continue. "
                            "Press [ESC] to save and exit.");
                    flag_record = false;
                } else if (!recorder.IsRecordCreated()) {
                    if (recorder.OpenRecord(recording_filename)) {
                        utility::LogInfo(
                                "Recording started. "
                                "Press [SPACE] to pause. "
                                "Press [ESC] to save and exit.");
                        flag_record = true;
                    }  // else flag_record keeps false
                } else {
                    utility::LogInfo(
                            "Recording resumed, video may be discontinuous. "
                            "Press [SPACE] to pause. "
                            "Press [ESC] to save and exit.");
                    flag_record = true;
                }
                return false;
            });

    vis.RegisterKeyCallback(
            GLFW_KEY_ESCAPE, [&](visualization::Visualizer *vis) {
                flag_exit = true;
                if (recorder.IsRecordCreated()) {
                    utility::LogInfo("Recording finished.");
                } else {
                    utility::LogInfo("Nothing has been recorded.");
                }
                return false;
            });

    utility::LogInfo(
            "In the visualizer window, "
            "press [SPACE] to start recording, "
            "press [ESC] to exit.");

    vis.CreateVisualizerWindow("Open3D Azure Kinect Recorder", 1920, 540);
    do {
        auto im_rgbd =
                recorder.RecordFrame(flag_record, enable_align_depth_to_color);
        if (im_rgbd == nullptr) {
            utility::LogDebug("Invalid capture, skipping this frame");
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

    recorder.CloseRecord();

    return 0;
}
