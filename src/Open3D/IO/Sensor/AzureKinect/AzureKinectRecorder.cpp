// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectRecorder.h"

#include <assert.h>
#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <iostream>

#include <k4a/k4a.h>
#include <k4arecord/record.h>

#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/IO/Sensor/AzureKinect/MKVReader.h"
#include "Open3D/Visualization/Utility/ColorMap.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithKeyCallback.h"

namespace open3d {
namespace io {

inline static uint32_t k4a_convert_fps_to_uint(k4a_fps_t fps) {
    uint32_t fps_int;
    switch (fps) {
        case K4A_FRAMES_PER_SECOND_5:
            fps_int = 5;
            break;
        case K4A_FRAMES_PER_SECOND_15:
            fps_int = 15;
            break;
        case K4A_FRAMES_PER_SECOND_30:
            fps_int = 30;
            break;
        default:
            fps_int = 0;
            break;
    }
    return fps_int;
}

AzureKinectRecorder::AzureKinectRecorder(
        const AzureKinectSensorConfig& sensor_config, size_t sensor_index)
    : RGBDRecorder(),
      sensor_(AzureKinectSensor(sensor_config)),
      device_index_(sensor_index) {}

AzureKinectRecorder::~AzureKinectRecorder() {}

int AzureKinectRecorder::Record(const std::string& recording_filename) {
    // Convert to k4a native config
    k4a_device_configuration_t device_config =
            sensor_.sensor_config_.ConvertToNativeConfig();

    const uint32_t installed_devices = k4a_device_get_installed_count();
    if (device_index_ >= installed_devices) {
        utility::LogError("Device not found.\n");
        return 1;
    }
    sensor_.Connect(device_index_);

    uint32_t camera_fps = k4a_convert_fps_to_uint(device_config.camera_fps);

    if (camera_fps <= 0 ||
        (device_config.color_resolution == K4A_COLOR_RESOLUTION_OFF &&
         device_config.depth_mode == K4A_DEPTH_MODE_OFF)) {
        utility::LogError(
                "Runtime error: either the color or depth modes must be "
                "enabled to record.\n");
        return 1;
    }

    // Assume absoluteExposureValue == 0
    if (K4A_FAILED(k4a_device_set_color_control(
                sensor_.device_, K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                K4A_COLOR_CONTROL_MODE_AUTO, 0))) {
        utility::LogError(
                "Runtime error: k4a_device_set_color_control() failed\n");
    }

    utility::LogInfo("Device started\n");

    k4a_record_t recording;

    // Get transformation
    k4a_calibration_t calibration;
    k4a_device_get_calibration(sensor_.device_, device_config.depth_mode,
                               device_config.color_resolution, &calibration);
    k4a_transformation_t transformation =
            k4a_transformation_create(&calibration);

    // Flags to control the logic
    bool record_file_created = false;
    bool record_on = false;
    bool record_finished = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.CreateVisualizerWindow("Open3D Azure Kinect Recorder", 1920, 540);
    vis.GetRenderOption().image_stretch_option_ =
            visualization::RenderOption::ImageStretchOption::StretchKeepRatio;

    // Toggle pause callback
    vis.RegisterKeyCallback(
            GLFW_KEY_SPACE, [&](visualization::Visualizer* vis) {
                if (record_on) {
                    utility::LogInfo(
                            "Recording paused. "
                            "Press [SPACE] to continue. "
                            "Press [ESC] to save and exit.\n");
                } else if (!record_file_created) {
                    // Lazy mkv file creation
                    if (K4A_FAILED(k4a_record_create(
                                recording_filename.c_str(), sensor_.device_,
                                device_config, &recording))) {
                        utility::LogError(
                                "Unable to create recording file: {}\n",
                                recording_filename);
                        return false;
                    }
                    if (K4A_FAILED(k4a_record_write_header(recording))) {
                        utility::LogError("Unable to write header\n");
                        return false;
                    }

                    record_file_created = true;
                    utility::LogInfo(
                            "Recording started. "
                            "Press [SPACE] to pause. "
                            "Press [ESC] to save and exit.\n");
                } else {
                    utility::LogInfo(
                            "Recording resumed, video may be discontinuous. "
                            "Press [SPACE] to pause. "
                            "Press [ESC] to save and exit.\n");
                }

                record_on = !record_on;
                return false;
            });

    // Finish callback
    vis.RegisterKeyCallback(
            GLFW_KEY_ESCAPE, [&](visualization::Visualizer* vis) {
                record_finished = true;
                if (record_file_created) {
                    utility::LogInfo("Recording finished.\n");
                } else {
                    utility::LogInfo("Nothing has been recorded.\n");
                }
                return false;
            });

    k4a_capture_t capture;
    k4a_wait_result_t result;
    utility::LogInfo(
            "In the visulizer window, "
            "press [SPACE] to start recording, "
            "press [ESC] to exit.\n");

    int32_t timeout_ms = 1000 / camera_fps;
    std::shared_ptr<geometry::RGBDImage> vis_rgbd_ = nullptr;
    do {
        // Get capture
        result = k4a_device_get_capture(sensor_.device_, &capture, timeout_ms);
        if (result == K4A_WAIT_RESULT_TIMEOUT) {
            continue;
        } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
            utility::LogError(
                    "Runtime error: k4a_device_get_capture() returned %d\n",
                    result);
            break;
        }

        // Unwarp depth
        std::shared_ptr<geometry::RGBDImage> im_rgbd =
                io::MKVReader::DecompressCapture(capture, transformation);
        if (im_rgbd == nullptr) {
            utility::LogInfo("Invalid capture, skipping this frame\n");
            continue;
        }

        // Init visualizer
        if (vis_rgbd_ == nullptr) {
            vis_rgbd_ = im_rgbd;
            vis.AddGeometry(vis_rgbd_);
        } else {
            *vis_rgbd_ = *im_rgbd;
        }

        // Update visualizer
        vis.UpdateGeometry();
        vis.PollEvents();
        vis.UpdateRender();

        // Write to file
        if (record_on && record_file_created) {
            if (K4A_FAILED(k4a_record_write_capture(recording, capture))) {
                utility::LogError("Unable to write to capture\n");
                break;
            }
        }
        k4a_capture_release(capture);
    } while (!record_finished);

    if (record_file_created) {
        utility::LogInfo("Saving recording...\n");
        if (K4A_FAILED(k4a_record_flush(recording))) {
            utility::LogError("Unable to flush record file\n");
        }
        k4a_record_close(recording);
        utility::LogInfo("Done\n");
    }

    k4a_device_stop_cameras(sensor_.device_);
    k4a_device_close(sensor_.device_);

    return 0;
}  // namespace io
}  // namespace io
}  // namespace open3d
