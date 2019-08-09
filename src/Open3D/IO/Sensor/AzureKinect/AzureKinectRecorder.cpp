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

// call k4a_device_close on every failed CHECK
#define CHECK(x, device)                                                   \
    {                                                                      \
        auto retval = (x);                                                 \
        if (retval) {                                                      \
            std::cerr << "Runtime error: " << #x << " returned " << retval \
                      << std::endl;                                        \
            k4a_device_close(device);                                      \
            return 1;                                                      \
        }                                                                  \
    }

int Record(uint8_t device_index,
           char* recording_filename,
           k4a_device_configuration_t* device_config,
           bool record_imu,
           int32_t absoluteExposureValue) {
    const uint32_t installed_devices = k4a_device_get_installed_count();
    if (device_index >= installed_devices) {
        utility::LogError("Device not found.\n");
        return 1;
    }

    k4a_device_t device;
    if (K4A_FAILED(k4a_device_open(device_index, &device))) {
        utility::LogError("Runtime error: k4a_device_open() failed\n");
        return 1;
    }

    char serial_number_buffer[256];
    size_t serial_number_buffer_size = sizeof(serial_number_buffer);
    CHECK(k4a_device_get_serialnum(device, serial_number_buffer,
                                   &serial_number_buffer_size),
          device);
    k4a_hardware_version_t version_info;
    CHECK(k4a_device_get_version(device, &version_info), device);

    utility::LogInfo("Device serial number: {}\n", serial_number_buffer);
    utility::LogInfo("Device version: {};\n",
                     (version_info.firmware_build == K4A_FIRMWARE_BUILD_RELEASE
                              ? "Rel"
                              : "Dbg"));
    utility::LogInfo("C: {}.{}.{};\n", version_info.rgb.major,
                     version_info.rgb.minor, version_info.rgb.iteration);
    utility::LogInfo("D: {}.{}.{}[{}.{}]\n", version_info.depth.major,
                     version_info.depth.minor, version_info.depth.iteration,
                     version_info.depth_sensor.major,
                     version_info.depth_sensor.minor);
    utility::LogInfo("A: {}.{}.{};\n", version_info.audio.major,
                     version_info.audio.minor, version_info.audio.iteration);

    uint32_t camera_fps = k4a_convert_fps_to_uint(device_config->camera_fps);

    if (camera_fps <= 0 ||
        (device_config->color_resolution == K4A_COLOR_RESOLUTION_OFF &&
         device_config->depth_mode == K4A_DEPTH_MODE_OFF)) {
        utility::LogError(
                "Either the color or depth modes must be enabled to record.\n");
        return 1;
    }

    if (absoluteExposureValue != 0) {
        if (K4A_FAILED(k4a_device_set_color_control(
                    device, K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                    K4A_COLOR_CONTROL_MODE_MANUAL, absoluteExposureValue))) {
            utility::LogError(
                    "Runtime error: k4a_device_set_color_control() failed\n");
        }
    } else {
        if (K4A_FAILED(k4a_device_set_color_control(
                    device, K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                    K4A_COLOR_CONTROL_MODE_AUTO, 0))) {
            utility::LogError(
                    "Runtime error: k4a_device_set_color_control() failed\n");
        }
    }

    CHECK(k4a_device_start_cameras(device, device_config), device);
    if (record_imu) {
        CHECK(k4a_device_start_imu(device), device);
    }

    utility::LogInfo("Device started\n");

    k4a_record_t recording;
    if (K4A_FAILED(k4a_record_create(recording_filename, device, *device_config,
                                     &recording))) {
        utility::LogError("Unable to create recording file: %s\n",
                          recording_filename);
        return 1;
    }

    if (record_imu) {
        CHECK(k4a_record_add_imu_track(recording), device);
    }
    CHECK(k4a_record_write_header(recording), device);

    // Get transformation
    k4a_calibration_t calibration;
    k4a_device_get_calibration(device, device_config->depth_mode,
                               device_config->color_resolution, &calibration);
    k4a_transformation_t transformation =
            k4a_transformation_create(&calibration);

    bool record_on = false;
    bool record_finished = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.CreateVisualizerWindow("Open3D Azure Kinect Recorder", 1920, 640);
    vis.RegisterKeyCallback(
            GLFW_KEY_SPACE, [&](visualization::Visualizer* vis) {
                if (record_on) {
                    utility::LogInfo(
                            "Recording paused, press [SPACE] to continue; "
                            "press [ESC] to save and exit.\n");
                } else {
                    utility::LogInfo(
                            "Recording, press [SPACE] to pause recording.\n");
                }
                record_on = !record_on;
                return false;
            });
    vis.RegisterKeyCallback(GLFW_KEY_ESCAPE,
                            [&](visualization::Visualizer* vis) {
                                record_finished = true;
                                utility::LogInfo("Recording finished.\n");
                                return false;
                            });

    // Wait for the first capture before starting recording.
    k4a_capture_t capture;
    int32_t timeout_sec_for_first_capture = 60;
    if (device_config->wired_sync_mode == K4A_WIRED_SYNC_MODE_SUBORDINATE) {
        timeout_sec_for_first_capture = 360;
        utility::LogInfo("[subordinate mode] Waiting for signal from master\n");
    }

    // Wait for the first capture in a loop so Ctrl-C will still exit.
    clock_t first_capture_start = clock();
    k4a_wait_result_t result = K4A_WAIT_RESULT_TIMEOUT;
    while (!record_finished &&
           (clock() - first_capture_start) <
                   (CLOCKS_PER_SEC * timeout_sec_for_first_capture)) {
        result = k4a_device_get_capture(device, &capture, 100);
        if (result == K4A_WAIT_RESULT_SUCCEEDED) {
            k4a_capture_release(capture);
            break;
        } else if (result == K4A_WAIT_RESULT_FAILED) {
            utility::LogError(
                    "Runtime error: k4a_device_get_capture() returned "
                    "error: %d\n",
                    result);
            return 1;
        }
    }

    if (record_finished) {
        k4a_device_close(device);
        return 0;
    } else if (result == K4A_WAIT_RESULT_TIMEOUT) {
        utility::LogInfo("Timed out waiting for first capture.\n");
        return 1;
    }

    utility::LogInfo(
            "In the visulizer window, press [SPACE] to start recording, press "
            "[ESC] to exit\n");

    int32_t timeout_ms = 1000 / camera_fps;

    std::shared_ptr<geometry::Image> im_rgb_depth_hstack = nullptr;

    std::shared_ptr<geometry::RGBDImage> vis_rgbd_ = nullptr;
    do {
        result = k4a_device_get_capture(device, &capture, timeout_ms);
        if (result == K4A_WAIT_RESULT_TIMEOUT) {
            continue;
        } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
            utility::LogError(
                    "Runtime error: k4a_device_get_capture() returned %d\n",
                    result);
            break;
        }

        std::shared_ptr<geometry::RGBDImage> im_rgbd =
                io::MKVReader::DecompressCapture(capture, transformation);
        if (im_rgbd == nullptr) {
            utility::LogInfo("invalid capture, skipping this frame\n");
            continue;
        }

        if (vis_rgbd_ == nullptr) {
            vis_rgbd_ = im_rgbd;
            vis.AddGeometry(vis_rgbd_);
        } else {
            *vis_rgbd_ = *im_rgbd;
        }

        vis.UpdateGeometry();
        vis.PollEvents();
        vis.UpdateRender();

        if (record_on) {
            CHECK(k4a_record_write_capture(recording, capture), device);

            if (record_imu) {
                do {
                    k4a_imu_sample_t sample;
                    result = k4a_device_get_imu_sample(device, &sample, 0);
                    if (result == K4A_WAIT_RESULT_TIMEOUT) {
                        break;
                    } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
                        utility::LogError(
                                "Runtime error: k4a_imu_get_sample() "
                                "returned %d\n",
                                result);
                        break;
                    }
                    k4a_result_t write_result =
                            k4a_record_write_imu_sample(recording, sample);
                    if (K4A_FAILED(write_result)) {
                        utility::LogError(
                                "Runtime error: "
                                "k4a_record_write_imu_sample() "
                                "returned %d\n",
                                write_result);
                        break;
                    }
                } while (!record_finished && result != K4A_WAIT_RESULT_FAILED);
            }
        }
        k4a_capture_release(capture);

    } while (!record_finished && result != K4A_WAIT_RESULT_FAILED);

    if (record_imu) {
        k4a_device_stop_imu(device);
    }
    k4a_device_stop_cameras(device);

    utility::LogInfo("Saving recording...\n");
    CHECK(k4a_record_flush(recording), device);
    k4a_record_close(recording);
    utility::LogInfo("Done\n");

    k4a_device_close(device);

    return 0;
}
}  // namespace io
}  // namespace open3d
