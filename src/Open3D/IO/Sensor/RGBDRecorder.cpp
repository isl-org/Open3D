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

#include "Open3D/IO/Sensor/RGBDRecorder.h"

#include <assert.h>
#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <iostream>

#include <k4a/k4a.h>
#include <k4arecord/record.h>

#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/IO/Sensor/MKVReader.h"
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

inline static void HstackRGBDepth(
        const std::shared_ptr<geometry::RGBDImage>& im_rgbd,
        geometry::Image& im_rgb_depth_hstack) {
    int width = im_rgbd->color_.width_;
    int height = im_rgbd->color_.height_;
    if (width != im_rgbd->depth_.width_ || height != im_rgbd->depth_.height_) {
        throw std::runtime_error("Color and depth image size mismatch");
    }
    if (im_rgbd->color_.num_of_channels_ != 3) {
        throw std::runtime_error("Color image does not have 3 channels");
    }
    if (im_rgbd->color_.bytes_per_channel_ != 1) {
        throw std::runtime_error("Color image is not uint_8");
    }
    if (im_rgbd->depth_.num_of_channels_ != 1) {
        throw std::runtime_error("Depth image does not have 1 channel");
    }
    if (im_rgbd->depth_.bytes_per_channel_ != 4) {
        throw std::runtime_error("Depth image is not float");
    }
    int double_width = width * 2;

    geometry::Image im_full_res;
    im_full_res.Prepare(double_width, height, 3, 1);

    // float depth_min = *im_rgbd->depth_.PointerAt<float>(0, 0);
    // float depth_max = depth_min;
    // for (int u = 0; u < width; ++u) {
    //     for (int v = 0; v < height; ++v) {
    //         float depth = *im_rgbd->depth_.PointerAt<float>(u, v);
    //         depth_min = std::min(depth, depth_min);
    //         depth_max = std::max(depth, depth_max);
    //     }
    // }
    float depth_min = 0;
    float depth_max = 3;

    visualization::ColorMapJet color_map;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int u = 0; u < width; ++u) {
        for (int v = 0; v < height; ++v) {
            // Color image
            for (int c = 0; c < 3; ++c) {
                *im_full_res.PointerAt<uint8_t>(u, v, c) =
                        *im_rgbd->color_.PointerAt<uint8_t>(u, v, c);
            }
            // Depth image
            float depth = *im_rgbd->depth_.PointerAt<float>(u, v);
            float ratio = (depth - depth_min) / (depth_max - depth_min);
            Eigen::Vector3d color = color_map.GetColor(ratio);
            *im_full_res.PointerAt<uint8_t>(u + width, v, 0) =
                    static_cast<uint8_t>(color(0) * 255);
            *im_full_res.PointerAt<uint8_t>(u + width, v, 1) =
                    static_cast<uint8_t>(color(1) * 255);
            *im_full_res.PointerAt<uint8_t>(u + width, v, 2) =
                    static_cast<uint8_t>(color(2) * 255);
        }
    }

    // Downsample for now
    int half_width = (int)floor((double)im_full_res.width_ / 2.0);
    int half_height = (int)floor((double)im_full_res.height_ / 2.0);
    im_rgb_depth_hstack.Prepare(half_width, half_height, 3, 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int u = 0; u < half_width; ++u) {
        for (int v = 0; v < half_height; ++v) {
            for (int c = 0; c < 3; ++c) {
                uint8_t* p1 = im_full_res.PointerAt<uint8_t>(u * 2, v * 2, c);
                uint8_t* p2 =
                        im_full_res.PointerAt<uint8_t>(u * 2 + 1, v * 2, c);
                uint8_t* p3 =
                        im_full_res.PointerAt<uint8_t>(u * 2, v * 2 + 1, c);
                uint8_t* p4 =
                        im_full_res.PointerAt<uint8_t>(u * 2 + 1, v * 2 + 1, c);
                uint8_t* p = im_rgb_depth_hstack.PointerAt<uint8_t>(u, v, c);
                float avg =
                        (static_cast<float>(*p1) + static_cast<float>(*p2) +
                         static_cast<float>(*p3) + static_cast<float>(*p4)) /
                        4.f;
                avg = std::max(0.f, avg);
                avg = std::min(255.f, avg);
                *p = static_cast<uint8_t>(avg);
            }
        }
    }
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

std::atomic_bool exiting(false);

int do_recording(uint8_t device_index,
                 char* recording_filename,
                 int recording_length,
                 k4a_device_configuration_t* device_config,
                 bool record_imu,
                 int32_t absoluteExposureValue) {
    const uint32_t installed_devices = k4a_device_get_installed_count();
    if (device_index >= installed_devices) {
        std::cerr << "Device not found." << std::endl;
        return 1;
    }

    k4a_device_t device;
    if (K4A_FAILED(k4a_device_open(device_index, &device))) {
        std::cerr << "Runtime error: k4a_device_open() failed " << std::endl;
    }

    char serial_number_buffer[256];
    size_t serial_number_buffer_size = sizeof(serial_number_buffer);
    CHECK(k4a_device_get_serialnum(device, serial_number_buffer,
                                   &serial_number_buffer_size),
          device);

    std::cout << "Device serial number: " << serial_number_buffer << std::endl;

    k4a_hardware_version_t version_info;
    CHECK(k4a_device_get_version(device, &version_info), device);

    std::cout << "Device version: "
              << (version_info.firmware_build == K4A_FIRMWARE_BUILD_RELEASE
                          ? "Rel"
                          : "Dbg")
              << "; C: " << version_info.rgb.major << "."
              << version_info.rgb.minor << "." << version_info.rgb.iteration
              << "; D: " << version_info.depth.major << "."
              << version_info.depth.minor << "." << version_info.depth.iteration
              << "[" << version_info.depth_sensor.major << "."
              << version_info.depth_sensor.minor << "]"
              << "; A: " << version_info.audio.major << "."
              << version_info.audio.minor << "." << version_info.audio.iteration
              << std::endl;

    uint32_t camera_fps = k4a_convert_fps_to_uint(device_config->camera_fps);

    if (camera_fps <= 0 ||
        (device_config->color_resolution == K4A_COLOR_RESOLUTION_OFF &&
         device_config->depth_mode == K4A_DEPTH_MODE_OFF)) {
        std::cerr
                << "Either the color or depth modes must be enabled to record."
                << std::endl;
        return 1;
    }

    if (absoluteExposureValue != 0) {
        if (K4A_FAILED(k4a_device_set_color_control(
                    device, K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                    K4A_COLOR_CONTROL_MODE_MANUAL, absoluteExposureValue))) {
            std::cerr << "Runtime error: k4a_device_set_color_control() failed "
                      << std::endl;
        }
    } else {
        if (K4A_FAILED(k4a_device_set_color_control(
                    device, K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                    K4A_COLOR_CONTROL_MODE_AUTO, 0))) {
            std::cerr << "Runtime error: k4a_device_set_color_control() failed "
                      << std::endl;
        }
    }

    CHECK(k4a_device_start_cameras(device, device_config), device);
    if (record_imu) {
        CHECK(k4a_device_start_imu(device), device);
    }

    std::cout << "Device started" << std::endl;

    k4a_record_t recording;
    if (K4A_FAILED(k4a_record_create(recording_filename, device, *device_config,
                                     &recording))) {
        std::cerr << "Unable to create recording file: " << recording_filename
                  << std::endl;
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

    // Init visualizer
    bool record_on = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.CreateVisualizerWindow("Open3D Azure Kinect Recorder", 1920, 640);
    vis.RegisterKeyCallback(GLFW_KEY_SPACE, [&](visualization::Visualizer*
                                                        vis) {
        if (record_on) {
            std::cout << "Recording paused, press space to start recording."
                      << std::endl;
        } else {
            std::cout << "Recording started, press space to pause recording."
                      << std::endl;
        }
        record_on = !record_on;
        return false;
    });

    // Wait for the first capture before starting recording.
    k4a_capture_t capture;
    int32_t timeout_sec_for_first_capture = 60;
    if (device_config->wired_sync_mode == K4A_WIRED_SYNC_MODE_SUBORDINATE) {
        timeout_sec_for_first_capture = 360;
        std::cout << "[subordinate mode] Waiting for signal from master"
                  << std::endl;
    }
    clock_t first_capture_start = clock();
    k4a_wait_result_t result = K4A_WAIT_RESULT_TIMEOUT;
    // Wait for the first capture in a loop so Ctrl-C will still exit.
    while (!exiting &&
           (clock() - first_capture_start) <
                   (CLOCKS_PER_SEC * timeout_sec_for_first_capture)) {
        result = k4a_device_get_capture(device, &capture, 100);
        if (result == K4A_WAIT_RESULT_SUCCEEDED) {
            k4a_capture_release(capture);
            break;
        } else if (result == K4A_WAIT_RESULT_FAILED) {
            std::cerr << "Runtime error: k4a_device_get_capture() returned "
                         "error: "
                      << result << std::endl;
            return 1;
        }
    }

    if (exiting) {
        k4a_device_close(device);
        return 0;
    } else if (result == K4A_WAIT_RESULT_TIMEOUT) {
        std::cerr << "Timed out waiting for first capture." << std::endl;
        return 1;
    }

    // std::cout << "Started recording" << std::endl;
    if (recording_length <= 0) {
        std::cout << "Press Ctrl-C to stop recording." << std::endl;
    }

    std::cout << "Recording paused, press space to start recording."
              << std::endl;

    clock_t recording_start = clock();
    int32_t timeout_ms = 1000 / camera_fps;

    std::shared_ptr<geometry::Image> im_rgb_depth_hstack = nullptr;

    do {
        result = k4a_device_get_capture(device, &capture, timeout_ms);
        if (result == K4A_WAIT_RESULT_TIMEOUT) {
            continue;
        } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
            std::cerr << "Runtime error: k4a_device_get_capture() returned "
                      << result << std::endl;
            break;
        }

        std::shared_ptr<geometry::RGBDImage> im_rgbd =
                io::MKVReader::DecompressCapture(capture, transformation);

        if (im_rgb_depth_hstack == nullptr) {
            im_rgb_depth_hstack = std::make_shared<geometry::Image>();
            HstackRGBDepth(im_rgbd, *im_rgb_depth_hstack);
            vis.AddGeometry(im_rgb_depth_hstack);
        } else {
            HstackRGBDepth(im_rgbd, *im_rgb_depth_hstack);
        }

        // im_depth = std::make_shared<geometry::Image>(im_rgbd->depth_);
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
                        std::cerr << "Runtime error: k4a_imu_get_sample() "
                                     "returned "
                                  << result << std::endl;
                        break;
                    }
                    k4a_result_t write_result =
                            k4a_record_write_imu_sample(recording, sample);
                    if (K4A_FAILED(write_result)) {
                        std::cerr << "Runtime error: "
                                     "k4a_record_write_imu_sample() "
                                     "returned "
                                  << write_result << std::endl;
                        break;
                    }
                } while (!exiting && result != K4A_WAIT_RESULT_FAILED &&
                         (recording_length < 0 ||
                          (clock() - recording_start <
                           recording_length * CLOCKS_PER_SEC)));
            }
        }
        k4a_capture_release(capture);

    } while (!exiting && result != K4A_WAIT_RESULT_FAILED &&
             (recording_length < 0 ||
              (clock() - recording_start < recording_length * CLOCKS_PER_SEC)));

    if (!exiting) {
        exiting = true;
        std::cout << "Stopping recording..." << std::endl;
    }

    if (record_imu) {
        k4a_device_stop_imu(device);
    }
    k4a_device_stop_cameras(device);

    std::cout << "Saving recording..." << std::endl;
    CHECK(k4a_record_flush(recording), device);
    k4a_record_close(recording);

    std::cout << "Done" << std::endl;

    k4a_device_close(device);

    return 0;
}
}  // namespace io
}  // namespace open3d
