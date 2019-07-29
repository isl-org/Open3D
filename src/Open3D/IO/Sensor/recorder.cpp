// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "recorder.h"
#include <algorithm>
#include <atomic>
#include <ctime>
#include <iostream>

#include <k4a/k4a.h>
#include <k4arecord/record.h>

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

std::atomic_bool exiting(false);

int do_recording(uint8_t device_index,
                 char *recording_filename,
                 int recording_length,
                 k4a_device_configuration_t *device_config,
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

    std::cout << "Started recording" << std::endl;
    if (recording_length <= 0) {
        std::cout << "Press Ctrl-C to stop recording." << std::endl;
    }

    clock_t recording_start = clock();
    int32_t timeout_ms = 1000 / camera_fps;
    do {
        result = k4a_device_get_capture(device, &capture, timeout_ms);
        if (result == K4A_WAIT_RESULT_TIMEOUT) {
            continue;
        } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
            std::cerr << "Runtime error: k4a_device_get_capture() returned "
                      << result << std::endl;
            break;
        }
        CHECK(k4a_record_write_capture(recording, capture), device);
        k4a_capture_release(capture);

        if (record_imu) {
            do {
                k4a_imu_sample_t sample;
                result = k4a_device_get_imu_sample(device, &sample, 0);
                if (result == K4A_WAIT_RESULT_TIMEOUT) {
                    break;
                } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
                    std::cerr << "Runtime error: k4a_imu_get_sample() returned "
                              << result << std::endl;
                    break;
                }
                k4a_result_t write_result =
                        k4a_record_write_imu_sample(recording, sample);
                if (K4A_FAILED(write_result)) {
                    std::cerr << "Runtime error: k4a_record_write_imu_sample() "
                                 "returned "
                              << write_result << std::endl;
                    break;
                }
            } while (!exiting && result != K4A_WAIT_RESULT_FAILED &&
                     (recording_length < 0 ||
                      (clock() - recording_start <
                       recording_length * CLOCKS_PER_SEC)));
        }
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
