// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "open3d/io/sensor/azure_kinect/AzureKinectRecorder.h"

#include <assert.h>
#include <k4a/k4a.h>
#include <k4arecord/record.h>

#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <iostream>

#include "open3d/geometry/RGBDImage.h"
#include "open3d/io/sensor/azure_kinect/K4aPlugin.h"
#include "open3d/io/sensor/azure_kinect/MKVReader.h"
#include "open3d/visualization/utility/ColorMap.h"
#include "open3d/visualization/visualizer/VisualizerWithKeyCallback.h"

namespace open3d {
namespace io {

AzureKinectRecorder::AzureKinectRecorder(
        const AzureKinectSensorConfig& sensor_config, size_t sensor_index)
    : RGBDRecorder(),
      sensor_(AzureKinectSensor(sensor_config)),
      device_index_(sensor_index) {}

AzureKinectRecorder::~AzureKinectRecorder() { CloseRecord(); }

bool AzureKinectRecorder::InitSensor() {
    return sensor_.Connect(device_index_);
}

bool AzureKinectRecorder::OpenRecord(const std::string& filename) {
    if (!is_record_created_) {
        if (K4A_FAILED(k4a_plugin::k4a_record_create(
                    filename.c_str(), sensor_.device_,
                    sensor_.sensor_config_.ConvertToNativeConfig(),
                    &recording_))) {
            utility::LogWarning("Unable to create recording file: {}",
                                filename);
            return false;
        }
        if (K4A_FAILED(k4a_plugin::k4a_record_write_header(recording_))) {
            utility::LogWarning("Unable to write header");
            return false;
        }
        utility::LogInfo("Writing to header");

        is_record_created_ = true;
    }
    return true;
}

bool AzureKinectRecorder::CloseRecord() {
    if (is_record_created_) {
        utility::LogInfo("Saving recording...");
        if (K4A_FAILED(k4a_plugin::k4a_record_flush(recording_))) {
            utility::LogWarning("Unable to flush record file");
            return false;
        }
        k4a_plugin::k4a_record_close(recording_);
        utility::LogInfo("Done");

        is_record_created_ = false;
    }
    return true;
}

std::shared_ptr<geometry::RGBDImage> AzureKinectRecorder::RecordFrame(
        bool write, bool enable_align_depth_to_color) {
    k4a_capture_t capture = sensor_.CaptureRawFrame();
    if (!capture) return nullptr;

    if (capture != nullptr && is_record_created_ && write) {
        if (K4A_FAILED(k4a_plugin::k4a_record_write_capture(recording_,
                                                            capture))) {
            utility::LogError("Unable to write to capture");
        }
    }

    auto im_rgbd = AzureKinectSensor::DecompressCapture(
            capture, enable_align_depth_to_color
                             ? sensor_.transform_depth_to_color_
                             : nullptr);
    if (im_rgbd == nullptr) {
        utility::LogDebug("Invalid capture, skipping this frame");
        return nullptr;
    }
    k4a_plugin::k4a_capture_release(capture);
    return im_rgbd;
}
}  // namespace io
}  // namespace open3d
