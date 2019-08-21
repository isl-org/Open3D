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
#include "Open3D/IO/Sensor/AzureKinect/K4aPlugin.h"
#include "Open3D/IO/Sensor/AzureKinect/MKVReader.h"
#include "Open3D/Visualization/Utility/ColorMap.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithKeyCallback.h"

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
            utility::LogError("Unable to create recording file: {}\n",
                              filename);
            return false;
        }
        if (K4A_FAILED(k4a_plugin::k4a_record_write_header(recording_))) {
            utility::LogError("Unable to write header\n");
            return false;
        }
        utility::LogInfo("Writing to header\n");

        is_record_created_ = true;
    }
    return true;
}

bool AzureKinectRecorder::CloseRecord() {
    if (is_record_created_) {
        utility::LogInfo("Saving recording...\n");
        if (K4A_FAILED(k4a_plugin::k4a_record_flush(recording_))) {
            utility::LogError("Unable to flush record file\n");
            return false;
        }
        k4a_plugin::k4a_record_close(recording_);
        utility::LogInfo("Done\n");

        is_record_created_ = false;
    }
    return true;
}

std::shared_ptr<geometry::RGBDImage> AzureKinectRecorder::RecordFrame(
        bool write, bool enable_align_depth_to_color) {
    k4a_capture_t capture = sensor_.CaptureRawFrame();
    if (capture != nullptr && is_record_created_ && write) {
        if (K4A_FAILED(k4a_plugin::k4a_record_write_capture(recording_,
                                                            capture))) {
            utility::LogError("Unable to write to capture\n");
            return nullptr;
        }
    }

    auto im_rgbd = AzureKinectSensor::DecompressCapture(
            capture, enable_align_depth_to_color
                             ? sensor_.transform_depth_to_color_
                             : nullptr);
    if (im_rgbd == nullptr) {
        utility::LogInfo("Invalid capture, skipping this frame\n");
        return nullptr;
    }
    k4a_plugin::k4a_capture_release(capture);
    return im_rgbd;
}
}  // namespace io
}  // namespace open3d
