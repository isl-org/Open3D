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

#include "Open3D/IO/Sensor/AzureKinect/MKVReader.h"

#include <json/json.h>
#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
#include <turbojpeg.h>
#include <iostream>

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensor.h"
#include "Open3D/IO/Sensor/AzureKinect/K4aPlugin.h"

namespace open3d {
namespace io {

MKVReader::MKVReader() : handle_(nullptr), transformation_(nullptr) {}

bool MKVReader::IsOpened() { return handle_ != nullptr; }

std::string MKVReader::GetTagInMetadata(const std::string &tag_name) {
    char res_buffer[256];
    size_t res_size = 256;

    k4a_buffer_result_t result = k4a_plugin::k4a_playback_get_tag(
            handle_, tag_name.c_str(), res_buffer, &res_size);
    if (K4A_BUFFER_RESULT_SUCCEEDED == result) {
        return res_buffer;
    } else if (K4A_BUFFER_RESULT_TOO_SMALL == result) {
        utility::LogError("{} tag's content is too long.\n", tag_name);
        return "";
    } else {
        utility::LogError("{} tag does not exist.\n", tag_name);
        return "";
    }
}

bool MKVReader::Open(const std::string &filename) {
    if (IsOpened()) {
        Close();
    }

    if (K4A_RESULT_SUCCEEDED !=
        k4a_plugin::k4a_playback_open(filename.c_str(), &handle_)) {
        utility::LogError("Unable to open file {}\n", filename);
        return false;
    }

    metadata_.ConvertFromJsonValue(GetMetadataJson());
    is_eof_ = false;

    return true;
}

void MKVReader::Close() { k4a_plugin::k4a_playback_close(handle_); }

Json::Value MKVReader::GetMetadataJson() {
    static const std::unordered_map<std::string, std::pair<int, int>>
            resolution_to_width_height({{"720P", std::make_pair(1280, 720)},
                                        {"1080P", std::make_pair(1920, 1080)},
                                        {"1440P", std::make_pair(2560, 1440)},
                                        {"1536P", std::make_pair(2048, 1536)},
                                        {"2160P", std::make_pair(3840, 2160)},
                                        {"3072P", std::make_pair(4096, 3072)}});

    if (!IsOpened()) {
        utility::LogError("Null file handler. Please call Open().\n");
        return Json::Value();
    }

    Json::Value value;

    k4a_calibration_t calibration;
    if (K4A_RESULT_SUCCEEDED !=
        k4a_plugin::k4a_playback_get_calibration(handle_, &calibration)) {
        utility::LogError("Failed to get calibration\n");
    }

    camera::PinholeCameraIntrinsic pinhole_camera;
    auto color_camera_calibration = calibration.color_camera_calibration;
    auto param = color_camera_calibration.intrinsics.parameters.param;
    pinhole_camera.SetIntrinsics(color_camera_calibration.resolution_width,
                                 color_camera_calibration.resolution_height,
                                 param.fx, param.fy, param.cx, param.cy);
    pinhole_camera.ConvertToJsonValue(value);

    value["serial_number"] = GetTagInMetadata("K4A_DEVICE_SERIAL_NUMBER");
    value["depth_mode"] = GetTagInMetadata("K4A_DEPTH_MODE");
    value["color_mode"] = GetTagInMetadata("K4A_COLOR_MODE");

    value["stream_length_usec"] =
            k4a_plugin::k4a_playback_get_last_timestamp_usec(handle_);
    auto color_mode = value["color_mode"].asString();
    size_t pos = color_mode.find('_');
    if (pos == std::string::npos) {
        utility::LogError("Unknown color format {}\n", color_mode);
        return value;
    }
    std::string resolution =
            std::string(color_mode.begin() + pos + 1, color_mode.end());
    if (resolution_to_width_height.count(resolution) == 0) {
        utility::LogError("Unknown resolution format {}\n", resolution);
        return value;
    }

    auto width_height = resolution_to_width_height.at(resolution);
    value["width"] = width_height.first;
    value["height"] = width_height.second;

    // For internal usages
    transformation_ = k4a_plugin::k4a_transformation_create(&calibration);

    return value;
}

bool MKVReader::SeekTimestamp(size_t timestamp) {
    if (!IsOpened()) {
        utility::LogError("Null file handler. Please call Open().\n");
        return false;
    }

    if (timestamp >= metadata_.stream_length_usec_) {
        utility::LogError("Timestamp {} exceeds maximum {} (us).\n", timestamp,
                          metadata_.stream_length_usec_);
        return false;
    }

    if (K4A_RESULT_SUCCEEDED !=
        k4a_plugin::k4a_playback_seek_timestamp(handle_, timestamp,
                                                K4A_PLAYBACK_SEEK_BEGIN)) {
        utility::LogError("Unable to go to timestamp {}\n", timestamp);
        return false;
    }
    return true;
}

std::shared_ptr<geometry::RGBDImage> MKVReader::NextFrame() {
    if (!IsOpened()) {
        utility::LogError("Null file handler. Please call Open().\n");
        return nullptr;
    }

    k4a_capture_t k4a_capture;
    k4a_stream_result_t res =
            k4a_plugin::k4a_playback_get_next_capture(handle_, &k4a_capture);
    if (K4A_STREAM_RESULT_EOF == res) {
        utility::LogInfo("EOF reached\n");
        is_eof_ = true;
        return nullptr;
    } else if (K4A_STREAM_RESULT_FAILED == res) {
        utility::LogInfo("Empty frame encountered, skip\n");
        return nullptr;
    }

    auto rgbd =
            AzureKinectSensor::DecompressCapture(k4a_capture, transformation_);
    k4a_plugin::k4a_capture_release(k4a_capture);

    return rgbd;
}
}  // namespace io
}  // namespace open3d
