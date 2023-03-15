// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/sensor/RGBDVideoMetadata.h"

#include <json/json.h>

#include <unordered_map>

namespace open3d {
namespace t {
namespace io {

bool RGBDVideoMetadata::ConvertToJsonValue(Json::Value &value) const {
    intrinsics_.ConvertToJsonValue(value);

    value["device_name"] = device_name_;
    value["serial_number"] = serial_number_;
    value["color_format"] = color_format_;
    value["depth_format"] = depth_format_;
    value["depth_scale"] = depth_scale_;

    value["stream_length_usec"] = stream_length_usec_;
    value["width"] = width_;
    value["height"] = height_;
    value["fps"] = fps_;

    return true;
}

bool RGBDVideoMetadata::ConvertFromJsonValue(const Json::Value &value) {
    intrinsics_.ConvertFromJsonValue(value);

    serial_number_ = value["serial_number"].asString();
    device_name_ = value["device_name"].asString();
    color_format_ = value["color_format"].asString();
    depth_format_ = value["depth_format"].asString();
    depth_scale_ = value["depth_scale"].asFloat();

    stream_length_usec_ = value["stream_length_usec"].asUInt64();
    width_ = value["width"].asInt();
    height_ = value["height"].asInt();
    fps_ = value["fps"].asDouble();

    return true;
}
}  // namespace io
}  // namespace t
}  // namespace open3d
