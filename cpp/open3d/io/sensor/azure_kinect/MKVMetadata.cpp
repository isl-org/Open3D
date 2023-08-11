// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/sensor/azure_kinect/MKVMetadata.h"

#include <json/json.h>

#include <unordered_map>

namespace open3d {
namespace io {

bool MKVMetadata::ConvertToJsonValue(Json::Value &value) const {
    intrinsics_.ConvertToJsonValue(value);

    value["serial_number"] = serial_number_;
    value["color_mode"] = color_mode_;
    value["depth_mode"] = depth_mode_;

    value["stream_length_usec"] = stream_length_usec_;
    value["width"] = width_;
    value["height"] = height_;

    return true;
}
bool MKVMetadata::ConvertFromJsonValue(const Json::Value &value) {
    intrinsics_.ConvertFromJsonValue(value);

    serial_number_ = value["serial_number"].asString();
    color_mode_ = value["color_mode"].asString();
    depth_mode_ = value["depth_mode"].asString();

    stream_length_usec_ = value["stream_length_usec"].asUInt64();
    width_ = value["width"].asInt();
    height_ = value["height"].asInt();

    return true;
}
}  // namespace io
}  // namespace open3d
