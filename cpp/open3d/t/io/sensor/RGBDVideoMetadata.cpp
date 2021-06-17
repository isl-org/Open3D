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
