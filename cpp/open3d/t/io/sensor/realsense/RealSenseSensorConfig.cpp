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

#include "open3d/t/io/sensor/realsense/RealSenseSensorConfig.h"

#include <json/json.h>

#include <cstdlib>
#include <librealsense2/rs.hpp>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace io {

// clang-format off
// Maps invalid values to RS2_STREAM_ANY
//STRINGIFY_ENUM(rs2_stream, {
//        {RS2_STREAM_ANY, "RS2_STREAM_ANY"},
//        {RS2_STREAM_DEPTH, "RS2_STREAM_DEPTH"},
//        {RS2_STREAM_COLOR, "RS2_STREAM_COLOR"},
//        {RS2_STREAM_INFRARED, "RS2_STREAM_INFRARED"},
//        {RS2_STREAM_FISHEYE, "RS2_STREAM_FISHEYE"},
//        {RS2_STREAM_GYRO, "RS2_STREAM_GYRO"},
//        {RS2_STREAM_ACCEL, "RS2_STREAM_ACCEL"},
//        {RS2_STREAM_GPIO, "RS2_STREAM_GPIO"},
//        {RS2_STREAM_POSE, "RS2_STREAM_POSE"},
//        {RS2_STREAM_CONFIDENCE, "RS2_STREAM_CONFIDENCE"}
//});
//
//STRINGIFY_ENUM(rs2_format, {
//        {RS2_FORMAT_ANY, "RS2_FORMAT_ANY"},
//        {RS2_FORMAT_Z16, "RS2_FORMAT_Z16"},
//        {RS2_FORMAT_DISPARITY16, "RS2_FORMAT_DISPARITY16"},
//        {RS2_FORMAT_XYZ32F, "RS2_FORMAT_XYZ32F"},
//        {RS2_FORMAT_YUYV, "RS2_FORMAT_YUYV"},
//        {RS2_FORMAT_RGB8, "RS2_FORMAT_RGB8"},
//        {RS2_FORMAT_BGR8, "RS2_FORMAT_BGR8"},
//        {RS2_FORMAT_RGBA8, "RS2_FORMAT_RGBA8"},
//        {RS2_FORMAT_BGRA8, "RS2_FORMAT_BGRA8"},
//        {RS2_FORMAT_Y8, "RS2_FORMAT_Y8"},
//        {RS2_FORMAT_Y16, "RS2_FORMAT_Y16"},
//        {RS2_FORMAT_RAW10, "RS2_FORMAT_RAW10"},
//        {RS2_FORMAT_RAW16, "RS2_FORMAT_RAW16"},
//        {RS2_FORMAT_RAW8, "RS2_FORMAT_RAW8"},
//        {RS2_FORMAT_UYVY, "RS2_FORMAT_UYVY"},
//        {RS2_FORMAT_MOTION_RAW, "RS2_FORMAT_MOTION_RAW"},
//        {RS2_FORMAT_MOTION_XYZ32F, "RS2_FORMAT_MOTION_XYZ32F"},
//        {RS2_FORMAT_GPIO_RAW, "RS2_FORMAT_GPIO_RAW"},
//        {RS2_FORMAT_6DOF, "RS2_FORMAT_6DOF"},
//        {RS2_FORMAT_DISPARITY32, "RS2_FORMAT_DISPARITY32"},
//        {RS2_FORMAT_Y10BPACK, "RS2_FORMAT_Y10BPACK"},
//        {RS2_FORMAT_DISTANCE, "RS2_FORMAT_DISTANCE"},
//        {RS2_FORMAT_MJPEG, "RS2_FORMAT_MJPEG"},
//        {RS2_FORMAT_Y8I, "RS2_FORMAT_Y8I"},
//        {RS2_FORMAT_Y12I, "RS2_FORMAT_Y12I"},
//        {RS2_FORMAT_INZI, "RS2_FORMAT_INZI"},
//        {RS2_FORMAT_INVI, "RS2_FORMAT_INVI"},
//        {RS2_FORMAT_W10, "RS2_FORMAT_W10"},
//        {RS2_FORMAT_Z16H, "RS2_FORMAT_Z16H"}
//});
//
//STRINGIFY_ENUM(rs2_l500_visual_preset, {
//    {RS2_L500_VISUAL_PRESET_DEFAULT, "RS2_L500_VISUAL_PRESET_DEFAULT"},
//    {RS2_L500_VISUAL_PRESET_CUSTOM, "RS2_L500_VISUAL_PRESET_CUSTOM"},
//    {RS2_L500_VISUAL_PRESET_NO_AMBIENT, "RS2_L500_VISUAL_PRESET_NO_AMBIENT"},
//    {RS2_L500_VISUAL_PRESET_LOW_AMBIENT, "RS2_L500_VISUAL_PRESET_LOW_AMBIENT"},
//    {RS2_L500_VISUAL_PRESET_MAX_RANGE, "RS2_L500_VISUAL_PRESET_MAX_RANGE"},
//    {RS2_L500_VISUAL_PRESET_SHORT_RANGE, "RS2_L500_VISUAL_PRESET_SHORT_RANGE"}
//});
//
//STRINGIFY_ENUM(rs2_rs400_visual_preset, {
//        {RS2_RS400_VISUAL_PRESET_DEFAULT, "RS2_RS400_VISUAL_PRESET_DEFAULT"},
//        {RS2_RS400_VISUAL_PRESET_CUSTOM, "RS2_RS400_VISUAL_PRESET_CUSTOM"},
//        {RS2_RS400_VISUAL_PRESET_HAND, "RS2_RS400_VISUAL_PRESET_HAND"},
//        {RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY, "RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY"},
//        {RS2_RS400_VISUAL_PRESET_HIGH_DENSITY, "RS2_RS400_VISUAL_PRESET_HIGH_DENSITY"},
//        {RS2_RS400_VISUAL_PRESET_MEDIUM_DENSITY, "RS2_RS400_VISUAL_PRESET_MEDIUM_DENSITY"},
//        {RS2_RS400_VISUAL_PRESET_REMOVE_IR_PATTERN, "RS2_RS400_VISUAL_PRESET_REMOVE_IR_PATTERN"}
//});
//
//STRINGIFY_ENUM(rs2_sr300_visual_preset, {
//    {RS2_SR300_VISUAL_PRESET_DEFAULT, "RS2_SR300_VISUAL_PRESET_DEFAULT"},
//    {RS2_SR300_VISUAL_PRESET_SHORT_RANGE, "RS2_SR300_VISUAL_PRESET_SHORT_RANGE"},
//    {RS2_SR300_VISUAL_PRESET_LONG_RANGE, "RS2_SR300_VISUAL_PRESET_LONG_RANGE"},
//    {RS2_SR300_VISUAL_PRESET_BACKGROUND_SEGMENTATION, "RS2_SR300_VISUAL_PRESET_BACKGROUND_SEGMENTATION"},
//    {RS2_SR300_VISUAL_PRESET_GESTURE_RECOGNITION, "RS2_SR300_VISUAL_PRESET_GESTURE_RECOGNITION"},
//    {RS2_SR300_VISUAL_PRESET_OBJECT_SCANNING, "RS2_SR300_VISUAL_PRESET_OBJECT_SCANNING"},
//    {RS2_SR300_VISUAL_PRESET_FACE_ANALYTICS, "RS2_SR300_VISUAL_PRESET_FACE_ANALYTICS"},
//    {RS2_SR300_VISUAL_PRESET_FACE_LOGIN, "RS2_SR300_VISUAL_PRESET_FACE_LOGIN"},
//    {RS2_SR300_VISUAL_PRESET_GR_CURSOR, "RS2_SR300_VISUAL_PRESET_GR_CURSOR"},
//    {RS2_SR300_VISUAL_PRESET_MID_RANGE, "RS2_SR300_VISUAL_PRESET_MID_RANGE"},
//    {RS2_SR300_VISUAL_PRESET_IR_ONLY, "RS2_SR300_VISUAL_PRESET_IR_ONLY"}
//});

// clang-format on

static std::unordered_map<std::string, std::string> standard_config{
        {"serial", ""},
        {"color_format", "RS2_FORMAT_ANY"},
        {"color_resolution", "0,0"},
        {"color_fps", "0"},
        {"depth_format", "RS2_FORMAT_ANY"},
        {"depth_resolution", "0,0"},
        {"depth_fps", "0"},
        {"visual_preset", "VISUAL_PRESET_DEFAULT"}};

RealSenseSensorConfig::RealSenseSensorConfig() { config_ = standard_config; }

RealSenseSensorConfig::RealSenseSensorConfig(
        const std::unordered_map<std::string, std::string> &config)
    : RealSenseSensorConfig() {
    for (const auto &it : config) {
        config_[it.first] = it.second;
    }
}

bool RealSenseSensorConfig::ConvertToJsonValue(Json::Value &value) const {
    for (auto &kv : config_) {
        value[kv.first] = config_.at(kv.first);
    }
    return true;
}

bool RealSenseSensorConfig::ConvertFromJsonValue(const Json::Value &value) {
    for (auto &kv : config_) {
        config_.at(kv.first) = value[kv.first].asString();
    }
    return true;
}

void RealSenseSensorConfig::ConvertFromNativeConfig(
        const rs2::config &rs_config) {
    utility::LogError("Not Implemented!");
}

rs2::config RealSenseSensorConfig::ConvertToNativeConfig() const {
    rs2::config cfg;
    auto it = config_.find("serial");
    if (it != config_.cend() && !it->second.empty())
        cfg.enable_device(it->second);
    cfg.disable_all_streams();

    auto set_config = [this](const std::string &stream_type) {
        int width = 0, height = 0, fps = 0;
        rs2_format format = RS2_FORMAT_ANY;
        it = config_.find(stream_type + "_format");
        if (it != config_.cend() && !it->second.empty())
            format = enum_from_string<rs2_format>(it->second);
        it = config_.find(stream_type + "_resolution");
        if (it != config_.cend() && !it->second.empty()) {
            auto res = it->second.c_str();
            char *remaining;
            width = strtol(res,
                           &remaining);  // [640],480 - return 0 if bad format
            height = strtol(
                    remaining +
                    1);  // 640,[480] - return 0 if bad format or no comma
        }
        it = config_.find(stream_type + "_fps");
        if (it != config_.cend() && !it->second.empty()) fps = stoi(it->second);
        return std::tuple{width, height, format, fps};
    };

    int width = 0, height = 0, fps = 0;
    rs2_format format = RS2_FORMAT_ANY;
    std::tie(width, height, format, fps) = set_config("color");
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, format, fps);
    std::tie(width, height, format, fps) = set_config("depth");
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, format, fps);
    return cfg;
}

bool RealSenseSensorConfig::IsValidConfig() const {
    return ConvertToNativeConfig().can_resolve();
}

}  // namespace io
}  // namespace t
}  // namespace open3d
