// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/sensor/realsense/RealSenseSensorConfig.h"

#include <json/json.h>

#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/t/io/sensor/RGBDVideoMetadata.h"
#include "open3d/t/io/sensor/realsense/RealSensePrivate.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

// clang-format off
//  Bidirectional string conversions for RS2 enums. Invalid values are mapped to
//  the first entry.  Reference:
//  - https://intelrealsense.github.io/librealsense/doxygen/rs__sensor_8h.html

/// RS2 stream types
STRINGIFY_ENUM(rs2_stream, {
        {RS2_STREAM_ANY, "RS2_STREAM_ANY"},
        {RS2_STREAM_DEPTH, "RS2_STREAM_DEPTH"},
        {RS2_STREAM_COLOR, "RS2_STREAM_COLOR"},
        {RS2_STREAM_INFRARED, "RS2_STREAM_INFRARED"},
        {RS2_STREAM_FISHEYE, "RS2_STREAM_FISHEYE"},
        {RS2_STREAM_GYRO, "RS2_STREAM_GYRO"},
        {RS2_STREAM_ACCEL, "RS2_STREAM_ACCEL"},
        {RS2_STREAM_GPIO, "RS2_STREAM_GPIO"},
        {RS2_STREAM_POSE, "RS2_STREAM_POSE"},
        {RS2_STREAM_CONFIDENCE, "RS2_STREAM_CONFIDENCE"}
});

/// RS2 pixel formats
STRINGIFY_ENUM(rs2_format, {
        {RS2_FORMAT_ANY, "RS2_FORMAT_ANY"},
        {RS2_FORMAT_Z16, "RS2_FORMAT_Z16"},
        {RS2_FORMAT_DISPARITY16, "RS2_FORMAT_DISPARITY16"},
        {RS2_FORMAT_XYZ32F, "RS2_FORMAT_XYZ32F"},
        {RS2_FORMAT_YUYV, "RS2_FORMAT_YUYV"},
        {RS2_FORMAT_RGB8, "RS2_FORMAT_RGB8"},
        {RS2_FORMAT_BGR8, "RS2_FORMAT_BGR8"},
        {RS2_FORMAT_RGBA8, "RS2_FORMAT_RGBA8"},
        {RS2_FORMAT_BGRA8, "RS2_FORMAT_BGRA8"},
        {RS2_FORMAT_Y8, "RS2_FORMAT_Y8"},
        {RS2_FORMAT_Y16, "RS2_FORMAT_Y16"},
        {RS2_FORMAT_RAW10, "RS2_FORMAT_RAW10"},
        {RS2_FORMAT_RAW16, "RS2_FORMAT_RAW16"},
        {RS2_FORMAT_RAW8, "RS2_FORMAT_RAW8"},
        {RS2_FORMAT_UYVY, "RS2_FORMAT_UYVY"},
        {RS2_FORMAT_MOTION_RAW, "RS2_FORMAT_MOTION_RAW"},
        {RS2_FORMAT_MOTION_XYZ32F, "RS2_FORMAT_MOTION_XYZ32F"},
        {RS2_FORMAT_GPIO_RAW, "RS2_FORMAT_GPIO_RAW"},
        {RS2_FORMAT_6DOF, "RS2_FORMAT_6DOF"},
        {RS2_FORMAT_DISPARITY32, "RS2_FORMAT_DISPARITY32"},
        {RS2_FORMAT_Y10BPACK, "RS2_FORMAT_Y10BPACK"},
        {RS2_FORMAT_DISTANCE, "RS2_FORMAT_DISTANCE"},
        {RS2_FORMAT_MJPEG, "RS2_FORMAT_MJPEG"},
        {RS2_FORMAT_Y8I, "RS2_FORMAT_Y8I"},
        {RS2_FORMAT_Y12I, "RS2_FORMAT_Y12I"},
        {RS2_FORMAT_INZI, "RS2_FORMAT_INZI"},
        {RS2_FORMAT_INVI, "RS2_FORMAT_INVI"},
        {RS2_FORMAT_W10, "RS2_FORMAT_W10"},
        {RS2_FORMAT_Z16H, "RS2_FORMAT_Z16H"}
});

/// RS2 visual presets for L500 devices
STRINGIFY_ENUM(rs2_l500_visual_preset, {
    {RS2_L500_VISUAL_PRESET_DEFAULT, "RS2_L500_VISUAL_PRESET_DEFAULT"},
    {RS2_L500_VISUAL_PRESET_CUSTOM, "RS2_L500_VISUAL_PRESET_CUSTOM"},
    {RS2_L500_VISUAL_PRESET_NO_AMBIENT, "RS2_L500_VISUAL_PRESET_NO_AMBIENT"},
    {RS2_L500_VISUAL_PRESET_LOW_AMBIENT, "RS2_L500_VISUAL_PRESET_LOW_AMBIENT"},
    {RS2_L500_VISUAL_PRESET_MAX_RANGE, "RS2_L500_VISUAL_PRESET_MAX_RANGE"},
    {RS2_L500_VISUAL_PRESET_SHORT_RANGE, "RS2_L500_VISUAL_PRESET_SHORT_RANGE"}
});

/// RS2 visual presets for RS400 devices
STRINGIFY_ENUM(rs2_rs400_visual_preset, {
        {RS2_RS400_VISUAL_PRESET_DEFAULT, "RS2_RS400_VISUAL_PRESET_DEFAULT"},
        {RS2_RS400_VISUAL_PRESET_CUSTOM, "RS2_RS400_VISUAL_PRESET_CUSTOM"},
        {RS2_RS400_VISUAL_PRESET_HAND, "RS2_RS400_VISUAL_PRESET_HAND"},
        {RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY, "RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY"},
        {RS2_RS400_VISUAL_PRESET_HIGH_DENSITY, "RS2_RS400_VISUAL_PRESET_HIGH_DENSITY"},
        {RS2_RS400_VISUAL_PRESET_MEDIUM_DENSITY, "RS2_RS400_VISUAL_PRESET_MEDIUM_DENSITY"},
        {RS2_RS400_VISUAL_PRESET_REMOVE_IR_PATTERN, "RS2_RS400_VISUAL_PRESET_REMOVE_IR_PATTERN"}
});

// RS2 visual presets for SR300 devices
STRINGIFY_ENUM(rs2_sr300_visual_preset, {
    {RS2_SR300_VISUAL_PRESET_DEFAULT, "RS2_SR300_VISUAL_PRESET_DEFAULT"},
    {RS2_SR300_VISUAL_PRESET_SHORT_RANGE, "RS2_SR300_VISUAL_PRESET_SHORT_RANGE"},
    {RS2_SR300_VISUAL_PRESET_LONG_RANGE, "RS2_SR300_VISUAL_PRESET_LONG_RANGE"},
    {RS2_SR300_VISUAL_PRESET_BACKGROUND_SEGMENTATION, "RS2_SR300_VISUAL_PRESET_BACKGROUND_SEGMENTATION"},
    {RS2_SR300_VISUAL_PRESET_GESTURE_RECOGNITION, "RS2_SR300_VISUAL_PRESET_GESTURE_RECOGNITION"},
    {RS2_SR300_VISUAL_PRESET_OBJECT_SCANNING, "RS2_SR300_VISUAL_PRESET_OBJECT_SCANNING"},
    {RS2_SR300_VISUAL_PRESET_FACE_ANALYTICS, "RS2_SR300_VISUAL_PRESET_FACE_ANALYTICS"},
    {RS2_SR300_VISUAL_PRESET_FACE_LOGIN, "RS2_SR300_VISUAL_PRESET_FACE_LOGIN"},
    {RS2_SR300_VISUAL_PRESET_GR_CURSOR, "RS2_SR300_VISUAL_PRESET_GR_CURSOR"},
    {RS2_SR300_VISUAL_PRESET_MID_RANGE, "RS2_SR300_VISUAL_PRESET_MID_RANGE"},
    {RS2_SR300_VISUAL_PRESET_IR_ONLY, "RS2_SR300_VISUAL_PRESET_IR_ONLY"}
});
// clang-format on

std::pair<core::Dtype, uint8_t> RealSenseSensorConfig::get_dtype_channels(
        int rs2_format_enum) {
    static const std::unordered_map<rs2_format, core::Dtype> format_dtype = {
            {RS2_FORMAT_Z16, core::UInt16},  {RS2_FORMAT_YUYV, core::UInt8},
            {RS2_FORMAT_RGB8, core::UInt8},  {RS2_FORMAT_BGR8, core::UInt8},
            {RS2_FORMAT_RGBA8, core::UInt8}, {RS2_FORMAT_BGRA8, core::UInt8},
            {RS2_FORMAT_Y8, core::UInt8},    {RS2_FORMAT_Y16, core::UInt16}};
    static const std::unordered_map<rs2_format, uint8_t> format_channels = {
            {RS2_FORMAT_Z16, 1},  {RS2_FORMAT_YUYV, 2},  {RS2_FORMAT_RGB8, 3},
            {RS2_FORMAT_BGR8, 3}, {RS2_FORMAT_RGBA8, 4}, {RS2_FORMAT_BGRA8, 4},
            {RS2_FORMAT_Y8, 1},   {RS2_FORMAT_Y16, 1}};

    return std::make_pair(
            format_dtype.at(static_cast<rs2_format>(rs2_format_enum)),
            format_channels.at(static_cast<rs2_format>(rs2_format_enum)));
}

static std::unordered_map<std::string, std::string> standard_config{
        {"serial", ""},
        {"color_format", "RS2_FORMAT_ANY"},
        {"color_resolution", "0,0"},
        {"depth_format", "RS2_FORMAT_ANY"},
        {"depth_resolution", "0,0"},
        {"fps", "0"},
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

rs2::config RealSenseSensorConfig::ConvertToNativeConfig() const {
    rs2::config cfg;
    auto it = config_.find("serial");
    if (it != config_.cend() && !it->second.empty())
        cfg.enable_device(it->second);
    cfg.disable_all_streams();

    auto set_config = [this](const std::string &stream_type) {
        int width = 0, height = 0;
        rs2_format format = RS2_FORMAT_ANY;
        auto it = config_.find(stream_type + "_format");
        if (it != config_.cend() && !it->second.empty())
            enum_from_string(it->second, format);
        it = config_.find(stream_type + "_resolution");
        if (it != config_.cend() && !it->second.empty()) {
            auto res = it->second.c_str();
            char *remaining;
            width = strtol(res, &remaining,
                           10);  // [640],480 - return 0 if bad format
            height = strtol(
                    remaining + 1, nullptr,
                    10);  // 640,[480] - return 0 if bad format or no comma
        }
        return std::make_tuple(width, height, format);
    };

    int width = 0, height = 0, fps = 0;
    rs2_format format = RS2_FORMAT_ANY;
    it = config_.find("fps");
    if (it != config_.cend() && !it->second.empty()) fps = stoi(it->second);
    std::tie(width, height, format) = set_config("color");
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, format, fps);
    std::tie(width, height, format) = set_config("depth");
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, format, fps);
    return cfg;
}

void RealSenseSensorConfig::GetPixelDtypes(const rs2::pipeline_profile &profile,
                                           RGBDVideoMetadata &metadata) {
    const auto rs_color = profile.get_stream(RS2_STREAM_COLOR)
                                  .as<rs2::video_stream_profile>();
    std::tie(metadata.color_dt_, metadata.color_channels_) =
            RealSenseSensorConfig::get_dtype_channels(
                    static_cast<int>(rs_color.format()));
    if (metadata.color_dt_ != core::UInt8) {
        utility::LogError("Only 8 bit unsigned int color is supported!");
    }
    const auto rs_depth = profile.get_stream(RS2_STREAM_DEPTH)
                                  .as<rs2::video_stream_profile>();
    metadata.depth_dt_ = RealSenseSensorConfig::get_dtype_channels(
                                 static_cast<int>(rs_depth.format()))
                                 .first;
    if (metadata.depth_dt_ != core::UInt16) {
        utility::LogError("Only 16 bit unsigned int depth is supported!");
    }
}

Json::Value RealSenseSensorConfig::GetMetadataJson(
        const rs2::pipeline_profile &profile) {
    if (!profile) {
        utility::LogError("Invalid RealSense pipeline profile.");
    }
    Json::Value value;

    const auto rs_device = profile.get_device();
    const auto rs_depth = profile.get_stream(RS2_STREAM_DEPTH)
                                  .as<rs2::video_stream_profile>();
    const auto rs_color = profile.get_stream(RS2_STREAM_COLOR)
                                  .as<rs2::video_stream_profile>();

    rs2_intrinsics rgb_intr = rs_color.get_intrinsics();
    camera::PinholeCameraIntrinsic pinhole_camera;
    pinhole_camera.SetIntrinsics(rgb_intr.width, rgb_intr.height, rgb_intr.fx,
                                 rgb_intr.fy, rgb_intr.ppx, rgb_intr.ppy);
    // TODO: Add support for distortion
    pinhole_camera.ConvertToJsonValue(value);

    value["device_name"] = rs_device.get_info(RS2_CAMERA_INFO_NAME);
    value["serial_number"] = rs_device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    value["depth_format"] = enum_to_string(rs_depth.format())
                                    .substr(11);  // remove RS2_FORMAT_ prefix
    value["depth_scale"] =
            1. / rs_device.first<rs2::depth_sensor>()
                         .get_depth_scale();  // convert meters -> m^(-1)
    value["color_format"] = enum_to_string(rs_color.format())
                                    .substr(11);  // remove RS2_FORMAT_ prefix
    value["fps"] = rs_color.fps();
    if (value["fps"] != rs_depth.fps()) {
        utility::LogError(
                "Different frame rates for color ({} fps) and depth ({} fps) "
                "streams is not supported.",
                value["fps"], rs_depth.fps());
    }
    if (rs_device.is<rs2::playback>()) {
        value["stream_length_usec"] =
                std::chrono::duration_cast<std::chrono::microseconds>(
                        rs_device.as<rs2::playback>().get_duration())
                        .count();
    }

    return value;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
