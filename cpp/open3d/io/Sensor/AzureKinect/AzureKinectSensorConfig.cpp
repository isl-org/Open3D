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

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensorConfig.h"

#include <json/json.h>
#include <k4a/k4a.h>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

#include "Open3D/IO/Sensor/AzureKinect/K4aPlugin.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace io {

static std::map<k4a_image_format_t, std::string> k4a_image_format_t_to_string{
        {K4A_IMAGE_FORMAT_COLOR_MJPG, "K4A_IMAGE_FORMAT_COLOR_MJPG"},
        {K4A_IMAGE_FORMAT_COLOR_NV12, "K4A_IMAGE_FORMAT_COLOR_NV12"},
        {K4A_IMAGE_FORMAT_COLOR_YUY2, "K4A_IMAGE_FORMAT_COLOR_YUY2"},
        {K4A_IMAGE_FORMAT_COLOR_BGRA32, "K4A_IMAGE_FORMAT_COLOR_BGRA32"},
        {K4A_IMAGE_FORMAT_DEPTH16, "K4A_IMAGE_FORMAT_DEPTH16"},
        {K4A_IMAGE_FORMAT_IR16, "K4A_IMAGE_FORMAT_IR16"},
        {K4A_IMAGE_FORMAT_CUSTOM, "K4A_IMAGE_FORMAT_CUSTOM"},
};

static std::map<k4a_color_resolution_t, std::string>
        k4a_color_resolution_t_to_string{
                {K4A_COLOR_RESOLUTION_OFF, "K4A_COLOR_RESOLUTION_OFF"},
                {K4A_COLOR_RESOLUTION_720P, "K4A_COLOR_RESOLUTION_720P"},
                {K4A_COLOR_RESOLUTION_1080P, "K4A_COLOR_RESOLUTION_1080P"},
                {K4A_COLOR_RESOLUTION_1440P, "K4A_COLOR_RESOLUTION_1440P"},
                {K4A_COLOR_RESOLUTION_1536P, "K4A_COLOR_RESOLUTION_1536P"},
                {K4A_COLOR_RESOLUTION_2160P, "K4A_COLOR_RESOLUTION_2160P"},
                {K4A_COLOR_RESOLUTION_3072P, "K4A_COLOR_RESOLUTION_3072P"},
        };

static std::map<k4a_depth_mode_t, std::string> k4a_depth_mode_t_to_string{
        {K4A_DEPTH_MODE_OFF, "K4A_DEPTH_MODE_OFF"},
        {K4A_DEPTH_MODE_NFOV_2X2BINNED, "K4A_DEPTH_MODE_NFOV_2X2BINNED"},
        {K4A_DEPTH_MODE_NFOV_UNBINNED, "K4A_DEPTH_MODE_NFOV_UNBINNED"},
        {K4A_DEPTH_MODE_WFOV_2X2BINNED, "K4A_DEPTH_MODE_WFOV_2X2BINNED"},
        {K4A_DEPTH_MODE_WFOV_UNBINNED, "K4A_DEPTH_MODE_WFOV_UNBINNED"},
        {K4A_DEPTH_MODE_PASSIVE_IR, "K4A_DEPTH_MODE_PASSIVE_IR"},
};

static std::map<k4a_fps_t, std::string> k4a_fps_t_to_string{
        {K4A_FRAMES_PER_SECOND_5, "K4A_FRAMES_PER_SECOND_5"},
        {K4A_FRAMES_PER_SECOND_15, "K4A_FRAMES_PER_SECOND_15"},
        {K4A_FRAMES_PER_SECOND_30, "K4A_FRAMES_PER_SECOND_30"},
};

static std::map<k4a_wired_sync_mode_t, std::string>
        k4a_wired_sync_mode_t_to_string{
                {K4A_WIRED_SYNC_MODE_STANDALONE,
                 "K4A_WIRED_SYNC_MODE_STANDALONE"},
                {K4A_WIRED_SYNC_MODE_MASTER, "K4A_WIRED_SYNC_MODE_MASTER"},
                {K4A_WIRED_SYNC_MODE_SUBORDINATE,
                 "K4A_WIRED_SYNC_MODE_SUBORDINATE"},
        };

static std::map<std::string, k4a_image_format_t> string_to_k4a_image_format_t({
        {"K4A_IMAGE_FORMAT_COLOR_MJPG", K4A_IMAGE_FORMAT_COLOR_MJPG},
        {"K4A_IMAGE_FORMAT_COLOR_NV12", K4A_IMAGE_FORMAT_COLOR_NV12},
        {"K4A_IMAGE_FORMAT_COLOR_YUY2", K4A_IMAGE_FORMAT_COLOR_YUY2},
        {"K4A_IMAGE_FORMAT_COLOR_BGRA32", K4A_IMAGE_FORMAT_COLOR_BGRA32},
        {"K4A_IMAGE_FORMAT_DEPTH16", K4A_IMAGE_FORMAT_DEPTH16},
        {"K4A_IMAGE_FORMAT_IR16", K4A_IMAGE_FORMAT_IR16},
        {"K4A_IMAGE_FORMAT_CUSTOM", K4A_IMAGE_FORMAT_CUSTOM},
});

static std::map<std::string, k4a_color_resolution_t>
        string_to_k4a_color_resolution_t{
                {"K4A_COLOR_RESOLUTION_OFF", K4A_COLOR_RESOLUTION_OFF},
                {"K4A_COLOR_RESOLUTION_720P", K4A_COLOR_RESOLUTION_720P},
                {"K4A_COLOR_RESOLUTION_1080P", K4A_COLOR_RESOLUTION_1080P},
                {"K4A_COLOR_RESOLUTION_1440P", K4A_COLOR_RESOLUTION_1440P},
                {"K4A_COLOR_RESOLUTION_1536P", K4A_COLOR_RESOLUTION_1536P},
                {"K4A_COLOR_RESOLUTION_2160P", K4A_COLOR_RESOLUTION_2160P},
                {"K4A_COLOR_RESOLUTION_3072P", K4A_COLOR_RESOLUTION_3072P},
        };

static std::map<std::string, k4a_depth_mode_t> string_to_k4a_depth_mode_t{
        {"K4A_DEPTH_MODE_OFF", K4A_DEPTH_MODE_OFF},
        {"K4A_DEPTH_MODE_NFOV_2X2BINNED", K4A_DEPTH_MODE_NFOV_2X2BINNED},
        {"K4A_DEPTH_MODE_NFOV_UNBINNED", K4A_DEPTH_MODE_NFOV_UNBINNED},
        {"K4A_DEPTH_MODE_WFOV_2X2BINNED", K4A_DEPTH_MODE_WFOV_2X2BINNED},
        {"K4A_DEPTH_MODE_WFOV_UNBINNED", K4A_DEPTH_MODE_WFOV_UNBINNED},
        {"K4A_DEPTH_MODE_PASSIVE_IR", K4A_DEPTH_MODE_PASSIVE_IR},
};

static std::map<std::string, k4a_fps_t> string_to_k4a_fps_t{
        {"K4A_FRAMES_PER_SECOND_5", K4A_FRAMES_PER_SECOND_5},
        {"K4A_FRAMES_PER_SECOND_15", K4A_FRAMES_PER_SECOND_15},
        {"K4A_FRAMES_PER_SECOND_30", K4A_FRAMES_PER_SECOND_30},
};

static std::map<std::string, k4a_wired_sync_mode_t>
        string_to_k4a_wired_sync_mode_t{
                {"K4A_WIRED_SYNC_MODE_STANDALONE",
                 K4A_WIRED_SYNC_MODE_STANDALONE},
                {"K4A_WIRED_SYNC_MODE_MASTER", K4A_WIRED_SYNC_MODE_MASTER},
                {"K4A_WIRED_SYNC_MODE_SUBORDINATE",
                 K4A_WIRED_SYNC_MODE_SUBORDINATE},
        };

static std::unordered_map<std::string, std::string> standard_config{
        {"color_format", "K4A_IMAGE_FORMAT_COLOR_MJPG"},
        {"color_resolution", "K4A_COLOR_RESOLUTION_720P"},
        {"depth_mode", "K4A_DEPTH_MODE_WFOV_2X2BINNED"},
        {"camera_fps", "K4A_FRAMES_PER_SECOND_30"},
        {"synchronized_images_only", "false"},
        {"depth_delay_off_color_usec", "0"},
        {"wired_sync_mode", "K4A_WIRED_SYNC_MODE_STANDALONE"},
        {"subordinate_delay_off_master_usec", "0"},
        {"disable_streaming_indicator", "false"},
};

AzureKinectSensorConfig::AzureKinectSensorConfig() {
    config_ = standard_config;
}

bool AzureKinectSensorConfig::IsValidConfig(
        const std::unordered_map<std::string, std::string> &config,
        bool verbose) {
    bool rc = true;

    // No extra keys
    for (const auto &it : config) {
        if (standard_config.find(it.first) == standard_config.end()) {
            rc = false;
            utility::LogWarning("IsValidConfig: Unrecognized key {}", it.first);
        }
    }

    if (config.count("color_format") != 0 &&
        string_to_k4a_image_format_t.count(config.at("color_format")) == 0) {
        rc = false;
        if (verbose) {
            utility::LogWarning("IsValidConfig: color_format invalid");
        }
    }

    if (config.count("color_resolution") != 0 &&
        string_to_k4a_color_resolution_t.count(config.at("color_resolution")) ==
                0) {
        rc = false;
        if (verbose) {
            utility::LogWarning("IsValidConfig: color_resolution invalid");
        }
    }

    if (config.count("depth_mode") != 0 &&
        string_to_k4a_depth_mode_t.count(config.at("depth_mode")) == 0) {
        rc = false;
        if (verbose) {
            utility::LogWarning("IsValidConfig: depth_mode invalid");
        }
    }

    if (config.count("camera_fps") != 0 &&
        string_to_k4a_fps_t.count(config.at("camera_fps")) == 0) {
        rc = false;
        if (verbose) {
            utility::LogWarning("IsValidConfig: camera_fps invalid");
        }
    } else {
        if (config.count("camera_fps") != 0 &&
            config.count("color_resolution") != 0 &&
            config.at("camera_fps") == "K4A_FRAMES_PER_SECOND_30" &&
            config.at("color_resolution") == "K4A_COLOR_RESOLUTION_3072P") {
            rc = false;
            if (verbose) {
                utility::LogWarning(
                        "K4A_COLOR_RESOLUTION_3072P does not support 30 FPS");
            }
        }
    }

    if (config.count("synchronized_images_only") != 0 &&
        config.at("synchronized_images_only") != "true" &&
        config.at("synchronized_images_only") != "false") {
        rc = false;
        if (verbose) {
            utility::LogWarning(
                    "IsValidConfig: synchronized_images_only invalid");
        }
    }

    if (config.count("wired_sync_mode") != 0 &&
        string_to_k4a_wired_sync_mode_t.count(config.at("wired_sync_mode")) ==
                0) {
        rc = false;
        if (verbose) {
            utility::LogWarning("IsValidConfig: wired_sync_mode invalid");
        }
    }

    if (config.count("disable_streaming_indicator") != 0 &&
        config.at("disable_streaming_indicator") != "true" &&
        config.at("disable_streaming_indicator") != "false") {
        rc = false;
        if (verbose) {
            utility::LogWarning(
                    "IsValidConfig: disable_streaming_indicator invalid");
        }
    }

    return rc;
}

AzureKinectSensorConfig::AzureKinectSensorConfig(
        const std::unordered_map<std::string, std::string> &config)
    : AzureKinectSensorConfig() {
    if (!IsValidConfig(config)) {
        utility::LogWarning("Invalid configs, default configs will be used.");
        return;
    }
    for (const auto &it : config) {
        config_[it.first] = it.second;
    }
}

bool AzureKinectSensorConfig::ConvertToJsonValue(Json::Value &value) const {
    // clang-format off
    value["color_format"]                      = config_.at("color_format");
    value["color_resolution"]                  = config_.at("color_resolution");
    value["depth_mode"]                        = config_.at("depth_mode");
    value["camera_fps"]                        = config_.at("camera_fps");
    value["synchronized_images_only"]          = config_.at("synchronized_images_only");
    value["depth_delay_off_color_usec"]        = config_.at("depth_delay_off_color_usec");
    value["wired_sync_mode"]                   = config_.at("wired_sync_mode");
    value["subordinate_delay_off_master_usec"] = config_.at("subordinate_delay_off_master_usec");
    value["disable_streaming_indicator"]       = config_.at("disable_streaming_indicator");
    // clang-format on
    return true;
}

bool AzureKinectSensorConfig::ConvertFromJsonValue(const Json::Value &value) {
    // clang-format off
    config_["color_format"]                      = value["color_format"].asString();
    config_["color_resolution"]                  = value["color_resolution"].asString();
    config_["depth_mode"]                        = value["depth_mode"].asString();
    config_["camera_fps"]                        = value["camera_fps"].asString();
    config_["synchronized_images_only"]          = value["synchronized_images_only"].asString();
    config_["depth_delay_off_color_usec"]        = value["depth_delay_off_color_usec"].asString();
    config_["wired_sync_mode"]                   = value["wired_sync_mode"].asString();
    config_["subordinate_delay_off_master_usec"] = value["subordinate_delay_off_master_usec"].asString();
    config_["disable_streaming_indicator"]       = value["disable_streaming_indicator"].asString();
    // clang-format on

    return IsValidConfig(config_, true);
}

void AzureKinectSensorConfig::ConvertFromNativeConfig(
        const k4a_device_configuration_t &k4a_config) {
    // clang-format off
    config_["color_format"]                      = k4a_image_format_t_to_string.at(k4a_config.color_format);
    config_["color_resolution"]                  = k4a_color_resolution_t_to_string.at(k4a_config.color_resolution);
    config_["depth_mode"]                        = k4a_depth_mode_t_to_string.at(k4a_config.depth_mode);
    config_["camera_fps"]                        = k4a_fps_t_to_string.at(k4a_config.camera_fps);
    config_["synchronized_images_only"]          = k4a_config.synchronized_images_only ? "true" : "false";
    config_["depth_delay_off_color_usec"]        = std::to_string(k4a_config.depth_delay_off_color_usec);
    config_["wired_sync_mode"]                   = k4a_wired_sync_mode_t_to_string.at(k4a_config.wired_sync_mode);
    config_["subordinate_delay_off_master_usec"] = std::to_string(k4a_config.subordinate_delay_off_master_usec);
    config_["disable_streaming_indicator"]       = k4a_config.disable_streaming_indicator ? "true" : "false";
    // clang-format on
    if (!IsValidConfig(config_)) {
        utility::LogError(
                "Internal error, invalid configs. Maybe SDK version is "
                "mismatched.");
    }
}

k4a_device_configuration_t AzureKinectSensorConfig::ConvertToNativeConfig()
        const {
    k4a_device_configuration_t k4a_config;
    // clang-format off
    k4a_config.color_format                      = string_to_k4a_image_format_t.at(config_.at("color_format"));
    k4a_config.color_resolution                  = string_to_k4a_color_resolution_t.at(config_.at("color_resolution"));
    k4a_config.depth_mode                        = string_to_k4a_depth_mode_t.at(config_.at("depth_mode"));
    k4a_config.camera_fps                        = string_to_k4a_fps_t.at(config_.at("camera_fps"));
    k4a_config.synchronized_images_only          = config_.at("synchronized_images_only") == "true";
    k4a_config.depth_delay_off_color_usec        = static_cast<int32_t>(std::stoi(config_.at("depth_delay_off_color_usec")));
    k4a_config.wired_sync_mode                   = string_to_k4a_wired_sync_mode_t.at(config_.at("wired_sync_mode"));
    k4a_config.subordinate_delay_off_master_usec = static_cast<uint32_t>(std::stoi(config_.at("subordinate_delay_off_master_usec")));
    k4a_config.disable_streaming_indicator       = config_.at("disable_streaming_indicator") == "true";
    // clang-format on
    return k4a_config;
}

}  // namespace io
}  // namespace open3d
