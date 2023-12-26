// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/sensor/realsense/RealSenseSensor.h"

#include <json/json.h>

#include <string>
#include <thread>
#include <vector>

#include "open3d/t/io/sensor/realsense/RealSensePrivate.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

bool RealSenseSensor::ListDevices() {
    auto all_device_info = EnumerateDevices();
    if (all_device_info.empty()) {
        utility::LogWarning("No RealSense devices detected.");
        return false;
    } else {
        size_t sensor_index = 0;
        for (auto& dev_info : all_device_info) {
            utility::LogInfo("[{}] {}: {}", sensor_index++, dev_info.name,
                             dev_info.serial);
            for (auto& config : dev_info.valid_configs)
                utility::LogInfo("\t{}: [{}]", config.first,
                                 fmt::join(config.second, " | "));
        }
        utility::LogInfo(
                "Open3D only supports synchronized color and depth capture "
                "(color_fps = depth_fps).");
        return true;
    }
}

std::vector<RealSenseValidConfigs> RealSenseSensor::EnumerateDevices() {
    rs2::context ctx;
    std::vector<RealSenseValidConfigs> all_device_info;
    for (auto&& dev : ctx.query_devices()) {
        RealSenseValidConfigs cfg{dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER),
                                  dev.get_info(RS2_CAMERA_INFO_NAME),
                                  {{"color_format", {}},
                                   {"color_resolution", {}},
                                   {"color_fps", {}},
                                   {"depth_format", {}},
                                   {"depth_resolution", {}},
                                   {"depth_fps", {}},
                                   {"visual_preset", {}}}};
        for (auto&& sensor : dev.query_sensors()) {
            for (auto&& stream : sensor.get_stream_profiles()) {
                if (stream.stream_type() == RS2_STREAM_COLOR) {
                    cfg.valid_configs["color_format"].insert(
                            enum_to_string(stream.format()));
                    cfg.valid_configs["color_resolution"].insert(fmt::format(
                            "{},{}",
                            stream.as<rs2::video_stream_profile>().width(),
                            stream.as<rs2::video_stream_profile>().height()));
                    cfg.valid_configs["color_fps"].insert(
                            std::to_string(stream.fps()));
                } else if (stream.stream_type() == RS2_STREAM_DEPTH) {
                    cfg.valid_configs["depth_format"].insert(
                            enum_to_string(stream.format()));
                    cfg.valid_configs["depth_resolution"].insert(fmt::format(
                            "{},{}",
                            stream.as<rs2::video_stream_profile>().width(),
                            stream.as<rs2::video_stream_profile>().height()));
                    cfg.valid_configs["depth_fps"].insert(
                            std::to_string(stream.fps()));
                }
            }
            if (sensor.supports(RS2_OPTION_VISUAL_PRESET)) {
                std::string product_line =
                        dev.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);
                if (product_line == "L500") {
                    for (int k = 0;
                         k < static_cast<int>(RS2_L500_VISUAL_PRESET_COUNT);
                         ++k)
                        cfg.valid_configs["visual_preset"].insert(
                                enum_to_string(
                                        static_cast<rs2_l500_visual_preset>(
                                                k)));
                } else if (product_line == "RS400") {
                    for (int k = 0;
                         k < static_cast<int>(RS2_RS400_VISUAL_PRESET_COUNT);
                         ++k)
                        cfg.valid_configs["visual_preset"].insert(
                                enum_to_string(
                                        static_cast<rs2_rs400_visual_preset>(
                                                k)));
                } else if (product_line == "SR300") {
                    for (int k = 0;
                         k < static_cast<int>(RS2_SR300_VISUAL_PRESET_COUNT);
                         ++k)
                        cfg.valid_configs["visual_preset"].insert(
                                enum_to_string(
                                        static_cast<rs2_sr300_visual_preset>(
                                                k)));
                }
            }
        }
        all_device_info.emplace_back(cfg);
    }
    return all_device_info;
}

RealSenseSensor::RealSenseSensor()
    : pipe_{new rs2::pipeline},
      align_to_color_{new rs2::align(rs2_stream::RS2_STREAM_COLOR)},
      rs_config_{new rs2::config} {
    *rs_config_ = RealSenseSensorConfig().ConvertToNativeConfig();
}

RealSenseSensor::~RealSenseSensor() { StopCapture(); }

bool RealSenseSensor::InitSensor(const RealSenseSensorConfig& sensor_config,
                                 size_t sensor_index,
                                 const std::string& filename) try {
    *rs_config_ = sensor_config.ConvertToNativeConfig();
    if (sensor_config.config_.find("serial") == sensor_config.config_.cend()) {
        // serial number not specified, use sensor_index
        rs2::context ctx;
        auto device_list = ctx.query_devices();
        if (sensor_index >= device_list.size()) {
            utility::LogError(
                    "No device for sensor_index {}. Only {} devices detected.",
                    sensor_index, device_list.size());
        } else {
            rs_config_->enable_device(
                    device_list[(uint32_t)sensor_index].get_info(
                            RS2_CAMERA_INFO_SERIAL_NUMBER));
        }
    }
    if (!filename.empty()) {
        if (utility::filesystem::FileExists(filename_)) {
            enable_recording_ = false;
            utility::LogError("Will not overwrite existing file {}.", filename);
        }
        const std::string parent_dir =
                utility::filesystem::GetFileParentDirectory(filename_);
        if (!utility::filesystem::DirectoryExists(parent_dir)) {
            utility::filesystem::MakeDirectoryHierarchy(parent_dir);
        }
        filename_ = filename;
        enable_recording_ = true;
    } else {
        filename_.clear();
        enable_recording_ = false;
    }
    auto profile = rs_config_->resolve(*pipe_);
    auto dev = profile.get_device().first<rs2::depth_sensor>();
    auto it = sensor_config.config_.find("visual_preset");
    // unknown option will map to default for each product line
    std::string option_str{"RS2_VISUAL_PRESET_DEFAULT"};
    if (it != sensor_config.config_.cend() && !it->second.empty())
        option_str = it->second;
    std::string product_line = dev.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);
    if (product_line == "L500") {
        rs2_l500_visual_preset option;
        enum_from_string(option_str, option);
        if (option != RS2_L500_VISUAL_PRESET_DEFAULT)
            dev.set_option(RS2_OPTION_VISUAL_PRESET,
                           static_cast<float>(option));
    } else if (product_line == "RS400") {
        rs2_rs400_visual_preset option;
        enum_from_string(option_str, option);
        if (option != RS2_RS400_VISUAL_PRESET_DEFAULT)
            dev.set_option(RS2_OPTION_VISUAL_PRESET,
                           static_cast<float>(option));
    } else if (product_line == "SR300") {
        rs2_sr300_visual_preset option;
        enum_from_string(option_str, option);
        if (option != RS2_SR300_VISUAL_PRESET_DEFAULT)
            dev.set_option(RS2_OPTION_VISUAL_PRESET,
                           static_cast<float>(option));
    }
    metadata_.ConvertFromJsonValue(
            RealSenseSensorConfig::GetMetadataJson(profile));
    RealSenseSensorConfig::GetPixelDtypes(profile, metadata_);
    return true;
} catch (const rs2::error& e) {
    utility::LogError(
            "Invalid RealSense camera configuration, or camera not connected:"
            "\n{}: {}",
            rs2_exception_type_to_string(e.get_type()), e.what());
    return false;
}

bool RealSenseSensor::StartCapture(bool start_record) {
    if (is_capturing_) {
        utility::LogWarning("Capture already in progress.");
        return true;
    }
    try {
        is_recording_ = enable_recording_ && start_record;
        if (is_recording_) rs_config_->enable_record_to_file(filename_);
        const auto profile = pipe_->start(*rs_config_);
        // This step is repeated here since the user may bypass InitSensor()
        metadata_.ConvertFromJsonValue(
                RealSenseSensorConfig::GetMetadataJson(profile));
        RealSenseSensorConfig::GetPixelDtypes(profile, metadata_);

        is_capturing_ = true;
        utility::LogInfo(
                "Capture started with RealSense camera {}",
                profile.get_device().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        if (enable_recording_) {
            utility::LogInfo("Recording {}to bag file {}",
                             start_record ? "" : "[Paused] ", filename_);
        }
        return true;
    } catch (const rs2::error& e) {
        utility::LogError("StartCapture() failed: {}: {}",
                          rs2_exception_type_to_string(e.get_type()), e.what());
        return false;
    }
}

void RealSenseSensor::PauseRecord() {
    if (!enable_recording_ || !is_recording_) return;
    pipe_->get_active_profile().get_device().as<rs2::recorder>().pause();
    is_recording_ = false;
    utility::LogDebug("Recording paused.");
}

void RealSenseSensor::ResumeRecord() {
    if (!enable_recording_ || is_recording_) return;
    try {
        if (auto dev = pipe_->get_active_profile()
                               .get_device()
                               .as<rs2::recorder>()) {
            dev.resume();
            utility::LogDebug("Recording resumed.");
        } else {
            rs_config_->enable_record_to_file(filename_);
            pipe_.reset(new rs2::pipeline);
            pipe_->start(*rs_config_);
            utility::LogDebug("Recording started.");
        }
        is_recording_ = true;
    } catch (const rs2::error& e) {
        utility::LogError("ResumeRecord() failed: {}: {}",
                          rs2_exception_type_to_string(e.get_type()), e.what());
    }
}

geometry::RGBDImage RealSenseSensor::CaptureFrame(bool wait,
                                                  bool align_depth_to_color) {
    if (!is_capturing_) {
        utility::LogError("Please StartCapture() first.");
        return geometry::RGBDImage();
    }
    try {
        rs2::frameset frames;
        if (!((wait && pipe_->try_wait_for_frames(&frames)) ||
              (!wait && pipe_->poll_for_frames(&frames))))
            return geometry::RGBDImage();
        if (align_depth_to_color) frames = align_to_color_->process(frames);
        timestamp_ = uint64_t(frames.get_timestamp() * MILLISEC_TO_MICROSEC);
        // Copy frame data to Tensors
        const auto& color_frame = frames.get_color_frame();
        current_frame_.color_ = core::Tensor(
                static_cast<const uint8_t*>(color_frame.get_data()),
                {color_frame.get_height(), color_frame.get_width(),
                 metadata_.color_channels_},
                metadata_.color_dt_);
        const auto& depth_frame = frames.get_depth_frame();
        current_frame_.depth_ = core::Tensor(
                static_cast<const uint16_t*>(depth_frame.get_data()),
                {depth_frame.get_height(), depth_frame.get_width()},
                metadata_.depth_dt_);
        return current_frame_;
    } catch (const rs2::error& e) {
        utility::LogError("CaptureFrame() failed: {}: {}",
                          rs2_exception_type_to_string(e.get_type()), e.what());
        return geometry::RGBDImage();
    }
}

void RealSenseSensor::StopCapture() {
    if (is_capturing_) {
        pipe_->stop();
        is_recording_ = false;
        is_capturing_ = false;
        utility::LogInfo("Capture stopped.");
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d
