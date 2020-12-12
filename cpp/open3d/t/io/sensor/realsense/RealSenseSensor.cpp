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

// TODO: validate channels_color_, dt_color_

#include "open3d/t/io/sensor/realsense/RealSenseSensor.h"

#include <librealsense2/rs.hpp>
#include <string>
#include <vector>

#include "open3d/utility/Console.h"

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
                    for (int k = 0; k < (int)RS2_L500_VISUAL_PRESET_COUNT; ++k)
                        cfg.valid_configs["visual_preset"].insert(
                                enum_to_string(
                                        static_cast<rs2_l500_visual_preset>(
                                                k)));
                } else if (product_line == "RS400") {
                    for (int k = 0; k < (int)RS2_RS400_VISUAL_PRESET_COUNT; ++k)
                        cfg.valid_configs["visual_preset"].insert(
                                enum_to_string(
                                        static_cast<rs2_rs400_visual_preset>(
                                                k)));
                } else if (product_line == "SR300") {
                    for (int k = 0; k < (int)RS2_SR300_VISUAL_PRESET_COUNT; ++k)
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
      rs_config_{new rs2::config} {}

RealSenseSensor::~RealSenseSensor() override { StopCapture(); }

bool RealSenseSensor::InitSensor(const RealSenseSensorConfig& sensor_config,
                                 size_t sensor_index,
                                 const std::string& filename) {
    /* https://www.intel.com/content/www/us/en/support/articles/000028416/emerging-technologies/intel-realsense-technology.html
     * auto sensor = profile.get_device().first();
    sensor.set_option(rs2_option::RS2_OPTION_VISUAL_PRESET,
    rs2_rs400_visual_preset::RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);
     */

    *rs_config_ = sensor_config.ConvertToNativeConfig();
    if (!filename.empty()) {
        filename_ = filename;
        enable_recording_ = true;
        rs_config_->enable_record_to_file(bag_file);
    } else {
        filename_.clear();
        enable_recording_ = false;
    }
    if (rs_config_->can_resolve())
        return true;
    else {
        utility::LogWarning(
                "Invalid RealSense camera configuration specified, or camera "
                "not connected.");
        return false;
    }
}

bool RealSenseSensor::StartCapture(bool start_record) {
    if (is_capturing_) {
        utility::LogWarning("Capture already in progress.");
        return true;
    }
    try {
        auto profile = pipe_->start(cfg);
    } catch (const rs2::error& e) {
        utility::LogError("StartCapture() failed: {}: {}",
                          rs2_exception_type_to_string(e.get_type()), e.what());
        return false;
    }
    is_capturing_ = true;
    if (enable_recording_) {
        utility::LogInfo("Recording to bag file {}", filename_);
        if (!start_record) profile.get_device().as<rs2::recorder>().pause();
    }
    is_recording_ = enable_recording_ && start_record;
    return true;
}

void RealSenseSensor::PauseRecord() {
    if (!enable_recording_ || !is_recording_) return;
    pipe_->get_active_profile().get_device().as<rs2::recorder>().pause();
    is_recording_ = false;
}

void RealSenseSensor::ResumeRecord() {
    if (!enable_recording_ || is_recording_) return;
    try {
        pipe_->get_active_profile().get_device().as<rs2::recorder>().resume();
    } catch (const rs2::error& e) {
        utility::LogError("ResumeRecord() failed: {}: {}",
                          rs2_exception_type_to_string(e.get_type()), e.what());
        return false;
    }
    is_recording_ = true;
}

geometry::RGBDImage RealSenseSensor::CaptureFrame(bool wait,
                                                  bool align_depth_to_color) {
    if (!is_capturing) {
        utility::LogError("Please StartCapture() first.");
        return geometry::RGBDImage();
    }
    try {
        if (!(wait && pipe_->try_wait_for_frames(&frames) ||
              !wait && pipe_->poll_for_frames(&frames)))
            return geometry::RGBDImage();
        if (align_depth_to_color) frames = align_to_color_->process(frames);
        const auto& color_frame = frames.get_color_frame();
        // Copy frame data to Tensors
        current_frame_.color_ = core::Tensor(
                static_cast<const uint8_t*>(color_frame.get_data()),
                {color_frame.get_height(), color_frame.get_width(),
                 channels_color_},
                dt_color_);
        const auto& depth_frame = frames.get_depth_frame();
        current_frame_.depth_ = core::Tensor(
                static_cast<const uint16_t*>(depth_frame.get_data()),
                {depth_frame.get_height(), depth_frame.get_width()}, dt_depth_);
        return current_frame_;
    } catch (const rs2::error& e) {
        utility::LogError("CaptureFrame() failed: {}: {}",
                          rs2_exception_type_to_string(e.get_type()), e.what());
        return geometry::RGBDImage();
    }
}

void RealSenseSensor::StopCapture() {
    pipe_->stop();
    is_recording_ = false;
    is_capturing_ = false;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
