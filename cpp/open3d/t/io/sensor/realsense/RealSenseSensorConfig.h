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

#pragma once

#include <librealsense2/rs.hpp>
#include <set>
#include <string>
#include <unordered_map>

#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/io/sensor/RGBDSensorConfig.h"

// Forward declarations
namespace rs2 {
class config;
}

namespace open3d {
using io::RGBDSensorConfig;
namespace t {
namespace io {

// clang-format off
DECLARE_STRINGIFY_ENUM(rs2_stream);
DECLARE_STRINGIFY_ENUM(rs2_format);
DECLARE_STRINGIFY_ENUM(rs2_l500_visual_preset);
DECLARE_STRINGIFY_ENUM(rs2_rs400_visual_preset);
DECLARE_STRINGIFY_ENUM(rs2_sr300_visual_preset);

// clang-format on

class RealSenseSensorConfig : public RGBDSensorConfig {
public:
    /** Default constructor, default configs will be used:
     * See RealSense documentation for the set of configuration values.
     * Supported configuration options will be specific to a device and other
     * chosen options.
     * https://intelrealsense.github.io/librealsense/doxygen/rs__option_8h.html
     * https://intelrealsense.github.io/librealsense/doxygen/rs__sensor_8h.html
     *  ~~~{.json}
     *  {
     *  // Pick a specific device, leave empty to pick the first available
     *  // device
     *     {"serial": ""},
     *  // pixel format for color frames
     *     {"color_format": "RS2_FORMAT_ANY"},
     *  // (width, height): Leave 0 to let RealSense pick a supported width or
     *  // height
     *     {"color_resolution": "0,0"},
     *  // color stream framerate. Leave 0 to let RealSense pick a supported
     *  // framerate
     *     {"color_fps": "0"},
     *  // pixel format for depth frames
     *     {"depth_format": "RS2_FORMAT_ANY"},
     *  // (width, height): Leave 0 to let RealSense pick a supported width or
     *  // height
     *     {"depth_resolution": "0,0"},
     *  // depth stream framerate. Leave 0 to let RealSense pick a supported
     *  // framerate
     *     {"depth_fps": "0"},
     *  // Controls depth computation on the device. Supported values are
     *  // specific to device family (SR300, RS400, L500). Leave empty to pick
     *  // the default.
     *     {"visual_preset": ""}
     *  }
     *  ~~~
     */
    RealSenseSensorConfig();
    /// Initialize config with a map
    RealSenseSensorConfig(
            const std::unordered_map<std::string, std::string> &config);
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

    void ConvertFromNativeConfig(const rs2::config &rs_config);
    rs2::config ConvertToNativeConfig() const;

    /// Check if the configuration is valid wrt connected devices.
    ///
    /// If a serial number is provided, the configuration will be checked
    /// against that device. The configuration is invalid if the device is not
    /// connected. If no serial number is provided, it will be checked against
    /// any connected device
    bool IsValidConfig() const;

public:
    // To avoid including RealSense or json header, configs is stored in a map
    std::unordered_map<std::string, std::string> config_;
};

/// Store set of valid configuration options for a connected RealSense device.
/// From this structure, a user can construct a RealSenseSensorConfig object
/// meeting their specifications.
struct RealSenseValidConfigs {
    std::string serial;  ///< device serial number
    std::string name;    ///< device name
    /// Mapping between configuraiton option name and a list of valid values
    std::unordered_map<std::string, std::set<std::string>> valid_configs;
};

}  // namespace io
}  // namespace t
}  // namespace open3d
