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

#pragma once

#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "open3d/core/Dtype.h"
#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/io/sensor/RGBDSensorConfig.h"
#include "open3d/t/io/sensor/RGBDVideoMetadata.h"

// Forward declarations
namespace rs2 {
class config;
class pipeline_profile;
}  // namespace rs2

namespace open3d {
using io::RGBDSensorConfig;
namespace t {
namespace io {

/// Configuration for a RealSense camera
///
/// See RealSense documentation for the set of configuration values.
/// Supported configuration options will be specific to a device and other
/// chosen options.
/// https://intelrealsense.github.io/librealsense/doxygen/rs__option_8h.html
/// https://intelrealsense.github.io/librealsense/doxygen/rs__sensor_8h.html
///  ~~~{.json}
///  {
///  // Pick a specific device, leave empty to pick the first available
///  // device
///     {"serial": ""},
///  // pixel format for color frames
///     {"color_format": "RS2_FORMAT_ANY"},
///  // (width, height): Leave 0 to let RealSense pick a supported width or
///  // height
///     {"color_resolution": "0,0"},
///  // pixel format for depth frames
///     {"depth_format": "RS2_FORMAT_ANY"},
///  // (width, height): Leave 0 to let RealSense pick a supported width or
///  // height
///     {"depth_resolution": "0,0"},
///  // framerate for both color and depth streams. Leave 0 to let RealSense
///  // pick a supported framerate
///     {"fps": "0"},
///  // Controls depth computation on the device. Supported values are
///  // specific to device family (SR300, RS400, L500). Leave empty to pick
///  // the default.
///     {"visual_preset": ""}
///  }
///  ~~~
class RealSenseSensorConfig : public RGBDSensorConfig {
public:
    /// Default constructor, default configs will be used
    RealSenseSensorConfig();
    /// Initialize config with a map
    RealSenseSensorConfig(
            const std::unordered_map<std::string, std::string> &config);
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

    /// Convert to RealSense config
    rs2::config ConvertToNativeConfig() const;

    /// Get metadata for a streaming RealSense camera or bag file
    static Json::Value GetMetadataJson(const rs2::pipeline_profile &profile);

    /// Get pixel data types for color and depth streams. These will be set in
    /// metadata.color_dt_, metadata.color_channels_ and metadata.depth_dt_
    static void GetPixelDtypes(const rs2::pipeline_profile &profile,
                               RGBDVideoMetadata &metadata);

public:
    /// Convert rs2_format enum to Open3D Dtype and number of channels
    /// \param rs2_format_enum An int is accepted instead of rs2_format enum
    /// to prevent dependence on the realsense headers.
    static std::pair<core::Dtype, uint8_t> get_dtype_channels(
            int rs2_format_enum);

    // To avoid including RealSense or json header, config is stored in a
    // map
    std::unordered_map<std::string, std::string> config_;
};

/// Store set of valid configuration options for a connected RealSense
/// device. From this structure, a user can construct a
/// RealSenseSensorConfig object meeting their specifications.
struct RealSenseValidConfigs {
    std::string serial;  ///< Device serial number.
    std::string name;    ///< Device name.
    /// Mapping between configuration option name and a list of valid
    /// values.
    std::unordered_map<std::string, std::set<std::string>> valid_configs;
};

}  // namespace io
}  // namespace t
}  // namespace open3d
