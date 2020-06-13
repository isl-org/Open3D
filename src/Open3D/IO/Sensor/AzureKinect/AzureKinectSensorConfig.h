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

#include <string>
#include <unordered_map>

#include "Open3D/IO/Sensor/RGBDSensorConfig.h"
#include "Open3D/Utility/IJsonConvertible.h"

struct _k4a_device_configuration_t;  // Alias of k4a_device_configuration_t

namespace open3d {
namespace io {

// Alternative implementation of _k4a_device_configuration_t with string values

/// \class AzureKinectSensorConfig
///
/// AzureKinect sensor configuration.
class AzureKinectSensorConfig : public RGBDSensorConfig {
public:
    /// Default constructor, default configs will be used
    AzureKinectSensorConfig();
    /// Initialize config with a map
    AzureKinectSensorConfig(
            const std::unordered_map<std::string, std::string> &config);
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    void ConvertFromNativeConfig(const _k4a_device_configuration_t &k4a_config);
    _k4a_device_configuration_t ConvertToNativeConfig() const;

public:
    // To avoid including k4a or json header, configs is stored in a map
    std::unordered_map<std::string, std::string> config_;

protected:
    static bool IsValidConfig(
            const std::unordered_map<std::string, std::string> &config,
            bool verbose = true);
};

}  // namespace io
}  // namespace open3d
