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
#include <k4a/k4atypes.h>

namespace open3d {
namespace io {

bool AzureKinectSensorConfig::ConvertToJsonValue(Json::Value &value) const {
    return true;
}

bool AzureKinectSensorConfig::ConvertFromJsonValue(const Json::Value &value) {
    return true;
}

void AzureKinectSensorConfig::ConvertFromNativeConfig(
        const _k4a_device_configuration_t &config) {}

_k4a_device_configuration_t AzureKinectSensorConfig::ConvertToK4AConfig() {
    _k4a_device_configuration_t k4a_config;
    return k4a_config;
}

}  // namespace io
}  // namespace open3d
