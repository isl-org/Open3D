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

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensor.h"
#include "Open3D/Geometry/RGBDImage.h"

#include <k4a/k4a.h>
#include <k4arecord/record.h>
#include <memory>

// call k4a_device_close on every failed CHECK
#define CHECK(x, device)                                                     \
    {                                                                        \
        auto retval = (x);                                                   \
        if (retval) {                                                        \
            open3d::utility::LogError("Runtime error: {} returned {}\n", #x, \
                                      retval);                               \
                                                                             \
            k4a_device_close(device);                                        \
            return 1;                                                        \
        }                                                                    \
    }

namespace open3d {
namespace io {

AzureKinectSensor::AzureKinectSensor(
        const AzureKinectSensorConfig& sensor_config)
    : RGBDSensor(), sensor_config_(sensor_config) {}

AzureKinectSensor::~AzureKinectSensor(){};

int AzureKinectSensor::Connect(size_t sensor_index) {
    // Convert to k4a native config
    k4a_device_configuration_t device_config =
            sensor_config_.ConvertToNativeConfig();

    // Check available devices
    const size_t installed_devices =
            static_cast<size_t>(k4a_device_get_installed_count());
    if (sensor_index >= installed_devices) {
        utility::LogError("Device not found.\n");
        return 1;
    }
    if (K4A_FAILED(k4a_device_open(sensor_index, &device_))) {
        utility::LogError("Runtime error: k4a_device_open() failed\n");
        return 1;
    }

    // Get device info
    char serial_number_buffer[256];
    size_t serial_number_buffer_size = sizeof(serial_number_buffer);
    CHECK(k4a_device_get_serialnum(device_, serial_number_buffer,
                                   &serial_number_buffer_size),
          device_);
    k4a_hardware_version_t version_info;
    CHECK(k4a_device_get_version(device_, &version_info), device_);

    utility::LogInfo("Device serial number: {}\n", serial_number_buffer);
    utility::LogInfo("Device version: {};\n",
                     (version_info.firmware_build == K4A_FIRMWARE_BUILD_RELEASE
                              ? "Rel"
                              : "Dbg"));
    utility::LogInfo("C: {}.{}.{};\n", version_info.rgb.major,
                     version_info.rgb.minor, version_info.rgb.iteration);
    utility::LogInfo("D: {}.{}.{}[{}.{}]\n", version_info.depth.major,
                     version_info.depth.minor, version_info.depth.iteration,
                     version_info.depth_sensor.major,
                     version_info.depth_sensor.minor);
    utility::LogInfo("A: {}.{}.{};\n", version_info.audio.major,
                     version_info.audio.minor, version_info.audio.iteration);

    CHECK(k4a_device_start_cameras(device_, &device_config), device_);
    return 0;
}

std::shared_ptr<geometry::RGBDImage> AzureKinectSensor::CaptureFrame() const {
    auto im_rgbd = std::make_shared<geometry::RGBDImage>();
    return im_rgbd;
};

}  // namespace io
}  // namespace open3d
