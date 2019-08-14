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

#include "Python/io/io.h"

#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensor.h"
#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensorConfig.h"

using namespace open3d;

void pybind_sensor(py::module &m) {
    // TODO: use Trampoline base class
    py::class_<io::AzureKinectSensorConfig> azure_kinect_sensor_config(
            m, "AzureKinectSensorConfig", "AzureKinect sensor configuration.");
    py::detail::bind_default_constructor<io::AzureKinectSensorConfig>(
            azure_kinect_sensor_config);
    azure_kinect_sensor_config.def(
            py::init([](const std::unordered_map<std::string, std::string>
                                &config) {
                return new io::AzureKinectSensorConfig(config);
            }),
            "config"_a);

    // TODO: use Trampoline base class
    py::class_<io::AzureKinectSensor> azure_kinect_sensor(
            m, "AzureKinectSensor", "AzureKinect sensor.");

    azure_kinect_sensor.def(
            py::init([](const io::AzureKinectSensorConfig &sensor_config) {
                return new io::AzureKinectSensor(sensor_config);
            }),
            "sensor_config"_a);
    azure_kinect_sensor
            .def("connect", &io::AzureKinectSensor::Connect, "sensor_index"_a,
                 "Connect to specified device.")
            .def("capture_frame", &io::AzureKinectSensor::CaptureFrame,
                 "enable_align_depth_to_color"_a, "Capture an RGBD frame.");
}
