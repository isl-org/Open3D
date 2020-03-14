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

#include <memory>

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensorConfig.h"
#include "Open3D/IO/Sensor/RGBDSensor.h"

struct _k4a_capture_t;         // typedef _k4a_capture_t* k4a_capture_t;
struct _k4a_device_t;          // typedef _k4a_device_t* k4a_device_t;
struct _k4a_transformation_t;  // typedef _k4a_transformation_t*
                               // k4a_transformation_t;

namespace open3d {
namespace geometry {
class RGBDImage;
class Image;
}  // namespace geometry

namespace io {

// Avoid including AzureKinectRecorder.h
class AzureKinectRecorder;

/// \class AzureKinectSensor
///
/// AzureKinect sensor.
class AzureKinectSensor : public RGBDSensor {
public:
    /// \brief Default Constructor.
    AzureKinectSensor(const AzureKinectSensorConfig& sensor_config);
    ~AzureKinectSensor();

    bool Connect(size_t sensor_index) override;
    std::shared_ptr<geometry::RGBDImage> CaptureFrame(
            bool enable_align_depth_to_color) const override;

    static bool PrintFirmware(_k4a_device_t* device);
    /// List available Azure Kinect devices.
    static bool ListDevices();
    static std::shared_ptr<geometry::RGBDImage> DecompressCapture(
            _k4a_capture_t* capture, _k4a_transformation_t* transformation);

protected:
    _k4a_capture_t* CaptureRawFrame() const;

    AzureKinectSensorConfig sensor_config_;
    _k4a_transformation_t* transform_depth_to_color_;
    _k4a_device_t* device_;
    int timeout_;

    friend class AzureKinectRecorder;
};

}  // namespace io
}  // namespace open3d
