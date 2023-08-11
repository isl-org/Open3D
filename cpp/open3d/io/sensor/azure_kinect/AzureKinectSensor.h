// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "open3d/io/sensor/RGBDSensor.h"
#include "open3d/io/sensor/azure_kinect/AzureKinectSensorConfig.h"

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
    void Disconnect();
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
