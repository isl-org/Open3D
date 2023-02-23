// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "open3d/io/sensor/RGBDRecorder.h"
#include "open3d/io/sensor/azure_kinect/AzureKinectSensor.h"
#include "open3d/io/sensor/azure_kinect/AzureKinectSensorConfig.h"

struct _k4a_record_t;  // typedef _k4a_record_t* k4a_record_t;

namespace open3d {

namespace geometry {
class RGBDImage;
class Image;
}  // namespace geometry

namespace io {

/// \class AzureKinectRecorder
///
/// AzureKinect recorder.
class AzureKinectRecorder : public RGBDRecorder {
public:
    AzureKinectRecorder(const AzureKinectSensorConfig& sensor_config,
                        size_t sensor_index);
    ~AzureKinectRecorder() override;

    /// Initialize sensor.
    bool InitSensor() override;
    /// Attempt to create and open an mkv file.
    ///
    /// \param filename Path to the mkv file.
    bool OpenRecord(const std::string& filename) override;
    /// Close the recorded mkv file.
    bool CloseRecord() override;
    /// Record a frame to mkv if flag is on and return an RGBD object.
    ///
    /// \param write Enable recording to mkv file.
    /// \param enable_align_depth_to_color Enable aligning WFOV depth image to
    /// the color image in visualizer.
    std::shared_ptr<geometry::RGBDImage> RecordFrame(
            bool write, bool enable_align_depth_to_color) override;

    /// Check if the mkv file is created.
    bool IsRecordCreated() { return is_record_created_; }

protected:
    AzureKinectSensor sensor_;
    _k4a_record_t* recording_;
    size_t device_index_;

    bool is_record_created_ = false;
};

}  // namespace io
}  // namespace open3d
