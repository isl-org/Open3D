// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <atomic>
#include <memory>
#include <string>

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensor.h"
#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensorConfig.h"
#include "Open3D/IO/Sensor/RGBDRecorder.h"

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
