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

#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/io/sensor/RGBDSensor.h"
#include "open3d/t/io/sensor/RGBDVideoMetadata.h"
#include "open3d/t/io/sensor/realsense/RealSenseSensorConfig.h"

namespace rs2 {
class pipeline;
class align;
class config;
}  // namespace rs2

namespace open3d {
namespace t {
namespace io {

/// RealSense camera discovery, configuration, streaming and recording
class RealSenseSensor : public RGBDSensor {
public:
    /// Query all connected RealSense cameras for their capabilities.
    ///
    /// \return A list of devices and their supported capabilities.
    static std::vector<RealSenseValidConfigs> EnumerateDevices();
    /// List all RealSense cameras connected to the system along with their
    /// capabilities. Use this listing to select an appropriate configuration
    /// for a camera.
    static bool ListDevices();

    /// Default constructor. Initialize with default settings.
    RealSenseSensor();
    RealSenseSensor(const RealSenseSensor &) = delete;
    RealSenseSensor &operator=(const RealSenseSensor &) = delete;
    virtual ~RealSenseSensor() override;

    /// Initialize sensor (optional).
    ///
    /// Configure sensor with custom settings. If this is skipped, default
    /// settings will be used. You can enable recording to a bag file by
    /// specifying a filename.
    /// \param sensor_config Camera configuration, such as resolution and
    /// framerate. A serial number can be entered here to connect to a specific
    /// camera.
    /// \param sensor_index Connect to a camera at this position in the
    /// enumeration of RealSense cameras that are currently connected. Use
    /// EnumerateDevices() or ListDevices() to obtain a list of connected
    /// cameras. This is ignored if \p sensor_config contains a serial entry.
    /// \param filename Save frames to a bag file
    /// \return true if a camera was found and initialized with the given
    /// settings, else false.
    virtual bool InitSensor(const RealSenseSensorConfig &sensor_config =
                                    RealSenseSensorConfig{},
                            size_t sensor_index = 0,
                            const std::string &filename = "");
    virtual bool InitSensor(const RGBDSensorConfig &sensor_config,
                            size_t sensor_index = 0,
                            const std::string &filename = "") override {
        return InitSensor(
                dynamic_cast<const RealSenseSensorConfig &>(sensor_config),
                sensor_index, filename);
    }

    /// Start capturing synchronized depth and color frames.
    /// \param start_record start recording to the specified bag file as well.
    virtual bool StartCapture(bool start_record = false) override;

    /// Pause recording to the bag file.
    ///
    /// \warning If this is called immediately after \c StartCapture(), the
    /// bag file may have an incorrect end time.
    virtual void PauseRecord() override;

    /// Resume recording to the bag file. The file will contain discontinuous
    /// segments.
    virtual void ResumeRecord() override;

    ///  Acquire the next synchronized RGBD frameset from the camera.
    ///
    /// \param wait If true wait for the next frame set, else return immediately
    /// with an empty RGBDImage if it is not yet available.
    /// \param align_depth_to_color Enable aligning WFOV depth image to
    /// the color image in visualizer.
    virtual geometry::RGBDImage CaptureFrame(
            bool wait = true, bool align_depth_to_color = true) override;

    /// Get current timestamp (in us)
    ///
    /// See
    /// https://intelrealsense.github.io/librealsense/doxygen/classrs2_1_1frame.html#a25f71d45193f2f4d77960320276b83f1
    /// for more details.
    virtual uint64_t GetTimestamp() const override { return timestamp_; }

    /// Stop capturing frames.
    virtual void StopCapture() override;

    /// Get metadata of the RealSense video capture.
    virtual const RGBDVideoMetadata &GetMetadata() const override {
        return metadata_;
    }

    /// Get filename being written.
    virtual std::string GetFilename() const override { return filename_; };

    /// Text Description.
    using RGBDSensor::ToString;

private:
    bool enable_recording_ = false;
    bool is_recording_ = false;
    bool is_capturing_ = false;
    std::string filename_;
    geometry::RGBDImage current_frame_;
    uint64_t timestamp_ = 0;
    RGBDVideoMetadata metadata_;
    std::unique_ptr<rs2::pipeline> pipe_;
    std::unique_ptr<rs2::align> align_to_color_;
    std::unique_ptr<rs2::config> rs_config_;

    static const uint64_t MILLISEC_TO_MICROSEC = 1000;
};

}  // namespace io
}  // namespace t
}  // namespace open3d
