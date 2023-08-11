// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/io/sensor/RGBDSensorConfig.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/io/sensor/RGBDVideoMetadata.h"

namespace open3d {
using io::RGBDSensorConfig;
namespace t {
namespace io {

/// Interface class for control of RGBD cameras.
class RGBDSensor {
public:
    /// Default constructor. Initialize with default settings.
    RGBDSensor() {}
    virtual ~RGBDSensor() {}

    /// Initialize sensor (optional).
    ///
    /// Configure sensor with custom settings. If this is skipped, default
    /// settings will be used. You can enable recording to a file by
    /// specifying a filename.
    /// \param sensor_config Camera configuration, such as resolution and
    /// framerate. A serial number can be entered here to connect to a specific
    /// camera.
    /// \param sensor_index Connect to a camera at this position in the
    /// enumeration of RealSense cameras that are currently connected. Use
    /// EnumerateDevices() or ListDevices() to obtain a list of connected
    /// cameras and their capabilities. This is ignored if \p sensor_config
    /// contains a "serial" entry.
    /// \param filename Save frames to this file.
    /// \return true if a camera was found and initialized with the given
    /// settings, else false.
    virtual bool InitSensor(const RGBDSensorConfig &sensor_config,
                            size_t sensor_index = 0,
                            const std::string &filename = "") = 0;

    /// Start capturing synchronized depth and color frames.
    /// \param start_record start recording to the specified file as well.
    virtual bool StartCapture(bool start_record = false) = 0;

    /// Pause recording to the file.
    virtual void PauseRecord() = 0;

    /// Resume recording to the file. The file will contain discontinuous
    /// segments.
    virtual void ResumeRecord() = 0;

    /// Acquire the next synchronized RGBD frameset from the camera.
    ///
    /// \param align_depth_to_color Enable aligning WFOV depth image to
    /// the color image in visualizer.
    /// \param wait If true wait for the next frame set, else return immediately
    /// with an empty RGBDImage if it is not yet available
    virtual geometry::RGBDImage CaptureFrame(
            bool wait = true, bool align_depth_to_color = true) = 0;

    /// Get current timestamp (in us).
    virtual uint64_t GetTimestamp() const = 0;

    /// Stop capturing frames.
    virtual void StopCapture() = 0;

    /// Get const reference to the metadata of the RGBD video capture.
    virtual const RGBDVideoMetadata &GetMetadata() const = 0;

    /// Get filename being written.
    virtual std::string GetFilename() const = 0;

    /// Text Description.
    virtual const std::string ToString() const {
        return fmt::format(
                "RGBD sensor {} (serial {}){}{}", GetMetadata().device_name_,
                GetMetadata().serial_number_,
                GetTimestamp() == 0 ? ""
                                    : "\nCapturing video: timestamp (us) = " +
                                              std::to_string(GetTimestamp()),
                GetFilename().empty() ? ""
                                      : "\nRecording to file " + GetFilename());
    }
};

}  // namespace io
}  // namespace t
}  // namespace open3d
