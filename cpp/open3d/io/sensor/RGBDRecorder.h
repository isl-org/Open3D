// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/geometry/RGBDImage.h"
#include "open3d/io/sensor/RGBDSensorConfig.h"

namespace open3d {
namespace io {

class RGBDRecorder {
public:
    RGBDRecorder() {}
    virtual ~RGBDRecorder() {}

    /// Init recorder, connect to sensor
    virtual bool InitSensor() = 0;

    /// Create recording file
    virtual bool OpenRecord(const std::string &filename) = 0;

    /// Record one frame, return an RGBDImage. If \p write is true, the
    /// RGBDImage frame will be written to file.
    /// If \p enable_align_depth_to_color is true, the depth image will be
    /// warped to align with the color image; otherwise the raw depth image
    /// output will be saved. Setting \p enable_align_depth_to_color to
    /// false is useful when recording at high resolution with high frame rates.
    /// In this case, the depth image must be warped to align with the color
    /// image with when reading from the recorded file.
    virtual std::shared_ptr<geometry::RGBDImage> RecordFrame(
            bool write, bool enable_align_depth_to_color) = 0;

    /// Flush data to recording file and disconnect from sensor
    virtual bool CloseRecord() = 0;
};

}  // namespace io
}  // namespace open3d
