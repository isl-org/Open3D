// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "open3d/io/sensor/RGBDSensorConfig.h"

namespace open3d {

namespace geometry {
class RGBDImage;
};

namespace io {

class RGBDSensor {
public:
    RGBDSensor() {}
    virtual bool Connect(size_t sensor_index) = 0;
    virtual ~RGBDSensor(){};

    /// Capture one frame, return an RGBDImage.
    /// If \p enable_align_depth_to_color is true, the depth image will be
    /// warped to align with the color image; otherwise the raw depth image
    /// output will be saved. Setting \p enable_align_depth_to_color to
    /// false is useful when capturing at high resolution with high frame rates.
    virtual std::shared_ptr<geometry::RGBDImage> CaptureFrame(
            bool enable_align_depth_to_color) const = 0;
};

}  // namespace io
}  // namespace open3d
