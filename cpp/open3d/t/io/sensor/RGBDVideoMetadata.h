// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/core/Dtype.h"
#include "open3d/utility/IJsonConvertible.h"

namespace open3d {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace t {
namespace io {

enum class SensorType { AZURE_KINECT = 0, REAL_SENSE = 1 };

/// RGBD video metadata.
class RGBDVideoMetadata : public utility::IJsonConvertible {
public:
    bool ConvertToJsonValue(Json::Value &value) const override;

    bool ConvertFromJsonValue(const Json::Value &value) override;

    /// Text description
    using utility::IJsonConvertible::ToString;

public:
    /// \brief Shared intrinsics between RGB & depth.
    ///
    /// We assume depth image is always warped to the color image system.
    camera::PinholeCameraIntrinsic intrinsics_;

    /// Capture device name.
    std::string device_name_ = "";

    /// Capture device serial number.
    std::string serial_number_ = "";

    /// Length of the video (usec). 0 for live capture.
    uint64_t stream_length_usec_ = 0;

    /// Width of the video frame.
    int width_;

    /// Height of the video frame.
    int height_;

    /// Frame rate.
    //
    /// We assume both color and depth streams have the same frame rate.
    double fps_;

    /// Pixel format for color data.
    std::string color_format_;

    /// Pixel format for depth data.
    std::string depth_format_;

    /// Pixel Dtype for color data.
    core::Dtype color_dt_;

    /// Pixel Dtype for depth data.
    core::Dtype depth_dt_;

    /// Number of color channels.
    uint8_t color_channels_;

    /// Number of depth units per meter (depth in m =
    /// depth_pixel_value/depth_scale).
    double depth_scale_;
};

}  // namespace io
}  // namespace t
}  // namespace open3d
