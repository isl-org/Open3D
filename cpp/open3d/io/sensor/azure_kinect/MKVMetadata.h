// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/utility/IJsonConvertible.h"

enum class SensorType { AZURE_KINECT = 0, REAL_SENSE = 1 };

namespace open3d {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace io {

/// class MKVMetadata
///
/// AzureKinect mkv metadata.
class MKVMetadata : public utility::IJsonConvertible {
public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// \brief Shared intrinsics between RGB & depth.
    ///
    /// We assume depth image is always warped to the color image system.
    camera::PinholeCameraIntrinsic intrinsics_;

    std::string serial_number_ = "";
    /// Length of the video (usec).
    uint64_t stream_length_usec_ = 0;
    /// Width of the video.
    int width_;
    /// Height of the video.
    int height_;
    std::string color_mode_;
    std::string depth_mode_;
};

}  // namespace io
}  // namespace open3d
