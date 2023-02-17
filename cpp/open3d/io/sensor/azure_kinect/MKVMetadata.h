// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
