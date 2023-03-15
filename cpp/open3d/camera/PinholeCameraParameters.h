// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "open3d/camera/PinholeCameraIntrinsic.h"

namespace open3d {
namespace camera {

/// \class PinholeCameraParameters
///
/// \brief Contains both intrinsic and extrinsic pinhole camera parameters.
class PinholeCameraParameters : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PinholeCameraParameters();
    ~PinholeCameraParameters() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// PinholeCameraIntrinsic object.
    PinholeCameraIntrinsic intrinsic_;
    /// Camera extrinsic parameters.
    Eigen::Matrix4d_u extrinsic_;
};
}  // namespace camera
}  // namespace open3d
