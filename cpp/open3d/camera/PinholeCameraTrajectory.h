// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "open3d/camera/PinholeCameraParameters.h"

namespace open3d {
namespace camera {

/// \class PinholeCameraTrajectory
///
/// Contains a list of PinholeCameraParameters, useful to storing trajectories.
class PinholeCameraTrajectory : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PinholeCameraTrajectory();
    ~PinholeCameraTrajectory() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// List of PinholeCameraParameters objects.
    std::vector<PinholeCameraParameters> parameters_;
};

}  // namespace camera
}  // namespace open3d
