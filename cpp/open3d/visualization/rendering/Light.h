// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>
#include <cmath>

namespace open3d {
namespace visualization {
namespace rendering {

struct Light {
    enum eLightType { POINT, SPOT, DIRECTIONAL };

    // common light parameters
    Eigen::Vector3f color = Eigen::Vector3f(1.f, 1.f, 1.f);
    Eigen::Vector3f position = Eigen::Vector3f(0.f, 0.f, 0.f);
    eLightType type = POINT;
    float intensity = 10000.f;
    float falloff = 10.f;
    bool cast_shadows = false;

    Eigen::Vector3f direction = Eigen::Vector3f(0.f, 0.f, -1.f);

    // Spot lights parameters
    float light_cone_inner = float(M_PI / 4.0);
    float light_cone_outer = float(M_PI / 2.0);
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
