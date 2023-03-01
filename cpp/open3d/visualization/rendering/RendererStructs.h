// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <json/json.h>

#include <Eigen/Geometry>

#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {

namespace geometry {
class Geometry3D;
}

namespace visualization {
namespace rendering {

struct LightDescription {
    enum eLightType { POINT, SPOT, DIRECTIONAL };

    eLightType type;
    float intensity;
    float falloff;
    // Spot lights only
    float light_cone_inner;
    // Spot lights only
    float light_cone_outer;
    Eigen::Vector3f color;
    Eigen::Vector3f direction;
    Eigen::Vector3f position;
    bool cast_shadows;

    Json::Value custom_attributes;

    LightDescription()
        : type(POINT),
          intensity(10000),
          falloff(10),
          light_cone_inner(float(M_PI / 4.0)),
          light_cone_outer(float(M_PI / 2.0)),
          color(1.f, 1.f, 1.f),
          direction(0.f, 0.f, -1.f),
          position(0.f, 0.f, 0.f),
          cast_shadows(true) {}
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
