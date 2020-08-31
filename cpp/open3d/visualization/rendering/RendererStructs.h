// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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
