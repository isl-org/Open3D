// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
