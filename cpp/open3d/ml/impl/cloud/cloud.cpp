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

#include "open3d/ml/impl/cloud/cloud.h"

namespace open3d {
namespace ml {
namespace impl {

// Getters
// *******

PointXYZ max_point(std::vector<PointXYZ> points) {
    // Initiate limits
    PointXYZ maxP(points[0]);

    // Loop over all points
    for (auto p : points) {
        if (p.x > maxP.x) maxP.x = p.x;

        if (p.y > maxP.y) maxP.y = p.y;

        if (p.z > maxP.z) maxP.z = p.z;
    }

    return maxP;
}

PointXYZ min_point(std::vector<PointXYZ> points) {
    // Initiate limits
    PointXYZ minP(points[0]);

    // Loop over all points
    for (auto p : points) {
        if (p.x < minP.x) minP.x = p.x;

        if (p.y < minP.y) minP.y = p.y;

        if (p.z < minP.z) minP.z = p.z;
    }

    return minP;
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
