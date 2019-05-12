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

#include "Open3D/Geometry/IntersectionTest.h"

#include <tritriintersect/tri_tri_intersect.h>

namespace open3d {
namespace geometry {

bool IntersectingTriangleTriangle3d(const Eigen::Vector3d& p0,
                                    const Eigen::Vector3d& p1,
                                    const Eigen::Vector3d& p2,
                                    const Eigen::Vector3d& q0,
                                    const Eigen::Vector3d& q1,
                                    const Eigen::Vector3d& q2) {
    return tri_tri_overlap_test_3d(
            const_cast<double*>(p0.data()), const_cast<double*>(p1.data()),
            const_cast<double*>(p2.data()), const_cast<double*>(q0.data()),
            const_cast<double*>(q1.data()), const_cast<double*>(q2.data()));
}

}  // namespace geometry
}  // namespace open3d
