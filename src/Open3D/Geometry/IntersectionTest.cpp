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

#include <tomasakeninemoeller/opttritri.h>
#include <tomasakeninemoeller/tribox3.h>

namespace open3d {
namespace geometry {

bool IntersectionTest::AABBAABB(const Eigen::Vector3d& min0,
                                const Eigen::Vector3d& max0,
                                const Eigen::Vector3d& min1,
                                const Eigen::Vector3d& max1) {
    if (max0(0) < min1(0) || min0(0) > max1(0)) {
        return false;
    }
    if (max0(1) < min1(1) || min0(1) > max1(1)) {
        return false;
    }
    if (max0(2) < min1(2) || min0(2) > max1(2)) {
        return false;
    }
    return true;
}

bool IntersectionTest::TriangleTriangle3d(const Eigen::Vector3d& p0,
                                          const Eigen::Vector3d& p1,
                                          const Eigen::Vector3d& p2,
                                          const Eigen::Vector3d& q0,
                                          const Eigen::Vector3d& q1,
                                          const Eigen::Vector3d& q2) {
    return NoDivTriTriIsect(const_cast<double*>(p0.data()),
                            const_cast<double*>(p1.data()),
                            const_cast<double*>(p2.data()),
                            const_cast<double*>(q0.data()),
                            const_cast<double*>(q1.data()),
                            const_cast<double*>(q2.data())) != 0;
}

bool IntersectionTest::TriangleAABB(const Eigen::Vector3d& box_center,
                                    const Eigen::Vector3d& box_half_size,
                                    const Eigen::Vector3d& vert0,
                                    const Eigen::Vector3d& vert1,
                                    const Eigen::Vector3d& vert2) {
    double* tri_verts[3] = {const_cast<double*>(vert0.data()),
                            const_cast<double*>(vert1.data()),
                            const_cast<double*>(vert2.data())};
    return triBoxOverlap(const_cast<double*>(box_center.data()),
                         const_cast<double*>(box_half_size.data()),
                         tri_verts) != 0;
}

}  // namespace geometry
}  // namespace open3d
