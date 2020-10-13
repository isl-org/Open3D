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

#include "open3d/geometry/IntersectionTest.h"

#include <tomasakeninemoeller/opttritri.h>
#include <tomasakeninemoeller/tribox3.h>

#include <cmath>

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
    const Eigen::Vector3d mu = (p0 + p1 + p2 + q0 + q1 + q2) / 6;
    const Eigen::Vector3d sigma =
            (((p0 - mu).array().square() + (p1 - mu).array().square() +
              (p2 - mu).array().square() + (q0 - mu).array().square() +
              (q1 - mu).array().square() + (q2 - mu).array().square()) /
             5)
                    .sqrt() +
            1e-12;
    Eigen::Vector3d p0m = (p0 - mu).array() / sigma.array();
    Eigen::Vector3d p1m = (p1 - mu).array() / sigma.array();
    Eigen::Vector3d p2m = (p2 - mu).array() / sigma.array();
    Eigen::Vector3d q0m = (q0 - mu).array() / sigma.array();
    Eigen::Vector3d q1m = (q1 - mu).array() / sigma.array();
    Eigen::Vector3d q2m = (q2 - mu).array() / sigma.array();
    return NoDivTriTriIsect(p0m.data(), p1m.data(), p2m.data(), q0m.data(),
                            q1m.data(), q2m.data()) != 0;
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

bool IntersectionTest::PointsCoplanar(const Eigen::Vector3d& p0,
                                      const Eigen::Vector3d& p1,
                                      const Eigen::Vector3d& p2,
                                      const Eigen::Vector3d& p3) {
    return (p1 - p0).dot((p2 - p0).cross(p3 - p0)) < 1e-12;
}

double IntersectionTest::LinesMinimumDistance(const Eigen::Vector3d& p1,
                                              const Eigen::Vector3d& p2,
                                              const Eigen::Vector3d& p3,
                                              const Eigen::Vector3d& p4) {
    constexpr double EPS = 1e-12;
    Eigen::Vector3d p13 = p1 - p3;
    Eigen::Vector3d p21 = p2 - p1;
    // not a line, but a single point
    if (std::abs(p21(0)) < EPS && std::abs(p21(1)) < EPS &&
        std::abs(p21(2)) < EPS) {
        return -1;
    }
    Eigen::Vector3d p43 = p4 - p3;
    // not a line, but a single point
    if (std::abs(p43(0)) < EPS && std::abs(p43(1)) < EPS &&
        std::abs(p43(2)) < EPS) {
        return -2;
    }

    double d1343 = p13.dot(p43);
    double d4321 = p43.dot(p21);
    double d1321 = p13.dot(p21);
    double d4343 = p43.dot(p43);
    double d2121 = p21.dot(p21);

    double denominator = d2121 * d4343 - d4321 * d4321;
    // lines are parallel
    if (std::abs(denominator) < EPS) {
        return -3;
    }
    double numerator = d1343 * d4321 - d1321 * d4343;

    double mua = numerator / denominator;
    double mub = (d1343 + d4321 * mua) / d4343;

    Eigen::Vector3d pa = p1 + mua * p21;
    Eigen::Vector3d pb = p3 + mub * p43;
    double dist = (pa - pb).norm();
    return dist;
}

double IntersectionTest::LineSegmentsMinimumDistance(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& q0,
        const Eigen::Vector3d& q1) {
    const Eigen::Vector3d p10 = p1 - p0;
    const Eigen::Vector3d q10 = q1 - q0;
    const Eigen::Vector3d p0q0 = p0 - q0;
    double a = p10.dot(p10);
    double b = p10.dot(q10);
    double c = q10.dot(q10);
    double d = p10.dot(p0q0);
    double e = q10.dot(p0q0);
    // double f = p0q0.dot(p0q0);
    double det = a * c - b * b;
    double s, t;
    if (det > 0)  // non parallel segments
    {
        double bte = b * e;
        double ctd = c * d;
        if (bte <= ctd) {
            if (e <= 0) {
                s = (-d >= a ? 1 : (-d > 0 ? -d / a : 0));
                t = 0;
            } else if (e < c) {
                s = 0;
                t = e / c;
            } else {
                s = (b - d >= a ? 1 : (b - d > 0 ? (b - d) / a : 0));
                t = 1;
            }
        } else {
            s = bte - ctd;
            if (s >= det) {
                if (b + e <= 0) {
                    s = (-d <= 0 ? 0 : (-d < a ? -d / a : 1));
                    t = 0;
                } else if (b + e < c) {
                    s = 1;
                    t = (b + e) / c;
                } else {
                    s = (b - d <= 0 ? 0 : (b - d < a ? (b - d) / a : 1));
                    t = 1;
                }
            } else {
                double ate = a * e;
                double btd = b * d;
                if (ate <= btd) {
                    s = (-d <= 0 ? 0 : (-d >= a ? 1 : -d / a));
                    t = 0;
                } else {
                    t = ate - btd;
                    if (t >= det) {
                        s = (b - d <= 0 ? 0 : (b - d >= a ? 1 : (b - d) / a));
                        t = 1;
                    } else {
                        s /= det;
                        t /= det;
                    }
                }
            }
        }
    } else  // parallel segments
    {
        if (e <= 0) {
            s = (-d <= 0 ? 0 : (-d >= a ? 1 : -d / a));
            t = 0;
        } else if (e >= c) {
            s = (b - d <= 0 ? 0 : (b - d >= a ? 1 : (b - d) / a));
            t = 1;
        } else {
            s = 0;
            t = e / c;
        }
    }

    Eigen::Vector3d p = (1 - s) * p0 + s * p1;
    Eigen::Vector3d q = (1 - t) * q0 + t * q1;
    double dist = (p - q).norm();
    return dist;
}

}  // namespace geometry
}  // namespace open3d
