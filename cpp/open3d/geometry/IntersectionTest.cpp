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

#include <cmath>
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

double IntersectionTest::LineAABBSlabParam(
        const Eigen::ParametrizedLine<double, 3>& line,
        const AxisAlignedBoundingBox& box,
        double dir_x_inv, double dir_y_inv, double dir_z_inv) {
    /* This check is based off of Tavian Barnes' branchless implementation of
     * the slab method for determining ray/AABB intersections. It returns the
     * distance from the line origin to the intersection point with the AABB,
     * or NaN if there is no intersection.
     * https://tavianator.com/2011/ray_box.html */

    double t_x0 = dir_x_inv * (box.min_bound_.x() - line.origin().x());
    double t_x1 = dir_x_inv * (box.max_bound_.x() - line.origin().x());
    double t_min = std::min(t_x0, t_x1);
    double t_max = std::max(t_x0, t_x1);

    double t_y0 = dir_y_inv * (box.min_bound_.y() - line.origin().y());
    double t_y1 = dir_y_inv * (box.max_bound_.y() - line.origin().y());
    t_min = std::max(t_min, std::min(t_y0, t_y1));
    t_max = std::min(t_max, std::max(t_y0, t_y1));

    double t_z0 = dir_z_inv * (box.min_bound_.z() - line.origin().z());
    double t_z1 = dir_z_inv * (box.max_bound_.z() - line.origin().z());
    t_min = std::max(t_min, std::min(t_z0, t_z1));
    t_max = std::min(t_max, std::max(t_z0, t_z1));

    if (t_max >= t_min)
        return t_min;
    return std::nan("");
}

double IntersectionTest::LineAABBSlabParam(
        const Eigen::ParametrizedLine<double, 3>& line,
        const AxisAlignedBoundingBox& box) {
    double x_inv = 1.0 / line.direction().x();
    double y_inv = 1.0 / line.direction().y();
    double z_inv = 1.0 / line.direction().z();
    return LineAABBSlabParam(line, box, x_inv, y_inv, z_inv);
}

double IntersectionTest::RayAABBSlabParam(
        const Eigen::ParametrizedLine<double, 3>& ray,
        const AxisAlignedBoundingBox& box) {
    double x_inv = 1.0 / ray.direction().x();
    double y_inv = 1.0 / ray.direction().y();
    double z_inv = 1.0 / ray.direction().z();
    return RayAABBSlabParam(ray, box, x_inv, y_inv, z_inv);
}

double IntersectionTest::RayAABBSlabParam(
        const Eigen::ParametrizedLine<double, 3>& ray,
        const AxisAlignedBoundingBox& box,
        double dir_x_inv, double dir_y_inv, double dir_z_inv) {
    /* This check is based off of Tavian Barnes' branchless implementation of
     * the slab method for determining ray/AABB intersections. It returns the
     * distance from the line origin to the intersection point with the AABB,
     * or NaN if there is no intersection.
     * https://tavianator.com/2011/ray_box.html */

    double t_x0 = dir_x_inv * (box.min_bound_.x() - ray.origin().x());
    double t_x1 = dir_x_inv * (box.max_bound_.x() - ray.origin().x());
    double t_min = std::min(t_x0, t_x1);
    double t_max = std::max(t_x0, t_x1);

    double t_y0 = dir_y_inv * (box.min_bound_.y() - ray.origin().y());
    double t_y1 = dir_y_inv * (box.max_bound_.y() - ray.origin().y());
    t_min = std::max(t_min, std::min(t_y0, t_y1));
    t_max = std::min(t_max, std::max(t_y0, t_y1));

    double t_z0 = dir_z_inv * (box.min_bound_.z() - ray.origin().z());
    double t_z1 = dir_z_inv * (box.max_bound_.z() - ray.origin().z());
    t_min = std::max(t_min, std::min(t_z0, t_z1));
    t_max = std::min(t_max, std::max(t_z0, t_z1));

    t_min = std::max(0., t_min);
    if (t_max >= t_min)
        return t_min;
    return std::nan("");
}

double IntersectionTest::LineAABBExactParam(
        const Eigen::ParametrizedLine<double, 3>& line,
        const AxisAlignedBoundingBox& box) {
    using namespace Eigen;
    /* This is a naive, exact method of computing the intersection with a
     * bounding box.  It is much slower than the highly optimized slab method,
     * but will perform correctly in the one case where the slab method
     * degenerates: when a ray lies exactly within one of the bounding planes.
     * If your problem is structured such that the slab method is likely to
     * encounter a degenerate scenario, AND you need an exact solution that can
     * not allow the occasional non-intersection, AND you care about maximal
     * performance, consider implementing a special check which takes advantage
     * of the reduced dimensionality of your problem.
     */

    // When running the stress test in examples/LineToAABB.cpp about 1% to 2%
    // of the randomly generated cases will fail when using this method due to
    // the round-trip vector coming back from the ParameterizedLine's
    // intersectionParameter method being off in the 11th or greater decimal
    // position from the original plane point. This tolerance seems to
    // eliminate the issue.
    double tol = 1e-10;
    AxisAlignedBoundingBox b_tol{box.min_bound_ - Vector3d(tol, tol, tol),
                                 box.max_bound_ + Vector3d(tol, tol, tol)};

    using plane_t = Eigen::Hyperplane<double, 3>;
    std::array<plane_t, 6> planes {{{{-1, 0, 0}, box.min_bound_},
                                           {{1, 0, 0}, box.max_bound_},
                                           {{0, -1, 0}, box.min_bound_},
                                           {{0, 1, 0}, box.max_bound_},
                                           {{0, 0, -1}, box.min_bound_},
                                           {{0, 0, 1}, box.max_bound_}}};

    // Get the intersections
    std::vector<double> parameters;
    std::vector<Eigen::Vector3d> points;

    for (int i = 0; i < 6; ++i) {
        double t =  line.intersectionParameter(planes[i]);
        if (!std::isinf(t)) {
            parameters.push_back(t);
            auto p = line.pointAt(t);
            points.push_back(p);
        }
    }

    // Find the ones which are contained
    auto contained_indices = b_tol.GetPointIndicesWithinBoundingBox(points);
    if (contained_indices.empty())
        return std::nan("");

    // Return the lowest parameter
    double minimum = parameters[contained_indices[0]];
    for (auto i : contained_indices) {
        minimum = std::min(minimum, parameters[i]);
    }
    return minimum;
}

double IntersectionTest::RayAABBExactParam(
        const Eigen::ParametrizedLine<double, 3>& ray,
        const AxisAlignedBoundingBox& box) {
    using namespace Eigen;
    /* This is a naive, exact method of computing the intersection with a
     * bounding box.  It is much slower than the highly optimized slab method,
     * but will perform correctly in the one case where the slab method
     * degenerates: when a ray lies exactly within one of the bounding planes.
     * If your problem is structured such that the slab method is likely to
     * encounter a degenerate scenario, AND you need an exact solution that can
     * not allow the occasional non-intersection, AND you care about maximal
     * performance, consider implementing a special check which takes advantage
     * of the reduced dimensionality of your problem.
     */

    // When running the stress test in examples/LineToAABB.cpp about 1% to 2%
    // of the randomly generated cases will fail when using this method due to
    // the round-trip vector coming back from the ParameterizedLine's
    // intersectionParameter method being off in the 11th or greater decimal
    // position from the original plane point. This tolerance seems to
    // eliminate the issue.
    double tol = 1e-10;
    AxisAlignedBoundingBox b_tol{box.min_bound_ - Vector3d(tol, tol, tol),
            box.max_bound_ + Vector3d(tol, tol, tol)};

    using plane_t = Eigen::Hyperplane<double, 3>;
    std::array<plane_t, 6> planes {{{{-1, 0, 0}, box.min_bound_},
                                           {{1, 0, 0}, box.max_bound_},
                                           {{0, -1, 0}, box.min_bound_},
                                           {{0, 1, 0}, box.max_bound_},
                                           {{0, 0, -1}, box.min_bound_},
                                           {{0, 0, 1}, box.max_bound_}}};

    // Get the intersections
    std::vector<double> parameters{0};
    std::vector<Eigen::Vector3d> points{ray.origin()};

    for (int i = 0; i < 6; ++i) {
        double t =  ray.intersectionParameter(planes[i]);
        if (!std::isinf(t) && t >= 0) {
            parameters.push_back(t);
            auto p = ray.pointAt(t);
            points.push_back(p);
        }
    }

    // Find the ones which are contained
    auto contained_indices = b_tol.GetPointIndicesWithinBoundingBox(points);
    if (contained_indices.empty())
        return std::nan("");

    // Return the lowest parameter
    double minimum = parameters[contained_indices[0]];
    for (auto i : contained_indices) {
        minimum = std::min(minimum, parameters[i]);
    }
    return minimum;
}

}  // namespace geometry
}  // namespace open3d
