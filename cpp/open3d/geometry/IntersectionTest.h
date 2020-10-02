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

#include <Eigen/Dense>

#include "BoundingVolume.h"
#include "Line3D.h"

namespace open3d {
namespace geometry {

class IntersectionTest {
public:
    static bool AABBAABB(const Eigen::Vector3d& min0,
                         const Eigen::Vector3d& max0,
                         const Eigen::Vector3d& min1,
                         const Eigen::Vector3d& max1);

    static bool TriangleTriangle3d(const Eigen::Vector3d& p0,
                                   const Eigen::Vector3d& p1,
                                   const Eigen::Vector3d& p2,
                                   const Eigen::Vector3d& q0,
                                   const Eigen::Vector3d& q1,
                                   const Eigen::Vector3d& q2);

    static bool TriangleAABB(const Eigen::Vector3d& box_center,
                             const Eigen::Vector3d& box_half_size,
                             const Eigen::Vector3d& vert0,
                             const Eigen::Vector3d& vert1,
                             const Eigen::Vector3d& vert2);

    /// Tests if the given four points all lie on the same plane.
    static bool PointsCoplanar(const Eigen::Vector3d& p0,
                               const Eigen::Vector3d& p1,
                               const Eigen::Vector3d& p2,
                               const Eigen::Vector3d& p3);

    /// Computes the minimum distance between two lines. The first line is
    /// defined by 3D points \p p0 and \p p1, the second line is defined
    /// by 3D points \p q0 and \p q1. The returned distance is negative
    /// if no minimum distance can be computed. This implementation is based on
    /// the description of Paul Bourke
    /// (http://paulbourke.net/geometry/pointlineplane/).
    static double LinesMinimumDistance(const Eigen::Vector3d& p0,
                                       const Eigen::Vector3d& p1,
                                       const Eigen::Vector3d& q0,
                                       const Eigen::Vector3d& q1);

    /// Computes the minimum distance between two line segments. The first line
    /// segment is defined by 3D points \p p0 and \p p1, the second line
    /// segment is defined by 3D points \p q0 and \p q1. This
    /// implementation is based on the description of David Eberly
    /// (https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf).
    static double LineSegmentsMinimumDistance(const Eigen::Vector3d& p0,
                                              const Eigen::Vector3d& p1,
                                              const Eigen::Vector3d& q0,
                                              const Eigen::Vector3d& q1);

    /// \brief Returns the lower intersection parameter for a line with an
    /// axis aligned bounding box or empty if no intersection. This method is
    /// about 20x slower than the slab method, see details to know when to use.
    /// This function wraps the underlying implementation on the Line3D class
    /// and is included here for API coherence; if you are testing large numbers
    /// of lines, rays, or segments use the Line3D, Ray3D, or Segment3D classes
    /// directly.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// line with an axis aligned bounding box. The intersection point can be
    /// recovered with .Line().pointAt(...). If the line does not intersect the
    /// box the return value will be empty. Also note that if the AABB is behind
    /// the line's origin point, the value returned will still be of the lower
    /// intersection, which is the first intersection in the direction of the
    /// line, not the intersection closer to the origin.
    ///
    /// This implementation is a naive exact method that considers intersections
    /// with all six bounding box planes. It is not optimized for speed and
    /// should only be used when a problem is conditioned such that the slab
    /// method is unacceptable. Use this when a line is likely to lie exactly
    /// in one of the AABB planes and false negatives are unacceptable.
    /// Typically this will only happen when lines are axis-aligned and both
    /// lines and bounding volumes are regularly spaced, and every intersection
    /// is important.  In such cases if performance is important, a simple
    /// custom implementation based on the problem directionality will likely
    /// outperform even the slab method.
    /// \code{.cpp}
    /// // Intersection with a line
    /// auto result = IntersectionTest::LineExactAABB(Line3D{p, n}, box);
    /// if (result.has_value()) {
    ///     ...
    /// }
    ///
    /// // Intersection with a ray
    /// auto result = IntersectionTest::LineExactAABB(Ray3D{p, n}, box);
    /// if (result.has_value()) {
    ///     ...
    /// }
    ///
    /// // Intersection with a segment
    /// auto result = IntersectionTest::LineExactAABB(Segment3D{p0, p1}, box);
    /// if (result.has_value()) {
    ///     ...
    /// }
    ///
    /// // Getting the intersection point
    /// Ray3D ray{p, n};
    /// auto result = IntersectionTest::LineSlabAABB(ray, box);
    /// if (result.has_value()) {
    ///     // the .Line() function retrieves the underlying Eigen object
    ///     ray.Line().pointAt(result.value());
    /// }
    /// \endcode
    static utility::optional<double> LineExactAABB(
            const Line3D& line, const AxisAlignedBoundingBox& box) {
        return line.ExactAABB(box);
    }

    /// \brief Returns the lower intersection parameter for a line with an
    /// axis aligned bounding box or no value if there is no intersection. This
    /// function wraps the underlying implementation on the Line3D class and
    /// is included here for API coherence; if you are testing large numbers
    /// of lines, rays, or segments use the Line3D, Ray3D, or Segment3D classes
    /// directly.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// line with an axis aligned bounding box. The intersection point can be
    /// recovered with .Line().pointAt(...). If the line does not intersect the
    /// box the return value will be empty. Also note that if the AABB is behind
    /// the line's origin point, the value returned will still be of the lower
    /// intersection, which is the first intersection in the direction of the
    /// line, not the intersection closer to the origin.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a line lies exactly in one of the AABB's
    /// planes.
    /// \code{.cpp}
    /// // Intersection with a line
    /// auto result = IntersectionTest::LineSlabAABB(Line3D{p, n}, box);
    /// if (result.has_value()) {
    ///     ...
    /// }
    ///
    /// // Intersection with a ray
    /// auto result = IntersectionTest::LineSlabAABB(Ray3D{p, n}, box);
    /// if (result.has_value()) {
    ///     ...
    /// }
    ///
    /// // Intersection with a segment
    /// auto result = IntersectionTest::LineSlabAABB(Segment3D{p0, p1}, box);
    /// if (result.has_value()) {
    ///     ...
    /// }
    ///
    /// // Getting the intersection point
    /// Ray3D ray{p, n};
    /// auto result = IntersectionTest::LineSlabAABB(ray, box);
    /// if (result.has_value()) {
    ///     // the .Line() function retrieves the underlying Eigen object
    ///     ray.Line().pointAt(result.value());
    /// }
    /// \endcode
    ///
    /// \warning A line that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static utility::optional<double> LineSlabAABB(
            const Line3D& line, const AxisAlignedBoundingBox& box) {
        return line.SlabAABB(box);
    }
};

}  // namespace geometry
}  // namespace open3d
