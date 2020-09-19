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
    /// axis aligned bounding box or NaN if no intersection. This method is
    /// about 20x slower than the slab method, see details to know when to use.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// line with an axis aligned bounding box. The intersection point can be
    /// recovered with line.PointAt(...). If the line does not intersect the
    /// box the return value will be NaN. Also note that if the AABB is behind
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
    static double LineAABBExactParam(
            const Eigen::ParametrizedLine<double, 3>& line,
            const AxisAlignedBoundingBox& box);

    /// \brief Returns true if the line intersects the AABB. This method is
    /// about 20x slower than the slab method, see details to know when to use.
    ///
    /// \details Checks if the line intersects the axis aligned bounding box and
    /// returns true if it does.
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
    static bool LineAABBExact(
            const Eigen::ParametrizedLine<double, 3>& line,
            const AxisAlignedBoundingBox& box) {
            return !std::isnan(LineAABBExactParam(line, box));
    }

    /// \brief Returns the lower intersection parameter for a ray with an
    /// axis aligned bounding box or NaN if no intersection. Returns 0 if the
    /// ray originates inside the volume. This method is about 20x slower than
    /// the slab method, see details to know when to use.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// ray with an axis aligned bounding box. The intersection point can be
    /// recovered with ray.PointAt(...). If the ray does not intersect the
    /// box the return value will be NaN. If the ray origin is contained within
    /// the box the parameter will be zero.
    ///
    /// This implementation is a naive exact method that considers intersections
    /// with all six bounding box planes. It is not optimized for speed and
    /// should only be used when a problem is conditioned such that the slab
    /// method is unacceptable. Use this when a ray is likely to lie exactly
    /// in one of the AABB planes and false negatives are unacceptable.
    /// Typically this will only happen when rays are axis-aligned and both
    /// lines and bounding volumes are regularly spaced, and every intersection
    /// is important.  In such cases if performance is important, a simple
    /// custom implementation based on the problem directionality will likely
    /// outperform even the slab method.
    static double RayAABBExactParam(
            const Eigen::ParametrizedLine<double, 3>& ray,
            const AxisAlignedBoundingBox& box);

    /// \brief Returns true if the ray intersects the AABB. This method is
    /// about 20x slower than the slab method, see details to know when to use.
    ///
    /// \details Checks if the ray intersects the axis aligned bounding box and
    /// returns true if it does.
    ///
    /// This implementation is a naive exact method that considers intersections
    /// with all six bounding box planes. It is not optimized for speed and
    /// should only be used when a problem is conditioned such that the slab
    /// method is unacceptable. Use this when a ray is likely to lie exactly
    /// in one of the AABB planes and false negatives are unacceptable.
    /// Typically this will only happen when rays are axis-aligned and both
    /// lines and bounding volumes are regularly spaced, and every intersection
    /// is important.  In such cases if performance is important, a simple
    /// custom implementation based on the problem directionality will likely
    /// outperform even the slab method.
    static bool RayAABBExact(
            const Eigen::ParametrizedLine<double, 3>& ray,
            const AxisAlignedBoundingBox& box) {
        return !std::isnan(RayAABBExactParam(ray, box));
    }

    /// \brief Returns the lower intersection parameter for a line with an
    /// axis aligned bounding box or NaN if no intersection. Takes pre-computed
    /// direction coefficients to speed up multiple checks using the same line.
    /// Uses the slab method, see warning below.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// line with an axis aligned bounding box. The intersection point can be
    /// recovered with line.PointAt(...). If the line does not intersect the
    /// box the return value will be NaN. Also note that if the AABB is behind
    /// the line's origin point, the value returned will still be of the lower
    /// intersection, which is the first intersection in the direction of the
    /// line, not the intersection closer to the origin.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a line lies exactly in one of the AABB's
    /// planes.
    ///
    /// \warning A line that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static double LineAABBSlabParam(
            const Eigen::ParametrizedLine<double, 3>& line,
            const AxisAlignedBoundingBox& box);

    /// \brief Returns the lower intersection parameter for a line with an
    /// axis aligned bounding box or NaN if no intersection. Takes pre-computed
    /// direction coefficients to speed up multiple checks using the same line.
    /// Uses the slab method, see warning below.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// line with an axis aligned bounding box. The intersection point can be
    /// recovered with line.PointAt(...). If the line does not intersect the
    /// box the return value will be NaN. Also note that if the AABB is behind
    /// the line's origin point, the value returned will still be of the lower
    /// intersection, which is the first intersection in the direction of the
    /// line, not the intersection closer to the origin.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a line lies exactly in one of the AABB's
    /// planes.
    ///
    /// This version allows pre-computed inversions of the line's direction
    /// vector components to speed up checks involving the same line, since the
    /// division operation is computationally expensive. This allows for about
    /// a 17% additional speed up over the plain slab method.
    ///
    /// \warning A line that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static double LineAABBSlabParam(
            const Eigen::ParametrizedLine<double, 3>& line,
            const AxisAlignedBoundingBox& box,
            double dir_x_inv, double dir_y_inv, double dir_z_inv);

    /// \brief Returns true if the line intersects the AABB. Uses the slab
    /// method, see warning.
    ///
    /// \details Checks if the line intersects the axis aligned bounding box and
    /// returns true if it does.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a line lies exactly in one of the AABB's
    /// planes.
    ///
    /// \warning A line that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static bool LineAABBSlab(const Eigen::ParametrizedLine<double, 3>& line,
                             const AxisAlignedBoundingBox& box) {
        return !std::isnan(LineAABBSlabParam(line, box));
    }

    /// \brief Returns true if the line intersects the AABB. Uses the slab
    /// method, see warning.
    ///
    /// \details Checks if the line intersects the axis aligned bounding box and
    /// returns true if it does.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a line lies exactly in one of the AABB's
    /// planes.
    ///
    /// This version allows pre-computed inversions of the line's direction
    /// vector components to speed up checks involving the same line, since the
    /// division operation is computationally expensive. This allows for about
    /// a 17% additional speed up over the plain slab method.
    ///
    /// \warning A line that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static bool LineAABBSlab(const Eigen::ParametrizedLine<double, 3>& line,
                 const AxisAlignedBoundingBox& box,
                 double dir_x_inv, double dir_y_inv, double dir_z_inv) {
        return !std::isnan(LineAABBSlabParam(line, box, dir_x_inv, dir_y_inv,
                                             dir_z_inv));
    }

    /// \brief Returns the lower intersection parameter for a ray with an
    /// axis aligned bounding box or NaN if no intersection. Returns 0 if the
    /// ray originates from inside the volume.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// ray with an axis aligned bounding box. The intersection point can be
    /// recovered with ray.PointAt(...). If the ray does not intersect the
    /// box the return value will be NaN. If the ray origin is contained within
    /// the box the parameter will be zero.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a ray lies exactly in one of the AABB's
    /// planes.
    ///
    /// \warning A ray that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static double RayAABBSlabParam(
            const Eigen::ParametrizedLine<double, 3> &ray,
            const AxisAlignedBoundingBox& box);

    /// \brief Returns the lower intersection parameter for a ray with an
    /// axis aligned bounding box or NaN if no intersection. Returns 0 if the
    /// ray originates from inside the volume. Takes pre-computed direction
    /// coefficients to speed up multiple checks using the same ray. Uses the
    /// slab method, see warning below.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// ray with an axis aligned bounding box. The intersection point can be
    /// recovered with ray.PointAt(...). If the ray does not intersect the
    /// box the return value will be NaN. If the ray origin is contained within
    /// the box the parameter will be zero.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a ray lies exactly in one of the AABB's
    /// planes.
    ///
    /// This version allows pre-computed inversions of the ray's direction
    /// vector components to speed up checks involving the same ray, since the
    /// division operation is computationally expensive. This allows for about
    /// a 17% additional speed up over the plain slab method.
    ///
    /// \warning A ray that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static double RayAABBSlabParam(
            const Eigen::ParametrizedLine<double, 3>& ray,
            const AxisAlignedBoundingBox& box,
            double dir_x_inv, double dir_y_inv, double dir_z_inv);

    /// \brief Returns true if the ray intersects the AABB. Uses the slab
    /// method, see warning.
    ///
    /// \details Checks if the ray intersects the axis aligned bounding box and
    /// returns true if it does.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a ray lies exactly in one of the AABB's
    /// planes.
    ///
    /// \warning A ray that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static bool RayAABBSlab(const Eigen::ParametrizedLine<double, 3>& ray,
                        const AxisAlignedBoundingBox& box) {
        return !std::isnan(RayAABBSlabParam(ray, box));
    }

    /// \brief Returns true if the ray intersects the AABB. Uses the slab
    /// method, see warning. Takes pre-computed direction coefficients to speed
    /// up multiple checks using the same line. Uses the slab method, see
    /// warning below.
    ///
    /// \details Checks if the ray intersects the axis aligned bounding box and
    /// returns true if it does.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/ray_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a ray lies exactly in one of the AABB's
    /// planes.
    ///
    /// This version allows pre-computed inversions of the ray's direction
    /// vector components to speed up checks involving the same ray, since the
    /// division operation is computationally expensive. This allows for about
    /// a 17% additional speed up over the plain slab method.
    ///
    /// \warning A ray that lies exactly in one of the AABB's planes within the
    /// double floating point precision will not intersect correctly by this
    /// method
    static bool RayAABBSlab(const Eigen::ParametrizedLine<double, 3>& ray,
                    const AxisAlignedBoundingBox& box,
                    double dir_x_inv, double dir_y_inv, double dir_z_inv) {
        return !std::isnan(RayAABBSlabParam(ray, box, dir_x_inv, dir_y_inv,
                                            dir_z_inv));
    }
};

}  // namespace geometry
}  // namespace open3d
