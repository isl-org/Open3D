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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/Geometry.h"
#include "open3d/utility/Optional.h"

#pragma once

namespace open3d {
namespace geometry {

/// \class Line3D
///
/// \brief Line3D is a class which derives from
/// Eigen::ParametrizedLine<double, 3> in order to capture the semantic
/// differences between a "line", "ray", and "line segment" for operations in
/// which the difference is important, such as intersection and distance tests.
/// The underlying Eigen object can always be retrieved with the .Line() method.
///
/// \details The taxonomy of the Line3D class and its derived classes, Ray3D
/// and Segment3D, was created in order to find a compromise between the
/// goals of providing an easy-to-use API, enforcing obvious correctness
/// when dealing with operations in which line semantics are important, and
/// maintaining reasonably high performance.
///
/// The underlying motivation is to enforce correctness when performing line
/// operations based on Eigen::ParametrizedLine<double, 3> even as subtleties
/// about how a line is represented begin to affect the outcomes of different
/// operations.  Some performance is sacrificed in the use of virtual functions
/// for a clean API in cases where the compiler cannot determine at compile time
/// which function will be called and cannot de-virtualize the call.
///
/// In such cases where performance is extremely important, avoid iterating
/// through a list of derived objects stored as Line3D and calling virtual
/// functions on them so that the compiler can hopefully remove the vtable
/// lookup, or consider a hand implementation of your problem in which you
/// carefully account for the semantics yourself.
class Line3D : protected Eigen::ParametrizedLine<double, 3> {
public:
    /// \brief Creates a line through two points.  The line origin will take the
    /// value of p0, and the line direction will be a normalized vector from
    /// p0 to p1
    static Line3D Through(const Eigen::Vector3d& p0,
                          const Eigen::Vector3d& p1) {
        return {p0, (p1 - p0).normalized()};
    }

    /// \enum LineType
    ///
    /// \brief Specifies different semantic interpretations of 3d lines
    enum class LineType {
        /// Lines extend infinitely in both directions
        Line = 0,

        /// Rays have an origin and a direction, and extend to infinity in
        /// that direction
        Ray = 1,

        /// Segments have both an origin and an endpoint and are finite in
        /// nature
        Segment = 2,
    };

    /// \brief Default user constructor
    Line3D(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction);

    virtual ~Line3D() = default;

    /// \brief Gets the semantic type of the line
    LineType GetLineType() const { return line_type_; }

    /// \brief Gets the line's origin point
    const Eigen::Vector3d& Origin() const { return m_origin; }

    /// \brief Gets the line's direction vector
    const Eigen::Vector3d& Direction() const { return m_direction; }

    /// \brief Gets the length of the line, which for lines and rays will return
    /// positive infinity, but for segments will return a finite positive value.
    virtual double Length() const {
        return std::numeric_limits<double>::infinity();
    }

    /// \brief Transform the Line3D by the given matrix
    virtual void Transform(
            const Eigen::Transform<double, 3, Eigen::Affine>& t) {
        this->transform(t);
    }

    /// \brief Returns a const reference to the underlying
    /// Eigen::ParametrizedLine object
    const Eigen::ParametrizedLine<double, 3>& Line() const { return *this; }

    /// \brief Calculates the intersection parameter between the line and a
    /// plane taking into account line semantics. Returns an empty result if
    /// there is no intersection. On a Line3D this returns the same result as
    /// .Line().intersectionParameter(plane)
    virtual utility::optional<double> IntersectionParameter(
            const Eigen::Hyperplane<double, 3>& plane) const;

    /// \brief Calculates the parameter of a point projected onto the line
    /// taking into account special semantics.
    ///
    /// \details On a Line3D this is the point directly projected onto the
    /// infinite line, and represents the closest point on the line to the test
    /// point. A negative value indicates the projection lies behind the origin,
    /// a positive value is in front of the origin. On a Ray3D this will be a
    /// positive value only, since rays do not exist in the negative direction.
    /// On a Segment3D this will be a positive value which is less than or equal
    /// to the segment length.
    double ProjectionParameter(const Eigen::Vector3d& point) const;

    /// \brief Calculates a point projected onto the line, taking into account
    /// special semantics.
    ///
    /// \details On a Line3D this is the point directly projected onto
    /// the infinite line, and represents the closest point on the line to the
    /// test point.  On a Ray3D this will either be a point on the ray's
    /// positive direction or the ray origin, whichever is closer. On a
    /// Segment3D this will be either one of the segment's endpoints or a point
    /// between them, whichever is closest to the test point.
    virtual Eigen::Vector3d Projection(const Eigen::Vector3d& point) const;

    /// \brief Returns the lower intersection parameter for a line with an
    /// axis aligned bounding box or empty if no intersection. Uses the slab
    /// method, see warning below.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// line with an axis aligned bounding box. The intersection point can be
    /// recovered with .Line().pointAt(...). If the line does not intersect the
    /// box the optional return value will be empty. Also note that if the AABB
    /// is behind the line's origin point, the value returned will still be of
    /// the lower intersection, which is the first intersection in the direction
    /// of the line, not the intersection closer to the origin.
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
    virtual utility::optional<double> SlabAABB(
            const AxisAlignedBoundingBox& box) const;

    /// \brief Returns the lower intersection parameter for a line with an
    /// axis aligned bounding box or empty if no intersection. This method is
    /// about 20x slower than the slab method, see details to know when to use.
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
    virtual utility::optional<double> ExactAABB(
            const AxisAlignedBoundingBox& box) const;

    /// \brief Computes the two corresponding parameters of the closest distance
    /// between two Line3D objects, including derived types Ray3D and Segment3D,
    /// respecting the semantics of the line type.
    std::pair<double, double> ClosestParameters(const Line3D& other) const;

    /// \brief Computes the two closest points between this Line3D object and
    /// the other, including of derived types Ray3D and Segment3D, respecting
    /// the semantics of the line types.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> ClosestPoints(
            const Line3D& other) const;

    /// \brief Gets the closest distance between two Line3D objects, including
    /// derived types Ray3D and Segment3D, respecting the semantics of the line
    /// type.
    double DistanceTo(const Line3D& other) const;

    /// \brief Clamps/bounds a parameter value to the closest valid place where
    /// the entity exists.  On a Line3D, the value will be unchanged, on a Ray3D
    /// a negative value will be made 0, and on a Segment3D a negative value
    /// will be made 0 and a positive value greater than Length() will take
    /// the value of Length()
    virtual double ClampParameter(double parameter) const { return parameter; }

    /// \brief Verifies that a given parameter value is valid for the semantics
    /// of the line object. For lines, any parameter is valid, for rays any
    /// positive parameter is valid, and for segments any parameter between 0
    /// and the segment length is valid.
    virtual bool IsParameterValid(double parameter) const { return true; }

protected:
    /// \brief Internal constructor for inherited classes that allows the
    /// setting of the LineType
    Line3D(const Eigen::Vector3d& origin,
           const Eigen::Vector3d& direction,
           LineType type);

    /// \brief Calculates the common t_min and t_max values of the slab AABB
    /// intersection method. These values are computed identically for any
    /// semantic interpretation of the line, it's up to the derived classes
    /// to use them in conjunction with other information to determine what the
    /// intersection parameter is.
    std::pair<double, double> SlabAABBBase(
            const AxisAlignedBoundingBox& box) const;

private:
    const LineType line_type_ = LineType::Line;

    // Pre-calculated inverse values for the line's direction used to
    // accelerate the slab method
    double x_inv_;
    double y_inv_;
    double z_inv_;
};

/// \class Ray3D
///
/// \brief A ray is a semantic interpretation of Eigen::ParametrizedLine which
/// has an origin and a direction and extends infinitely only in that specific
/// direction.
class Ray3D : public Line3D {
public:
    /// \brief Creates a Ray3D through two points.  The ray origin will take the
    /// value of p0, and the direction will be a normalized vector from p0 to p1
    static Ray3D Through(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) {
        return {p0, (p1 - p0).normalized()};
    }

    /// \brief Default constructor, requires point and direction
    Ray3D(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction);

    /// \brief Gets the length of the line, which for lines and rays will return
    /// positive infinity, but for segments will return a finite positive value.
    double Length() const override {
        return std::numeric_limits<double>::infinity();
    }

    /// \brief Calculates the intersection parameter between the line and a
    /// plane taking into account ray semantics. Returns an empty result if
    /// there is no intersection. On a Ray3D this means that intersections
    /// behind the origin are invalid, so the return value will always be
    /// positive.
    utility::optional<double> IntersectionParameter(
            const Eigen::Hyperplane<double, 3>& plane) const override;

    /// \brief Returns the lower intersection parameter for a ray with an
    /// axis aligned bounding box or empty if no intersection. Uses the slab
    /// method, see warning below.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// ray with an axis aligned bounding box. The intersection point can be
    /// recovered with .Line().pointAt(...). If the ray does not intersect the
    /// box the optional return value will be empty. No intersection behind the
    /// ray origin will be counted, and if the ray originates from within the
    /// bounding box the parameter value will be 0.
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
    utility::optional<double> SlabAABB(
            const AxisAlignedBoundingBox& box) const override;

    /// \brief Clamps/bounds a parameter value to the closest valid place where
    /// the entity exists.  On a Line3D, the value will be unchanged, on a Ray3D
    /// a negative value will be made 0, and on a Segment3D a negative value
    /// will be made 0 and a positive value greater than Length() will take
    /// the value of Length()
    double ClampParameter(double parameter) const override {
        return std::max(parameter, 0.);
    }

    /// \brief Verifies that a given parameter value is valid for the semantics
    /// of the line object. For lines, any parameter is valid, for rays any
    /// positive parameter is valid, and for segments any parameter between 0
    /// and the segment length is valid.
    bool IsParameterValid(double parameter) const override {
        return parameter >= 0;
    }
};

/// \class Segment3D
///
/// \brief A segment is a semantic interpretation of Eigen::ParametrizedLine
/// which has an origin and an endpoint and exists finitely between them.
///
/// \details One of the main motivations behind this class, and the Line3D
/// taxonomy in general, is the ambiguity of the Eigen documentation with
/// regards to the ParametrizedLine's direction. The documentation warns that
/// the direction vector is expected to be normalized and makes no guarantees
/// about behavior when this expectation is not met.  However, ParametrizedLine
/// does behave correctly when the direction vector is scaled. This class
/// exists as a seam to ensure the correct behavior can be produced regardless
/// of what happens in the underlying Eigen implementation without changing
/// the api surface for client code.
class Segment3D : public Line3D {
public:
    /// \brief Creates a Segment3D through two points.  The origin will take the
    /// value of p0, and the endpoint be p1. The direction will be a normalized
    /// vector from p0 to p1.
    static Segment3D Through(const Eigen::Vector3d& p0,
                             const Eigen::Vector3d& p1) {
        return {p0, p1};
    }

    /// \brief Default constructor for Segment3D takes the start and end points
    /// of the segment \param start_point \param end_point
    Segment3D(const Eigen::Vector3d& start_point,
              const Eigen::Vector3d& end_point);

    /// \brief Takes a std::pair of points, the first to be used as the start
    /// point/origin and the second to be the end point
    explicit Segment3D(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pair);

    /// \brief Get the scalar length of the segment as the distance between the
    /// start point (origin) and the end point.
    double Length() const override { return length_; }

    /// \brief Calculates the midpoint of the segment
    Eigen::Vector3d MidPoint() const { return 0.5 * (origin() + end_point_); }

    /// \brief Get the end point of the segment
    const Eigen::Vector3d& EndPoint() const { return end_point_; }

    /// \brief Transform the segment by the given matrix
    void Transform(
            const Eigen::Transform<double, 3, Eigen::Affine>& t) override {
        this->transform(t);
        end_point_ = t * end_point_;
    }

    /// \brief Get an axis-aligned bounding box representing the enclosed volume
    /// of the line segment.
    AxisAlignedBoundingBox GetBoundingBox() const;

    /// \brief Calculates the intersection parameter between the line and a
    /// plane taking into account segment semantics. Returns an empty result if
    /// there is no intersection. On a Segment3D this means that intersections
    /// behind the origin and beyond the endpoint are invalid.
    utility::optional<double> IntersectionParameter(
            const Eigen::Hyperplane<double, 3>& plane) const override;

    /// \brief Returns the lower intersection parameter for a segment with an
    /// axis aligned bounding box or empty if no intersection. Uses the slab
    /// method, see warning below.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// segment with an axis aligned bounding box. The intersection point can be
    /// recovered with .Line().pointAt(...). If the segment does not
    /// intersect the box the optional return value will be empty. No
    /// intersection behind the segment origin will be counted, and if the
    /// segment originates from within the bounding box the parameter value will
    /// be 0. No intersection beyond the endpoint of the segment will be
    /// considered either.
    ///
    /// This implementation is based off of Tavian Barnes' optimized branchless
    /// slab method. https://tavianator.com/2011/segment_box.html. It runs in
    /// roughly 5% of the time as the the naive exact method, but can degenerate
    /// in specific conditions where a segment lies exactly in one of the AABB's
    /// planes.
    ///
    /// \warning A segment that lies exactly in one of the AABB's planes within
    /// the double floating point precision will not intersect correctly by this
    /// method
    utility::optional<double> SlabAABB(
            const AxisAlignedBoundingBox& box) const override;

    /// \brief Returns the lower intersection parameter for a segment with an
    /// axis aligned bounding box or empty if no intersection. This method is
    /// about 20x slower than the slab method, see details to know when to use.
    ///
    /// \details Calculates the lower intersection parameter of a parameterized
    /// segment with an axis aligned bounding box. The intersection point can be
    /// recovered with .Line().pointAt(...). If the segment does not
    /// intersect the box the return value will be empty.
    ///
    /// This implementation is a naive exact method that considers intersections
    /// with all six bounding box planes. It is not optimized for speed and
    /// should only be used when a problem is conditioned such that the slab
    /// method is unacceptable. Use this when a segment is likely to lie exactly
    /// in one of the AABB planes and false negatives are unacceptable.
    /// Typically this will only happen when segments are axis-aligned and both
    /// segments and bounding volumes are regularly spaced, and every
    /// intersection is important.  In such cases if performance is important, a
    /// simple custom implementation based on the problem directionality will
    /// likely outperform even the slab method.
    utility::optional<double> ExactAABB(
            const AxisAlignedBoundingBox& box) const override;

    /// \brief Clamps/bounds a parameter value to the closest valid place where
    /// the entity exists.  On a Line3D, the value will be unchanged, on a Ray3D
    /// a negative value will be made 0, and on a Segment3D a negative value
    /// will be made 0 and a positive value greater than Length() will take
    /// the value of Length()
    double ClampParameter(double parameter) const override {
        return std::max(std::min(parameter, length_), 0.);
    }

    /// \brief Verifies that a given parameter value is valid for the semantics
    /// of the line object. For lines, any parameter is valid, for rays any
    /// positive parameter is valid, and for segments any parameter between 0
    /// and the segment length is valid.
    bool IsParameterValid(double parameter) const override {
        return parameter >= 0 && parameter <= length_;
    }

private:
    Eigen::Vector3d end_point_;
    double length_;
};

}  // namespace geometry
}  // namespace open3d
