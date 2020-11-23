#include "Line3D.h"

#include <cmath>

namespace open3d {
namespace geometry {

// Line3D Implementations
// ===========================================================================
Line3D::Line3D(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction)
    : Line3D(origin, direction, LineType::Line) {}

Line3D::Line3D(const Eigen::Vector3d& origin,
               const Eigen::Vector3d& direction,
               Line3D::LineType type)
    : Eigen::ParametrizedLine<double, 3>(origin, direction), line_type_(type) {
    // Here we pre-compute the inverses of the direction vector for use during
    // the slab AABB intersection algorithm. The choice to do this regardless
    // of the purpose of the line's creation isn't optimal, but has to do with
    // trying to keep both the outer API clean and avoid the overhead of having
    // to check that the values have been computed before running any of the
    // slab intersection algorithms, since the slab algorithm is the most
    // performance critical operation this construct is used for.
    x_inv_ = 1. / direction.x();
    y_inv_ = 1. / direction.y();
    z_inv_ = 1. / direction.z();
}

std::pair<double, double> Line3D::SlabAABBBase(
        const AxisAlignedBoundingBox& box) const {
    /* This code is based off of Tavian Barnes' branchless implementation of
     * the slab method for determining ray/AABB intersections. It treats the
     * space inside the bounding box as three sets of parallel planes and clips
     * the line where it passes between these planes to produce an
     * ever-shrinking line segment.
     *
     * The line segment is represented by two scalar parameters, one for the
     * clipped end on the far plane and one for the clipped end on the near
     * plane. When the line does not pass through the bounding box the t_max
     * and t_min distances will invert.
     *
     * https://tavianator.com/2011/ray_box.html */
    double t_x0 = x_inv_ * (box.min_bound_.x() - origin().x());
    double t_x1 = x_inv_ * (box.max_bound_.x() - origin().x());
    double t_min = std::min(t_x0, t_x1);
    double t_max = std::max(t_x0, t_x1);

    double t_y0 = y_inv_ * (box.min_bound_.y() - origin().y());
    double t_y1 = y_inv_ * (box.max_bound_.y() - origin().y());
    t_min = std::max(t_min, std::min(t_y0, t_y1));
    t_max = std::min(t_max, std::max(t_y0, t_y1));

    double t_z0 = z_inv_ * (box.min_bound_.z() - origin().z());
    double t_z1 = z_inv_ * (box.max_bound_.z() - origin().z());
    t_min = std::max(t_min, std::min(t_z0, t_z1));
    t_max = std::min(t_max, std::max(t_z0, t_z1));

    return {t_min, t_max};
}

utility::optional<double> Line3D::ExactAABB(
        const AxisAlignedBoundingBox& box) const {
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
    using namespace Eigen;

    // When running a stress test on randomly generated rays and boxes about 1%
    // to 2% of the randomly generated cases will fail when using this method
    // due to the round-trip vector coming back from the ParameterizedLine's
    // intersectionParameter method being off in the 11th or greater decimal
    // position from the original plane point. This tolerance seems to
    // eliminate the issue.
    double tol = 1e-10;
    AxisAlignedBoundingBox b_tol{box.min_bound_ - Vector3d(tol, tol, tol),
                                 box.max_bound_ + Vector3d(tol, tol, tol)};

    using plane_t = Eigen::Hyperplane<double, 3>;
    std::array<plane_t, 6> planes{{{{-1, 0, 0}, box.min_bound_},
                                   {{1, 0, 0}, box.max_bound_},
                                   {{0, -1, 0}, box.min_bound_},
                                   {{0, 1, 0}, box.max_bound_},
                                   {{0, 0, -1}, box.min_bound_},
                                   {{0, 0, 1}, box.max_bound_}}};

    // Get the intersections
    std::vector<double> parameters;
    std::vector<Eigen::Vector3d> points;
    parameters.reserve(7);
    points.reserve(7);

    if (line_type_ == LineType::Ray || line_type_ == LineType::Segment) {
        parameters.push_back(0);
        points.push_back(origin());
    }

    for (int i = 0; i < 6; ++i) {
        auto t = IntersectionParameter(planes[i]);
        if (t.has_value()) {
            parameters.push_back(t.value());
            auto p = pointAt(t.value());
            points.push_back(p);
        }
    }

    // Find the ones which are contained
    auto contained_indices = b_tol.GetPointIndicesWithinBoundingBox(points);
    if (contained_indices.empty()) return {};

    // Return the lowest parameter
    double minimum = parameters[contained_indices[0]];
    for (auto i : contained_indices) {
        minimum = std::min(minimum, parameters[i]);
    }
    return minimum;
}

utility::optional<double> Line3D::SlabAABB(
        const AxisAlignedBoundingBox& box) const {
    /* The base case of the Line/AABB intersection allows for any intersection
     * along the direction of the line at any distance, in accordance with the
     * semantic meaning of a line.
     *
     * In such a case the only test is to determine if t_min and t_max have
     * inverted, indicating that the line does not pass through the box at all.
     */
    const auto& t = SlabAABBBase(box);
    double t_min = std::get<0>(t);
    double t_max = std::get<1>(t);

    if (t_max >= t_min) return t_min;
    return {};
}

utility::optional<double> Line3D::IntersectionParameter(
        const Eigen::Hyperplane<double, 3>& plane) const {
    double value = intersectionParameter(plane);
    if (std::isinf(value)) {
        return {};
    } else {
        return value;
    }
}

Eigen::Vector3d Line3D::Projection(const Eigen::Vector3d& point) const {
    return Line().pointAt(ProjectionParameter(point));
}

double Line3D::ProjectionParameter(const Eigen::Vector3d& point) const {
    return ClampParameter(direction().dot(point - origin()));
}

std::pair<double, double> Line3D::ClosestParameters(const Line3D& other) const {
    /* The book "Real-Time Collision Detection" by Christer Ericson discusses
     * in section 5.1.9 (p148) finding the closest point between two line
     * segments.
     *
     *   * The line-line solution is valid for segments when the parameters it
     *     produces are valid on both segments
     *
     *   * If just one of the closest points between lines is outside of its
     *     segment, the point can be clamped to the segment and the result will
     *     be valid
     *
     *   * If both points are outside of their respective segments, the clamping
     *     procedure ust be repeated twice. First, one of the points is clamped
     *     to its segment. Then it is projected onto the second segment, which
     *     may return an out-of-bound value as well. If it does, it must be
     *     clamped to its (the second) segment, then projected back onto the
     *     first segment. At that point the projection should be valid, and the
     *     closest distance is spanned by the two final parameters.
     *
     * Lines and rays in this case can be seen as a special case of a segment
     * in which the validity of parameters are unbounded in one or more
     * directions. Thus the same algorithm will account for lines and rays as
     * well as segments.
     *
     * The last edge case is the case of parallel lines. In such a case an
     * arbitrary parameter on one or the other line may be selected and the
     * corresponding parameter on the other line computed. However, in the non-
     * infinite case of rays and segments, this will not necessarily return a
     * valid result.
     *
     * As such, the project/clamp + project/clamp procedure will work starting
     * from this arbitrary point.
     */

    // This first portion of the code performs the general computation of
    // line parameters at the closest span between two infinite parameterized
    // lines, and is adapted from the implementation found at
    // http://geomalgorithms.com/a07-_distance.html
    const auto& u = Direction();
    const auto& v = other.Direction();
    auto w = Origin() - other.Origin();
    double a = u.dot(u);
    double b = u.dot(v);
    double c = v.dot(v);
    double d = u.dot(w);
    double e = v.dot(w);
    double D = a * c - b * b;

    double sc, tc;
    if (D < 1e-10) {
        // In the case of almost parallel lines, we arbitrarily pick the first
        // parameter to be 0 and then compute the other line's corresponding
        // parameter.
        sc = 0.;
        tc = b > c ? d / b : e / c;
    } else {
        sc = (b * e - c * d) / D;
        tc = (a * e - b * d) / D;
    }

    // At this point, sc is the parameter of the closest point on *this line's
    // infinite representation and tc is the corresponding point on the other
    // line's infinite representation. If they are both valid, they can be
    // returned as-is, if not a clamp/project round-trip must be executed
    if (IsParameterValid(sc) && other.IsParameterValid(tc)) {
        return {sc, tc};
    }

    /* To avoid an edge case when two segments are parallel and disjoint and
     * the origin of the first segment is further from `other` than its
     * endpoint, we cannot take advantage of a single valid parameter. Instead
     * we must clamp, project to the other line, clamp, and project back to
     * this line.
     *
     * The .Project() method on Ray3D and Segment3D takes care of clamping, so
     * it's a straightforward procedure.
     */

    // Clamp sc to this line, then find tc by projecting onto the other line.
    sc = ClampParameter(sc);
    tc = other.ProjectionParameter(Line().pointAt(sc));

    // tc will already be clamped to other, so now we recompute sc
    sc = ProjectionParameter(other.Line().pointAt(tc));

    return {sc, tc};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> Line3D::ClosestPoints(
        const Line3D& other) const {
    /* The two closest points between two line entities is easily found by
     * using the ClosestParameters method and rendering the parameters into
     * points. */
    auto result = ClosestParameters(other);
    return {Line().pointAt(std::get<0>(result)),
            other.Line().pointAt(std::get<1>(result))};
}

double Line3D::DistanceTo(const Line3D& other) const {
    /* The distance between two line entities is the distance between the
     * closest points */
    auto pair = ClosestPoints(other);

    return (std::get<0>(pair) - std::get<1>(pair)).norm();
}

// Ray3D Implementations
// ===========================================================================

Ray3D::Ray3D(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction)
    : Line3D(origin, direction, LineType::Ray) {}

utility::optional<double> Ray3D::SlabAABB(
        const AxisAlignedBoundingBox& box) const {
    /* The ray case of the Line/AABB intersection allows for any intersection
     * along the positive direction of the line at any distance, in accordance
     * with the semantic meaning of a ray.
     *
     * In this case there are two conditions which must be met: t_max must be
     * greater than t_min (the check for inversion), but it must also be greater
     * than zero, as a negative t_max would indicate that the most positive
     * intersection along the ray direction still lies behind the ray origin.
     */
    const auto& t = SlabAABBBase(box);
    double t_min = std::get<0>(t);
    double t_max = std::get<1>(t);

    t_min = std::max(0., t_min);

    if (t_max >= t_min) return t_min;
    return {};
}

utility::optional<double> Ray3D::IntersectionParameter(
        const Eigen::Hyperplane<double, 3>& plane) const {
    // On a ray, the intersection parameter cannot be negative as the ray does
    // not exist in that direction.
    auto result = Line().intersectionParameter(plane);
    if (!std::isinf(result) && result >= 0) {
        return result;
    }
    return {};
}

// Segment3D Implementations
// ===========================================================================

Segment3D::Segment3D(const Eigen::Vector3d& start_point,
                     const Eigen::Vector3d& end_point)
    : Line3D(start_point,
             (end_point - start_point).normalized(),
             LineType::Segment),
      end_point_(end_point),
      length_((start_point - end_point_).norm()) {}

Segment3D::Segment3D(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pair)
    : Segment3D(std::get<0>(pair), std::get<1>(pair)) {}

utility::optional<double> Segment3D::SlabAABB(
        const AxisAlignedBoundingBox& box) const {
    /* The segment case of the Line/AABB intersection only allows intersections
     * along the positive direction of the line which are at a distance less
     * than the segment length, in accordance with the semantic meaning of a
     * line segment which is finite.
     *
     * In this case the same conditions apply as with the ray case, but with one
     * additional requirement: t_min must be less than the length of the
     * segment. If t_min were greater than the length of the segment it would
     * indicate that the very earliest intersection along the line direction
     * occurs beyond the endpoint of the segment.
     */
    const auto& t = SlabAABBBase(box);
    double t_min = std::get<0>(t);
    double t_max = std::get<1>(t);

    t_min = std::max(0., t_min);

    if (t_max >= t_min && t_min <= length_) return t_min;
    return {};
}

utility::optional<double> Segment3D::ExactAABB(
        const AxisAlignedBoundingBox& box) const {
    // For a line segment, the result must additionally be less than the
    // overall length of the segment
    auto result = Line3D::ExactAABB(box);
    if (!result.has_value() || result.value() <= length_) return result;

    return {};
}

AxisAlignedBoundingBox Segment3D::GetBoundingBox() const {
    Eigen::Vector3d min{std::min(origin().x(), end_point_.x()),
                        std::min(origin().y(), end_point_.y()),
                        std::min(origin().z(), end_point_.z())};

    Eigen::Vector3d max{std::max(origin().x(), end_point_.x()),
                        std::max(origin().y(), end_point_.y()),
                        std::max(origin().z(), end_point_.z())};
    return {min, max};
}

utility::optional<double> Segment3D::IntersectionParameter(
        const Eigen::Hyperplane<double, 3>& plane) const {
    // On a segment, the intersection parameter must be between zero and the
    // length of the segment
    auto result = Line().intersectionParameter(plane);
    if (!std::isinf(result) && result >= 0 && result <= length_) {
        return result;
    }
    return {};
}

}  // namespace geometry
}  // namespace open3d
