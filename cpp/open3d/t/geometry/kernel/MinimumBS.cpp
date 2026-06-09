// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/MinimumBS.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace bounding_sphere {

// ============================================================================
// MINIMUM BOUNDING SPHERE COMPUTATION
// ============================================================================
//
// Implementation of Welzl's randomized linear-time algorithm for the smallest
// enclosing sphere in 3D.
//
// The implementation explicitly separates:
//
//   • Full-rank 3D geometry (true tetrahedral configurations)
//   • Degenerate lower-dimensional geometry (coplanar / collinear sets)
//
// in order to achieve both numerical robustness and efficient geometric solves.
//
// Key properties:
// - Expected O(n) runtime with linear space usage
// - Robust handling of coplanar and collinear point clouds
// - Convex hull reduction before Welzl recursion
// - Single global shuffle for deterministic recursion order
//
// ============================================================================
// ARCHITECTURE
// ============================================================================
//
// The solver uses a deliberate dual-path design:
//
// -----------------------------------------------------------------------------
// PATH 1: Full-Rank 3D Solver (rank == 3)
// -----------------------------------------------------------------------------
//
// If the convex hull spans all 3 spatial dimensions, the algorithm uses
// specialized analytic circumsphere routines operating directly in R^3:
//
//     WelzlRecursive()
//         -> SphereFromBoundary()
//             -> ComputeCircumsphere2()
//             -> ComputeCircumsphere3()
//             -> ComputeCircumsphere4()
//
// These routines use explicit geometric constructions:
//
// - 2 points -> midpoint sphere
// - 3 points -> triangle circumcircle in 3D
// - 4 points -> tetrahedral circumsphere
//
// Advantages:
// - Fast fixed-size computations
// - Geometry-aware analytic formulas
// - Strong numerical behavior for true tetrahedral geometry
// - Clear geometric interpretation of boundary spheres
//
// This path is intentionally optimized for non-degenerate 3D point sets.
//
// -----------------------------------------------------------------------------
// PATH 2: Intrinsic-Dimension Solver (rank < 3)
// -----------------------------------------------------------------------------
//
// Degenerate point clouds are not solved using tetrahedral geometry.
//
// Instead:
//
//   1. Compute intrinsic rank using SVD
//   2. Construct an orthonormal basis for the intrinsic subspace
//   3. Project points into intrinsic coordinates
//   4. Run generic N-D Welzl recursion:
//
//          WelzlNDRecursive()
//              -> FitCircumsphereND()
//
//   5. Lift the center back into R^3
//
// This projection-based formulation avoids unstable or ill-defined 3D
// circumsphere computations for degenerate configurations.
//
// Examples:
//
// - Collinear 3D points  -> solved as a 1D interval problem
// - Coplanar 3D points   -> solved as a 2D circumcircle problem
//
// -----------------------------------------------------------------------------
// DESIGN RATIONALE
// -----------------------------------------------------------------------------
//
// The implementation intentionally avoids forcing all configurations through a
// single generic 3D circumsphere solver.
//
// Instead, it combines:
//
//   • Specialized analytic 3D geometry for full-rank tetrahedral problems
//   • Generic intrinsic-dimension solvers for degenerate geometry
//
// This hybrid architecture provides:
//
// - Stable behavior for nearly degenerate configurations
// - Efficient exact solves for small 3D boundary sets
// - Cleaner geometric reasoning
// - Improved numerical robustness
//
// The result is a minimum bounding sphere implementation capable of handling:
//
// - General 3D point sets
// - Coplanar convex hulls
// - Collinear convex hulls
// - Nearly degenerate inputs
//
// while preserving the expected linear-time behavior of Welzl's algorithm.
// ============================================================================

namespace {

// Tolerance for rank detection and geometric degeneracy tests
// Machine epsilon is ~1e-16 for float64, so 1e-12 is safely above noise
constexpr double kEps = 1e-12;
constexpr double kRankTol = 1e-9;

// ----------------------------------------------------------------------------
// Helper sphere type
// ----------------------------------------------------------------------------

// Minimal sphere representation: center, radius, and boundary points.
// The boundary_ vector stores the 1-4 points that define the sphere's
// minimal enclosing property (i.e., at least 2 boundary points on the
// sphere surface for valid small-dimension enclosing spheres).
struct EigenSphere {
    EigenSphere()
        : center_(Eigen::Vector3d::Zero()),
          radius_(0.0),
          boundary_() {}

    EigenSphere(const Eigen::Vector3d& c, double r)
        : center_(c),
          radius_(r),
          boundary_() {}

    EigenSphere(const Eigen::Vector3d& c,
                double r,
                const std::vector<Eigen::Vector3d>& b)
        : center_(c),
          radius_(r),
          boundary_(b) {}

    // Check if point is inside or on the sphere surface (within tolerance eps).
    // Uses squared-norm to avoid expensive sqrt operations.
    bool IsInside(const Eigen::Vector3d& point,
                  double eps = kEps) const {
        return (point - center_).squaredNorm() <=
               (radius_ + eps) * (radius_ + eps);
    }

    // Compute sphere volume: (4/3) * π * r³
    double Volume() const {
        return (4.0 / 3.0) * std::acos(-1.0) *
               radius_ * radius_ * radius_;
    }

    Eigen::Vector3d center_;
    double radius_;
    std::vector<Eigen::Vector3d> boundary_;
};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

// Verify that all points are contained within (or on) the sphere.
// Used as a sanity check for circumsphere solvers and fallback selection.
bool ContainsAll(const EigenSphere& sphere,
                 const std::vector<Eigen::Vector3d>& pts,
                 double eps = kEps) {
    for (const auto& p : pts) {
        if (!sphere.IsInside(p, eps)) {
            return false;
        }
    }
    return true;
}

static std::pair<Eigen::VectorXd, double> FitCircumsphereND(
        const Eigen::MatrixXd &points);

static EigenSphere ComputeCircumsphere2(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2);

static EigenSphere ComputeCircumsphere3(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3);

static EigenSphere ComputeCircumsphere4(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3,
        const Eigen::Vector3d& p4);

static int ComputeIntrinsicRank(
        const Eigen::MatrixXd &mat,
        double tol = kRankTol);

static const std::array<std::array<int, 3>, 4> kSubTriangles = {{
    {0, 1, 2},
    {0, 1, 3},
    {0, 2, 3},
    {1, 2, 3}
}};

static const std::array<std::array<int, 2>, 6> kSubPairs = {{
    {0, 1},
    {0, 2},
    {0, 3},
    {1, 2},
    {1, 3},
    {2, 3}
}};

static EigenSphere ComputeBestLowerDimensionalSphere(
        const std::vector<Eigen::Vector3d>& pts) {
    EigenSphere best;
    best.radius_ = std::numeric_limits<double>::infinity();

    for (const auto& c : kSubTriangles) {
        EigenSphere s = ComputeCircumsphere3(
                pts[c[0]],
                pts[c[1]],
                pts[c[2]]);
        if (ContainsAll(s, pts, 1e-9) && s.radius_ < best.radius_) {
            best = s;
        }
    }

    if (best.radius_ < std::numeric_limits<double>::infinity()) {
        return best;
    }

    for (const auto& c : kSubPairs) {
        EigenSphere s = ComputeCircumsphere2(
                pts[c[0]],
                pts[c[1]]);
        if (ContainsAll(s, pts, 1e-9) && s.radius_ < best.radius_) {
            best = s;
        }
    }

    if (best.radius_ < std::numeric_limits<double>::infinity()) {
        return best;
    }

    EigenSphere fallback = ComputeCircumsphere2(pts[0], pts[1]);
    for (const auto& c : kSubPairs) {
        EigenSphere s = ComputeCircumsphere2(
                pts[c[0]],
                pts[c[1]]);
        if (s.radius_ > fallback.radius_) {
            fallback = s;
        }
    }
    return fallback;
}

static std::pair<Eigen::VectorXd, double> WelzlNDRecursive(
        std::vector<Eigen::VectorXd> &points,
        std::vector<Eigen::VectorXd> boundary,
        size_t n,
        double eps);

static int ComputeIntrinsicRank(
        const Eigen::VectorXd &svals,
        double tol = kRankTol) {
    int rank = 0;
    for (int i = 0; i < static_cast<int>(svals.size()); ++i) {
        if (svals(i) > tol) {
            ++rank;
        }
    }
    return rank;
}

static int ComputeIntrinsicRank(
        const Eigen::MatrixXd &mat,
        double tol) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return ComputeIntrinsicRank(svd.singularValues(), tol);
}

// ----------------------------------------------------------------------------
// 2-point sphere
// ----------------------------------------------------------------------------

EigenSphere ComputeCircumsphere2(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2) {

    Eigen::Vector3d center = 0.5 * (p1 + p2);
    double radius = 0.5 * (p2 - p1).norm();

    return EigenSphere(center, radius, {p1, p2});
}

// ----------------------------------------------------------------------------
// 3-point sphere
// Robust against collinear points
// ----------------------------------------------------------------------------

EigenSphere ComputeCircumsphere3(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3) {

    Eigen::Vector3d a = p2 - p1;
    Eigen::Vector3d b = p3 - p1;

    Eigen::Vector3d n = a.cross(b);
    double n2 = n.squaredNorm();

    // ------------------------------------------------------------------------
    // Collinear fallback
    // ------------------------------------------------------------------------

    if (n2 < kEps) {

        double d12 = (p2 - p1).squaredNorm();
        double d13 = (p3 - p1).squaredNorm();
        double d23 = (p3 - p2).squaredNorm();

        if (d12 >= d13 && d12 >= d23) {
            return ComputeCircumsphere2(p1, p2);
        } else if (d13 >= d12 && d13 >= d23) {
            return ComputeCircumsphere2(p1, p3);
        } else {
            return ComputeCircumsphere2(p2, p3);
        }
    }

    // ------------------------------------------------------------------------
    // Analytic circumcenter for 3 points in 3D.
    // ------------------------------------------------------------------------

    double a2 = a.squaredNorm();
    double b2 = b.squaredNorm();
    Eigen::Vector3d center = p1 + ((b * a2 - a * b2).cross(n)) / (2.0 * n2);
    double radius = (center - p1).norm();

    return EigenSphere(center, radius, {p1, p2, p3});
}

// ----------------------------------------------------------------------------
// 4-point sphere
// Robust against coplanar tetrahedra
// ----------------------------------------------------------------------------

EigenSphere ComputeCircumsphere4(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3,
        const Eigen::Vector3d& p4) {

    Eigen::Matrix3d M;
    M.row(0) = p2 - p1;
    M.row(1) = p3 - p1;
    M.row(2) = p4 - p1;

    const int rank = ComputeIntrinsicRank(Eigen::MatrixXd(M));
    std::vector<Eigen::Vector3d> pts = {p1, p2, p3, p4};
    if (rank < 3) {
        return ComputeBestLowerDimensionalSphere(pts);
    }

    // ------------------------------------------------------------------------
    // Proper tetrahedral circumsphere
    // ------------------------------------------------------------------------

    // Circumsphere linear system:
    //     A c = b
    // where A = 2*(p_i - p_1)
    Eigen::Matrix3d A = 2.0 * M;

    Eigen::Vector3d b;
    b(0) = p2.squaredNorm() - p1.squaredNorm();
    b(1) = p3.squaredNorm() - p1.squaredNorm();
    b(2) = p4.squaredNorm() - p1.squaredNorm();

    // Find the unique point in 3D that is equidistant from the 
    // four tetrahedron vertices -> center of the circumsphere.
    Eigen::Vector3d center = A.jacobiSvd(
        Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
    double radius = (center - p1).norm();

    EigenSphere sphere(center, radius, pts);
    if (!ContainsAll(sphere, pts, 1e-9)) {
        return ComputeBestLowerDimensionalSphere(pts);
    }

    return sphere;
}

// ----------------------------------------------------------------------------
// Boundary sphere
// ----------------------------------------------------------------------------

EigenSphere SphereFromBoundary(
        const std::vector<Eigen::Vector3d>& boundary) {

    switch (boundary.size()) {

        case 0:
            return EigenSphere(
                    Eigen::Vector3d::Zero(),
                    0.0,
                    boundary);

        case 1:
            return EigenSphere(
                    boundary[0],
                    0.0,
                    boundary);

        case 2:
            return ComputeCircumsphere2(
                    boundary[0],
                    boundary[1]);

        case 3:
            return ComputeCircumsphere3(
                    boundary[0],
                    boundary[1],
                    boundary[2]);

        case 4:
            return ComputeCircumsphere4(
                    boundary[0],
                    boundary[1],
                    boundary[2],
                    boundary[3]);

        default:
            utility::LogError(
                    "Boundary size > 4 is invalid.");
    }
}

// ----------------------------------------------------------------------------
// Recursive Welzl
// IMPORTANT:
// - shuffle once globally
// ----------------------------------------------------------------------------

EigenSphere WelzlRecursive(
        std::vector<Eigen::Vector3d>& points,
        std::vector<Eigen::Vector3d> boundary,
        size_t n) {

    if (n == 0 || boundary.size() == 4) {
        return SphereFromBoundary(boundary);
    }

    const Eigen::Vector3d p = points[n - 1];

    EigenSphere sphere =
            WelzlRecursive(
                    points,
                    boundary,
                    n - 1);

    if (sphere.IsInside(p)) {
        return sphere;
    }

    boundary.push_back(p);

    return WelzlRecursive(
            points,
            boundary,
            n - 1);
}


// ----------------------------------------------------------------------------
// Generic N-D Welzl (small helper for projecting degenerate 3D sets)
// Supports D = 1 or 2 (used when points lie in a lower-dimensional subspace).
// ----------------------------------------------------------------------------

// Generic circumsphere fit for <= D+1 points in R^D.
// Used by the intrinsic-dimension Welzl solver.
// Typically exercised for D=1 or D=2 degenerate 3D point sets,
// but mathematically valid for arbitrary small D.
static std::pair<Eigen::VectorXd, double> FitCircumsphereND(
        const Eigen::MatrixXd &points) {
    const int M = static_cast<int>(points.rows());
    const int D = static_cast<int>(points.cols());

    if (M == 0) {
        return {Eigen::VectorXd::Constant(D, std::numeric_limits<double>::quiet_NaN()),
                std::numeric_limits<double>::quiet_NaN()};
    }
    if (M == 1) {
        return {points.row(0).transpose(), 0.0};
    }

    if (M == 2) {
        Eigen::VectorXd c = 0.5 * (points.row(0).transpose() + points.row(1).transpose());
        double r = (points.row(0).transpose() - c).norm();
        return {c, r};
    }

    // General small linear solve: 2*(p_i - p_0) · c = |p_i|^2 - |p_0|^2
    Eigen::MatrixXd V = points.block(1, 0, M - 1, D) -
                       points.row(0).replicate(M - 1, 1);
    Eigen::MatrixXd A = 2.0 * V;
    Eigen::VectorXd b(M - 1);
    for (int i = 1; i < M; ++i) {
        b(i - 1) = points.row(i).squaredNorm() - points.row(0).squaredNorm();
    }

    Eigen::VectorXd center;
    if (A.fullPivLu().rank() == D) {
        center = A.colPivHouseholderQr().solve(b);
    } else {
        center = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    }

    double radius = (points.row(0).transpose() - center).norm();
    return {center, radius};
}

// Recursive ND Welzl (points passed as Eigen vectors)
static std::pair<Eigen::VectorXd, double> WelzlNDRecursive(
        std::vector<Eigen::VectorXd> &points,
        std::vector<Eigen::VectorXd> boundary,
        size_t n,
        double eps = kEps) {

    const int D = static_cast<int>(points[0].size());

    if (n == 0 || boundary.size() == static_cast<size_t>(D + 1)) {
        Eigen::MatrixXd B_mat(static_cast<int>(boundary.size()), D);
        for (int i = 0; i < static_cast<int>(boundary.size()); ++i) {
            B_mat.row(i) = boundary[i].transpose();
        }
        auto [c, r] = FitCircumsphereND(B_mat);
        return {c, r};
    }

    Eigen::VectorXd p = points[n - 1];

    auto [c, r] = WelzlNDRecursive(points, boundary, n - 1, eps);

    if (std::isnan(r) == false && (p - c).squaredNorm() <= (r + eps) * (r + eps)) {
        return {c, r};
    }

    boundary.push_back(p);
    return WelzlNDRecursive(points, boundary, n - 1, eps);
}


}  // namespace

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

open3d::geometry::BoundingSphere ComputeMinimumBSWelzl(
        const std::vector<Eigen::Vector3d>& points,
        bool robust) {

    if (points.empty()) {
        utility::LogError(
                "Input point set is empty.");
    }

    if (static_cast<int>(points.size()) < 2) {
        utility::LogError(
                "Input point set has less than 2 points.");
    }

    // ------------------------------------------------------------------------
    // Convex hull reduction
    // ------------------------------------------------------------------------

    open3d::geometry::PointCloud pcd;
    pcd.points_ = points;

    auto [hull_mesh, hull_indices] =
            pcd.ComputeConvexHull(robust);

    if (hull_mesh->vertices_.empty()) {
        utility::LogWarning(
                "Failed to compute convex hull.");

        return open3d::geometry::BoundingSphere();
    }

    std::vector<Eigen::Vector3d> pts =
            hull_mesh->vertices_;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    // ========================================================================
    // DIMENSIONALITY DETECTION AND DUAL-PATH ROUTING
    // ========================================================================
    //   1. Center points at first point (affine origin)
    //   2. Compute SVD to find singular values
    //   3. Count non-zero singular values to determine rank
    //   4. Route to 3D Welzl (full rank) or N-D Welzl + projection (lower rank)
    //
    // Why: Coplanar (rank 2) or collinear (rank 1) point sets are degenerate
    // in 3D but well-defined in their intrinsic dimension. The projection
    // approach solves the problem in the intrinsic subspace, then lifts the
    // result back to 3D, ensuring numeric stability.
    // ========================================================================

    // --- Step 1: Convert to matrix format and center at first point ---
    const int M = static_cast<int>(pts.size());
    Eigen::MatrixXd Pmat(M, 3);
    for (int i = 0; i < M; ++i) {
        Pmat.row(i) = pts[i].transpose();
    }

    // Center by subtracting first point (affine subspace origin)
    Eigen::RowVector3d offset = Pmat.row(0);
    Eigen::MatrixXd centered = Pmat.rowwise() - offset;

    // --- Step 2: Compute intrinsic rank using a shared helper ---
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const int rank = ComputeIntrinsicRank(svd.singularValues());

    // --- Step 4: Route based on rank ---
    if (rank < 3 && rank > 0) {
        // LOWER-DIMENSIONAL PATH: Points lie in a rank-D subspace (D < 3)
        // Strategy: Project to intrinsic subspace, solve there, lift back
        
        // Extract basis: top 'rank' right singular vectors (columns of V)
        // These span the subspace containing the points
        Eigen::MatrixXd V = svd.matrixV();           // 3x3, orthonormal columns
        Eigen::MatrixXd basis = V.transpose().block(0, 0, rank, 3); // rank x 3

        // Project centered points onto basis: X_sub = centered * basis^T
        // Result: M x rank matrix of coordinates in intrinsic subspace
        Eigen::MatrixXd X_sub = centered * basis.transpose();

        // Convert to vector of column vectors for WelzlNDRecursive
        std::vector<Eigen::VectorXd> Psub;
        Psub.reserve(M);
        for (int i = 0; i < M; ++i) {
            Psub.push_back(X_sub.row(i).transpose());
        }

        // Shuffle for randomized algorithm
        std::shuffle(Psub.begin(), Psub.end(), gen);

        // Run Welzl algorithm in the intrinsic dimension
        auto [c_sub, r_sub] = WelzlNDRecursive(Psub, {}, Psub.size());

        // LIFT: Transform center from intrinsic coordinates back to 3D
        // center_3d = offset + basis^T * c_sub
        // (offset accounts for the translation in Step 1)
        Eigen::Vector3d center_nd = offset.transpose();
        if (rank > 0) {
            center_nd += basis.transpose() * c_sub;
        }

        return open3d::geometry::BoundingSphere(center_nd, r_sub);
    }

    // FULL-RANK PATH: Points span all 3 dimensions
    // Run standard 3D Welzl directly
    std::shuffle(pts.begin(), pts.end(), gen);

    EigenSphere bs = WelzlRecursive(pts, {}, pts.size());

    return open3d::geometry::BoundingSphere(bs.center_, bs.radius_);
}

BoundingSphere ComputeMinimumBSWelzl(
        const core::Tensor& points,
        bool robust) {

    core::AssertTensorShape(points, {std::nullopt, 3});

    core::AssertTensorDtypes(points, {core::Float32, core::Float64});

    if (points.GetShape(0) < 1) {
        utility::LogError(
                "Input point set must have at least 1 point.");
    }

    const std::vector<Eigen::Vector3d> eigen_points =
            core::eigen_converter::
                    TensorToEigenVector3dVector(
                            points.To(
                                    core::Device("CPU:0"),
                                    core::Float64));

    open3d::geometry::BoundingSphere sphere =
            ComputeMinimumBSWelzl(
                    eigen_points,
                    robust);

    return open3d::t::geometry::BoundingSphere::
            FromLegacy(
                    sphere,
                    points.GetDtype(),
                    points.GetDevice());
}


// // Approximate bounding sphere, based on "An Efficient Bounding Sphere" by Jack Ritter, from "Graphics Gems"
// Sphere nv::approximateSphere_Ritter(const Vector3 * pointArray, const uint pointCount) {
//     nvDebugCheck(pointArray != NULL);
//     nvDebugCheck(pointCount > 0);

//     Vector3 xmin, xmax, ymin, ymax, zmin, zmax;

//     xmin = xmax = ymin = ymax = zmin = zmax = pointArray[0];

//     // FIRST PASS: find 6 minima/maxima points
//     xmin.x = ymin.y = zmin.z = FLT_MAX;
//     xmax.x = ymax.y = zmax.z = -FLT_MAX;

//     for (uint i = 0; i < pointCount; i++) {
//         const Vector3 & p = pointArray[i];
//         if (p.x < xmin.x) xmin = p;
//         if (p.x > xmax.x) xmax = p;
//         if (p.y < ymin.y) ymin = p;
//         if (p.y > ymax.y) ymax = p;
//         if (p.z < zmin.z) zmin = p;
//         if (p.z > zmax.z) zmax = p;
//     }

//     float xspan = lengthSquared(xmax - xmin);
//     float yspan = lengthSquared(ymax - ymin);
//     float zspan = lengthSquared(zmax - zmin);

//     // Set points dia1 & dia2 to the maximally separated pair.
//     Vector3 dia1 = xmin; 
//     Vector3 dia2 = xmax;
//     float maxspan = xspan;
//     if (yspan > maxspan) {
//         maxspan = yspan;
//         dia1 = ymin;
//         dia2 = ymax;
//     }
//     if (zspan > maxspan) {
//         dia1 = zmin;
//         dia2 = zmax;
//     }

//     // |dia1-dia2| is a diameter of initial sphere

//     // calc initial center
//     Sphere sphere;
//     sphere.center = (dia1 + dia2) / 2.0f;

//     // calculate initial radius**2 and radius
//     float rad_sq = lengthSquared(dia2 - sphere.center);
//     sphere.radius = sqrtf(rad_sq);


//     // SECOND PASS: increment current sphere
//     for (uint i = 0; i < pointCount; i++) {
//         const Vector3 & p = pointArray[i];

//         float old_to_p_sq = lengthSquared(p - sphere.center);

//         if (old_to_p_sq > rad_sq) {    // do r**2 test first
            
//             // this point is outside of current sphere
//             float old_to_p = sqrtf(old_to_p_sq);

//             // calc radius of new sphere
//             sphere.radius = (sphere.radius + old_to_p) / 2.0f;
//             rad_sq = sphere.radius * sphere.radius;     // for next r**2 compare
            
//             float old_to_new = old_to_p - sphere.radius;

//             // calc center of new sphere
//             sphere.center = (sphere.radius * sphere.center + old_to_new * p) / old_to_p;
//         }
//     }

//     nvDebugCheck(allInside(sphere, pointArray, pointCount));

//     return sphere;
// }

EigenSphere RitterBSApproximation(
        const std::vector<Eigen::Vector3d>& points,
        size_t n) {

            Eigen::Vector3d xmin, xmax, ymin, ymax, zmin, zmax;
            xmin = xmax = ymin = ymax = zmin = zmax = points[0];

            // FIRST PASS: find 6 minima/maxima points
            for (size_t i = 0; i < n; i++) {
                const Eigen::Vector3d & p = points[i];
                if (p.x() < xmin.x()) xmin = p;
                if (p.x() > xmax.x()) xmax = p;
                if (p.y() < ymin.y()) ymin = p;
                if (p.y() > ymax.y()) ymax = p;
                if (p.z() < zmin.z()) zmin = p;
                if (p.z() > zmax.z()) zmax = p;
            }

            double x_span_sq = (xmax - xmin).squaredNorm();
            double y_span_sq = (ymax - ymin).squaredNorm();
            double z_span_sq = (zmax - zmin).squaredNorm();

            // Set points extreme_point1 & extreme_point2 to 
            // the maximally separated pair.
            Eigen::Vector3d extreme_point1 = xmin;
            Eigen::Vector3d extreme_point2 = xmax;
            double max_span_sq = x_span_sq;
            if (y_span_sq > max_span_sq) {
                max_span_sq = y_span_sq;
                extreme_point1 = ymin;
                extreme_point2 = ymax;
            }
            if (z_span_sq > max_span_sq) {
                extreme_point1 = zmin;
                extreme_point2 = zmax;
            }

            // Initial sphere from extrema pair
            Eigen::Vector3d center = 0.5 * (extreme_point1 + extreme_point2);
            double radius = (extreme_point2 - extreme_point1).norm() / 2.0;

            EigenSphere sphere(center, radius);

            // SECOND PASS: increment current sphere
            for (size_t i = 0; i < n; i++) {
                const Eigen::Vector3d & p = points[i];
                
                if (!sphere.IsInside(p)) {
                    
                    double center_to_point = (p - center).norm();

                    // update radius
                    radius = 0.5 * (radius + center_to_point);

                    double center_shift = center_to_point - radius;
                    
                    // update center
                    center = (radius * center + center_shift * p) / 
                                center_to_point;
                    
                    // update sphere for next iteration
                    sphere = EigenSphere(center, radius);
                }

            }

            return sphere;

}

open3d::geometry::BoundingSphere ComputeApproximateBSRitter(
        const std::vector<Eigen::Vector3d>& points) {

    if (points.empty()) {
        utility::LogError(
                "Input point set is empty.");
    }

    if (static_cast<int>(points.size()) < 2) {
        utility::LogError(
                "Input point set has less than 2 points.");
    }
    
    EigenSphere bs = RitterBSApproximation(points, points.size());

    return open3d::geometry::BoundingSphere(bs.center_, bs.radius_);
}

BoundingSphere ComputeApproximateBSRitter(
        const core::Tensor& points) {

    core::AssertTensorShape(points, {std::nullopt, 3});

    core::AssertTensorDtypes(points, {core::Float32, core::Float64});

    if (points.GetShape(0) < 1) {
        utility::LogError(
                "Input point set must have at least 1 point.");
    }

    const std::vector<Eigen::Vector3d> eigen_points =
            core::eigen_converter::
                    TensorToEigenVector3dVector(
                            points.To(
                                    core::Device("CPU:0"),
                                    core::Float64));

    open3d::geometry::BoundingSphere sphere =
            ComputeApproximateBSRitter(
                    eigen_points);

    return open3d::t::geometry::BoundingSphere::
            FromLegacy(
                    sphere,
                    points.GetDtype(),
                    points.GetDevice());
}

}  // namespace bounding_sphere
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
