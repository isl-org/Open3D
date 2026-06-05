// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/MinimumBS.h"

#include <algorithm>
#include <cmath>
#include <random>

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

namespace {

// Helper struct using Eigen datatypes on the stack for sphere computation
struct EigenSphere {
    EigenSphere()
        : center_(Eigen::Vector3d::Zero()), radius_(0.0), boundary_() {}

    EigenSphere(const Eigen::Vector3d& c, double r)
        : center_(c), radius_(r), boundary_() {}

    EigenSphere(const Eigen::Vector3d& c, double r,
                const std::vector<Eigen::Vector3d>& b)
        : center_(c), radius_(r), boundary_(b) {}

    bool IsInside(const Eigen::Vector3d& point, double eps = 1e-9) const {
        double dist = (point - center_).norm();
        return dist <= radius_ + eps;
    }

    double Volume() const {
        return 4.0 * M_PI * radius_ * radius_ * radius_ / 3.0;
    }

    Eigen::Vector3d center_;
    double radius_;
    std::vector<Eigen::Vector3d> boundary_;
};

// Compute circumsphere of 2 points
EigenSphere ComputeCircumsphere2(const Eigen::Vector3d& p1,
                                  const Eigen::Vector3d& p2) {
    Eigen::Vector3d center = (p1 + p2) / 2.0;
    double radius = (p2 - p1).norm() / 2.0;
    return EigenSphere(center, radius, std::vector<Eigen::Vector3d>{p1, p2});
}

// Compute circumsphere of 3 points
EigenSphere ComputeCircumsphere3(const Eigen::Vector3d& p1,
                                  const Eigen::Vector3d& p2,
                                  const Eigen::Vector3d& p3) {
    // Vectors from p1 to p2 and p1 to p3
    Eigen::Vector3d v1 = p2 - p1;
    Eigen::Vector3d v2 = p3 - p1;

    // Compute the circumcenter
    // Using the formula for circumcenter of a triangle
    double a = v1.squaredNorm();
    double b = v2.squaredNorm();
    double c = v1.dot(v2);

    double denom = 2.0 * (a * b - c * c);

    // Degenerate case: collinear points
    if (std::abs(denom) < 1e-12) {
        // Fall back to computing minimum sphere from 2 points
        Eigen::Vector3d p_far;
        double max_dist = 0.0;
        double d12 = (p2 - p1).norm();
        double d13 = (p3 - p1).norm();
        double d23 = (p3 - p2).norm();

        if (d12 >= d13 && d12 >= d23) {
            return ComputeCircumsphere2(p1, p2);
        } else if (d13 >= d12 && d13 >= d23) {
            return ComputeCircumsphere2(p1, p3);
        } else {
            return ComputeCircumsphere2(p2, p3);
        }
    }

    double s = (b - c) / denom;
    double t = (a - c) / denom;

    Eigen::Vector3d center = p1 + s * v1 + t * v2;
    double radius = (center - p1).norm();

    return EigenSphere(center, radius, std::vector<Eigen::Vector3d>{p1, p2, p3});
}

// Compute circumsphere of 4 points (minimal sphere through 4 points)
EigenSphere ComputeCircumsphere4(const Eigen::Vector3d& p1,
                                  const Eigen::Vector3d& p2,
                                  const Eigen::Vector3d& p3,
                                  const Eigen::Vector3d& p4) {
    // Solve the linear system to find the circumcenter
    // |p1.x p1.y p1.z 1| |x|   |p1.x^2 + p1.y^2 + p1.z^2|
    // |p2.x p2.y p2.z 1| |y| = |p2.x^2 + p2.y^2 + p2.z^2|
    // |p3.x p3.y p3.z 1| |z|   |p3.x^2 + p3.y^2 + p3.z^2|
    // |p4.x p4.y p4.z 1| |r|   |p4.x^2 + p4.y^2 + p4.z^2|

    Eigen::Matrix4d A;
    Eigen::Vector4d b;

    A.row(0) << p1.x(), p1.y(), p1.z(), 1.0;
    A.row(1) << p2.x(), p2.y(), p2.z(), 1.0;
    A.row(2) << p3.x(), p3.y(), p3.z(), 1.0;
    A.row(3) << p4.x(), p4.y(), p4.z(), 1.0;

    b(0) = p1.squaredNorm();
    b(1) = p2.squaredNorm();
    b(2) = p3.squaredNorm();
    b(3) = p4.squaredNorm();

    // Solve Ax = b
    Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

    Eigen::Vector3d center(x(0) / 2.0, x(1) / 2.0, x(2) / 2.0);
    double radius_sq = center.squaredNorm() - x(3) / 2.0;
    double radius = std::max(0.0, std::sqrt(radius_sq));
    return EigenSphere(center, radius,
                       std::vector<Eigen::Vector3d>{p1, p2, p3, p4});
}

// Recursive implementation of Welzl's algorithm
EigenSphere WelzlRecursive(
        std::vector<Eigen::Vector3d>& points,
        std::vector<Eigen::Vector3d> boundary,
        size_t n) {
    // Base cases
    if (n == 0 || boundary.size() == 4) {
        if (boundary.empty()) {
            return EigenSphere(Eigen::Vector3d::Zero(), 0.0, boundary);
        } else if (boundary.size() == 1) {
            return EigenSphere(boundary[0], 0.0, boundary);
        } else if (boundary.size() == 2) {
            return ComputeCircumsphere2(boundary[0], boundary[1]);
        } else if (boundary.size() == 3) {
            return ComputeCircumsphere3(boundary[0], boundary[1], boundary[2]);
        } else {  // boundary.size() == 4
            return ComputeCircumsphere4(boundary[0], boundary[1], boundary[2],
                                       boundary[3]);
        }
    }

    // Pick a random point
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    size_t idx = dis(gen);

    Eigen::Vector3d p = points[idx];

    // Swap the selected point to the end and reduce n
    std::swap(points[idx], points[n - 1]);

    // Recursively compute the minimum sphere for n-1 points
    EigenSphere sphere = WelzlRecursive(points, boundary, n - 1);

    // If p is inside the sphere, we're done
    if (sphere.IsInside(p)) {
        // Swap back
        std::swap(points[idx], points[n - 1]);
        return sphere;
    }

    // Otherwise, p must be on the boundary
    boundary.push_back(p);
    sphere = WelzlRecursive(points, boundary, n - 1);

    // Swap back
    std::swap(points[idx], points[n - 1]);

    return sphere;
}

// Iterative (non-recursive) implementation of Welzl's algorithm.
// Uses a randomized shuffle and nested loops to enforce the boundary
// (points on the sphere) up to size 4.
EigenSphere WelzlIterative(std::vector<Eigen::Vector3d>& points, size_t n) {
    if (n == 0) return EigenSphere(Eigen::Vector3d::Zero(), 0.0);

    // Random shuffle the first n points for randomized algorithm
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(n), gen);

    EigenSphere sphere(Eigen::Vector3d::Zero(), -1.0);

    auto make2 = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return ComputeCircumsphere2(a, b);
    };
    auto make3 = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
        return ComputeCircumsphere3(a, b, c);
    };
    auto make4 = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c, const Eigen::Vector3d& d) {
        return ComputeCircumsphere4(a, b, c, d);
    };

    for (size_t i = 0; i < n; ++i) {
        const Eigen::Vector3d& p_i = points[i];

        if (sphere.radius_ >= 0 && sphere.IsInside(p_i)) continue;

        // Sphere must include p_i. Start with trivial sphere at p_i and record boundary.
        sphere = EigenSphere(p_i, 0.0, std::vector<Eigen::Vector3d>{p_i});

        for (size_t j = 0; j < i; ++j) {
            const Eigen::Vector3d& p_j = points[j];
            if (sphere.IsInside(p_j)) continue;

            // Sphere must include p_j and p_i
            sphere = make2(p_j, p_i);

            for (size_t k = 0; k < j; ++k) {
                const Eigen::Vector3d& p_k = points[k];
                if (sphere.IsInside(p_k)) continue;

                // Sphere must include p_k, p_j, p_i
                sphere = make3(p_k, p_j, p_i);

                for (size_t l = 0; l < k; ++l) {
                    const Eigen::Vector3d& p_l = points[l];
                    if (sphere.IsInside(p_l)) continue;

                    // Sphere must include p_l, p_k, p_j, p_i
                    sphere = make4(p_l, p_k, p_j, p_i);
                }
            }
        }
    }

    // If we never set a sphere (shouldn't happen for n>0), fallback
    if (sphere.radius_ < 0) return EigenSphere(points[0], 0.0, std::vector<Eigen::Vector3d>{points[0]});

    return sphere;
}

}  // namespace

open3d::geometry::BoundingSphere ComputeMinimumBSWelzl(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    if (points.empty()) {
        utility::LogError("Input point set is empty.");
    }
    if (static_cast<int>(points.size()) < 2) {
        utility::LogError("Input point set has less than 2 points.");
    }

    // ------------------------------------------------------------
    // 0) Compute the convex hull of the input point cloud using legacy API
    // (Eigen types, CPU). This reduces the input size for Welzl's algorithm.
    // ------------------------------------------------------------
    open3d::geometry::PointCloud pcd;
    pcd.points_ = points;
    auto [hull_mesh, hull_indices] = pcd.ComputeConvexHull(/*robust=*/robust);
    if (hull_mesh->vertices_.empty()) {
        utility::LogWarning("Failed to compute convex hull.");
        return open3d::geometry::BoundingSphere();
    }

    // Get convex hull vertices
    const auto& hull_v = hull_mesh->vertices_;
    int num_vertices = static_cast<int>(hull_v.size());

    // Handle degenerate planar cases up front.
    if (num_vertices < 4) {
        utility::LogWarning("Convex hull is degenerate.");
        return open3d::geometry::BoundingSphere();
    }

    // Make a copy for processing (Welzl modifies the array)
    std::vector<Eigen::Vector3d> pts = hull_v;
    std::vector<Eigen::Vector3d> boundary;

    // Run Welzl's algorithm on convex hull vertices
    EigenSphere es = WelzlRecursive(pts, boundary, pts.size());
    // EigenSphere es = WelzlIterative(pts, pts.size());

    open3d::geometry::BoundingSphere sphere(es.center_, es.radius_);
    return sphere;
}

BoundingSphere ComputeMinimumBSWelzl(const core::Tensor& points, bool robust) {
    core::AssertTensorShape(points, {std::nullopt, 3});
    core::AssertTensorDtypes(points, {core::Float32, core::Float64});
    if (points.GetShape(0) < 1) {
        utility::LogError("Input point set must have at least 1 point.");
    }

    // Convert tensor → Eigen (Float64, CPU)
    const std::vector<Eigen::Vector3d> eigen_points =
            core::eigen_converter::TensorToEigenVector3dVector(
                    points.To(core::Device("CPU:0"), core::Float64));

    // Run Eigen-native core (returns result in Float64, CPU)
    open3d::geometry::BoundingSphere legacy_mbs = ComputeMinimumBSWelzl(
        eigen_points, robust);

    // Convert result back to the caller's dtype and device
    return open3d::t::geometry::BoundingSphere::FromLegacy(legacy_mbs, 
        points.GetDtype(), 
        points.GetDevice());
}

}  // namespace bounding_sphere
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
