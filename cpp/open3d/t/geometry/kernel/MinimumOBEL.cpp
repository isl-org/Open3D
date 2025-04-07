// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/MinimumOBEL.h"
#include <Eigen/Eigenvalues>
#include "open3d/core/EigenConverter.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/t/geometry/PointCloud.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace minimum_obel {

namespace {

OrientedBoundingEllipsoid ComputeMinimumOBELApprox(const core::Tensor& points,
                                                   bool robust) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    if (points.GetShape(0) == 0) {
        utility::LogError("Input point set is empty.");
        return OrientedBoundingEllipsoid();
    }
    if (points.GetShape(0) < 4) {
        utility::LogError("Input point set has less than 4 points.");
        return OrientedBoundingEllipsoid();
    }

    PointCloud pcd(points);
    auto hull_mesh = pcd.ComputeConvexHull(robust);
    if (hull_mesh.GetVertexPositions().NumElements() == 0) {
        utility::LogError("Failed to compute convex hull.");
        return OrientedBoundingEllipsoid();
    }

    // Get convex hull vertices and triangles
    const std::vector<Eigen::Vector3d>& hullV = hull_mesh->vertices_;
    // const std::vector<Eigen::Vector3i>& hullT = hullMesh->triangles_;
    const int numVertices = static_cast<int>(hullV.size());
    // int numTriangles = static_cast<int>(hullT.size());

    auto mapToClosestIdentity = [&](OrientedBoundingEllipsoid& obel) {
        Eigen::Matrix3d& R = obel.R_;
        Eigen::Vector3d& radii = obel.radii_;
        Eigen::Vector3d col[3] = {R.col(0), R.col(1), R.col(2)};
        double best_score = -1e9;
        Eigen::Matrix3d best_R;
        Eigen::Vector3d best_radii;
        // Hard-coded permutations of indices [0,1,2]
        static const std::array<std::array<int, 3>, 6> permutations = {
                {{{0, 1, 2}},
                 {{0, 2, 1}},
                 {{1, 0, 2}},
                 {{1, 2, 0}},
                 {{2, 0, 1}},
                 {{2, 1, 0}}}};

        // Evaluate all 6 permutations × 8 sign flips = 48 candidates
        for (const auto& p : permutations) {
            for (int sign_bits = 0; sign_bits < 8; ++sign_bits) {
                // Derive the sign of each axis from bits (0 => -1, 1 => +1)
                // s0 is bit0, s1 is bit1, s2 is bit2 of sign_bits
                const int s0 = (sign_bits & 1) ? 1 : -1;
                const int s1 = (sign_bits & 2) ? 1 : -1;
                const int s2 = (sign_bits & 4) ? 1 : -1;

                // Construct candidate columns
                Eigen::Vector3d c0 = s0 * col[p[0]];
                Eigen::Vector3d c1 = s1 * col[p[1]];
                Eigen::Vector3d c2 = s2 * col[p[2]];

                // Score: how close are we to the identity?
                // Since e_x = (1,0,0), e_y = (0,1,0), e_z = (0,0,1),
                // we can skip dot products & do c0(0)+c1(1)+c2(2).
                double score = c0(0) + c1(1) + c2(2);

                // If this orientation is better, update the best.
                if (score > best_score) {
                    best_score = score;
                    best_R.col(0) = c0;
                    best_R.col(1) = c1;
                    best_R.col(2) = c2;

                    // Re-permute extents: if the axis p[0] in old frame
                    // now goes to new X, etc.
                    best_radii(0) = radii(p[0]);
                    best_radii(1) = radii(p[1]);
                    best_radii(2) = radii(p[2]);
                }
            }
        }

        // Update the OBB with the best orientation found
        obel.R_ = best_R;
        obel.radii_ = best_radii;
    };

    auto KhachiyanAlgo = [&](const Eigen::MatrixXd& A, double eps, int maxiter,
                             Eigen::MatrixXd& Q, Eigen::VectorXd& c) -> double {
        // Initialize uniform weights: p_i = 1/N for all i
        Eigen::VectorXd p = Eigen::VectorXd::Constant(
                numVertices, 1.0 / static_cast<double>(numVertices));

        // Lift matrix A to Ap by adding a bottom row of ones
        Eigen::MatrixXd Ap(4, numVertices);
        Ap.topRows(3) = A;
        Ap.row(3).setOnes();

        double currentEps = 2.0 * eps;  // Start with a large difference
        int iter = 0;

        // Main iterative loop
        while (iter < maxiter && currentEps > eps) {
            // Compute Λ_p = Ap * diag(p) * Ap^T in a more efficient way
            // ApP = Ap * diag(p)  => shape: (4) x N
            // Then Lambda_p = ApP * Ap^T => shape: (4) x (4)
            Eigen::MatrixXd ApP = Ap * p.asDiagonal();       // (4) x N
            Eigen::Matrix4d LambdaP = ApP * Ap.transpose();  // (4) x (4)

            // Compute inverse of Lambda_p via an LDLT factorization
            // (faster and more numerically stable than .inverse())
            Eigen::LDLT<Eigen::MatrixXd> ldltOfLambdaP(LambdaP);
            if (ldltOfLambdaP.info() != Eigen::Success) {
                throw std::runtime_error(
                        "LDLT decomposition failed. Matrix may be singular.");
            }

            // M = Ap^T * (Lambda_p^{-1} * Ap)
            // We'll do this in steps:
            //    1) X = Lambda_p^{-1} * Ap
            //    2) M = Ap^T * X
            // Dimensions:
            //    Lambda_p^{-1}: (4 x (4)
            //    Ap: (4) x N
            // => X: (4) x N
            // => M: Nx(4) * (4)xN -> NxN
            Eigen::MatrixXd X = ldltOfLambdaP.solve(Ap);  // (4) x N
            Eigen::MatrixXd M = Ap.transpose() * X;       // NxN

            // Find max diagonal element and index
            Eigen::Index maxIndex;
            double maxVal = M.diagonal().maxCoeff(&maxIndex);

            // Compute step size alpha (called step_size here)
            // Formula: alpha = (maxVal - 4) / ((4) * (maxVal - 1))
            const double step_size = (maxVal - 4) / (4 * (maxVal - 1.0));

            // Update weights p
            Eigen::VectorXd newP = (1.0 - step_size) * p;
            newP(maxIndex) += step_size;

            // Compute the change for the stopping criterion
            currentEps = (newP - p).norm();
            p.swap(newP);  // Efficient swap instead of copy

            ++iter;
        }

        // After convergence or reaching max iterations,
        // compute Q and center c for the ellipsoid.

        // 1) PN = A * diag(p) * A^T
        Eigen::MatrixXd AP = A * p.asDiagonal();  // 3 x N
        Eigen::Matrix3d PN = AP * A.transpose();  // 3 x 3

        // 2) M2 = A * p => a 3-dimensional vector
        Eigen::Vector3d M2 = A * p;  // 3 x 1

        // 3) M3 = M2 * M2^T => 3 x 3 outer product
        Eigen::Matrix3d M3 = M2 * M2.transpose();  // 3 x 3

        // 4) Invert (PN - M3) via LDLT
        Eigen::Matrix3d toInvert = PN - M3;  // 3 x 3
        Eigen::LDLT<Eigen::Matrix3d> ldltOfToInvert(toInvert);
        if (ldltOfToInvert.info() != Eigen::Success) {
            throw std::runtime_error(
                    "LDLT decomposition failed in final step.");
        }

        // Q = (toInvert)^{-1} / 3   => shape matrix of the ellipsoid
        // c = A * p                 => center of the ellipsoid
        Q = ldltOfToInvert.solve(Eigen::Matrix3d::Identity()) /
            static_cast<double>(3);
        c = M2;  // Already computed as A*p

        return currentEps;
    };

    // Assemble matrix A with dimensions d x n_points, where each column is a
    // point
    Eigen::MatrixXd A(3, numVertices);
    for (int i = 0; i < numVertices; ++i) {
        A.col(i) = hullV[i];
    }

    // Set parameters for Khachiyan's algorithm
    double eps = 1e-6;
    size_t maxiter = 1000;

    // Variables to store the resulting ellipsoid parameters
    Eigen::MatrixXd Q;
    Eigen::VectorXd c;

    // Call Khachiyan's algorithm to compute Q (shape matrix) and c (center)
    KhachiyanAlgo(A, eps, maxiter, Q, c);
    // Optionally check final_error for convergence quality

    // Use eigen-decomposition on Q to extract axes lengths and orientation
    // For ellipsoid defined by (x-c)^T Q (x-c) <= 1,
    // the eigenvectors of Q give orientation directions,
    // and the axes lengths (radii) are 1/sqrt(eigenvalues).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(Q);
    if (eigenSolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed.");
    }
    Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigenSolver.eigenvectors();

    // Compute radii = 1/sqrt(eigenvalues)
    Eigen::Vector3d radii = (1.0 / eigenvalues.array().sqrt()).matrix();

    // Construct the final oriented bounding ellipsoid
    auto dtype = points.GetDtype();
    OrientedBoundingEllipsoid obel;
    obel.center_ = c.head<3>().To(dtype);  // center vector of length 3
    obel.R_ = eigenvectors.To(dtype);      // orientation matrix
    obel.radii_ = radii.To(dtype);         // ellipsoid radii

    // Check oritntation and permute axes to closest identity
    mapToClosestIdentity(obel);

    return obel;
}

}  // namespace minimum_obel
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d