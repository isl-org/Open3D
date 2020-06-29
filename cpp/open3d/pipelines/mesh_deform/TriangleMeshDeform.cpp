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

#include "open3d/pipelines/mesh_deform/TriangleMeshDeform.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>

namespace open3d {
namespace pipelines {
namespace mesh_deform {

/// Helper function to get an edge with ordered vertex indices.
inline Eigen::Vector2i GetOrderedEdge(int vidx0, int vidx1) {
    return Eigen::Vector2i(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
}

/// \brief Function that computes for each edge in the triangle mesh and
/// passed as parameter edges_to_vertices the cot weight.
///
/// \param edges_to_vertices map from edge to vector of neighbouring
/// vertices.
/// \param min_weight minimum weight returned. Weights smaller than this
/// get clamped.
/// \return cot weight per edge.
std::unordered_map<Eigen::Vector2i,
                   double,
                   utility::hash_eigen::hash<Eigen::Vector2i>>
ComputeEdgeWeightsCot(
        const geometry::TriangleMesh &mesh,
        const std::unordered_map<Eigen::Vector2i,
                                 std::vector<int>,
                                 utility::hash_eigen::hash<Eigen::Vector2i>>
                &edges_to_vertices,
        double min_weight) {
    std::unordered_map<Eigen::Vector2i, double,
                       utility::hash_eigen::hash<Eigen::Vector2i>>
            weights;
    for (const auto &edge_v2s : edges_to_vertices) {
        Eigen::Vector2i edge = edge_v2s.first;
        double weight_sum = 0;
        int N = 0;
        for (int v2 : edge_v2s.second) {
            Eigen::Vector3d a = mesh.vertices_[edge(0)] - mesh.vertices_[v2];
            Eigen::Vector3d b = mesh.vertices_[edge(1)] - mesh.vertices_[v2];
            double weight = a.dot(b) / (a.cross(b)).norm();
            weight_sum += weight;
            N++;
        }
        double weight = N > 0 ? weight_sum / N : 0;
        if (weight < min_weight) {
            weights[edge] = min_weight;
        } else {
            weights[edge] = weight;
        }
    }
    return weights;
}

std::shared_ptr<geometry::TriangleMesh> DeformAsRigidAsPossible(
        const geometry::TriangleMesh &mesh,
        const std::vector<int> &constraint_vertex_indices,
        const std::vector<Eigen::Vector3d> &constraint_vertex_positions,
        size_t max_iter,
        DeformAsRigidAsPossibleEnergy energy_model,
        double smoothed_alpha) {
    auto prime = std::make_shared<geometry::TriangleMesh>();
    prime->vertices_ = mesh.vertices_;
    prime->triangles_ = mesh.triangles_;

    utility::LogDebug("[DeformAsRigidAsPossible] setting up S'");
    prime->ComputeAdjacencyList();
    auto edges_to_vertices = prime->GetEdgeToVerticesMap();
    auto edge_weights =
            ComputeEdgeWeightsCot(*prime, edges_to_vertices, /*min_weight=*/0);
    utility::LogDebug("[DeformAsRigidAsPossible] done setting up S'");

    std::unordered_map<int, Eigen::Vector3d> constraints;
    for (size_t idx = 0; idx < constraint_vertex_indices.size() &&
                         idx < constraint_vertex_positions.size();
         ++idx) {
        constraints[constraint_vertex_indices[idx]] =
                constraint_vertex_positions[idx];
    }

    double surface_area = -1;
    // std::vector<Eigen::Matrix3d> Rs(vertices_.size(),
    // Eigen::Matrix3d::Identity());
    std::vector<Eigen::Matrix3d> Rs(mesh.vertices_.size());
    std::vector<Eigen::Matrix3d> Rs_old;
    if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
        surface_area = prime->GetSurfaceArea();
        Rs_old.resize(mesh.vertices_.size());
    }

    // Build system matrix L and its solver
    utility::LogDebug("[DeformAsRigidAsPossible] setting up system matrix L");
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < int(mesh.vertices_.size()); ++i) {
        if (constraints.count(i) > 0) {
            triplets.push_back(Eigen::Triplet<double>(i, i, 1));
        } else {
            double W = 0;
            for (int j : prime->adjacency_list_[i]) {
                double w = edge_weights[GetOrderedEdge(i, j)];
                triplets.push_back(Eigen::Triplet<double>(i, j, -w));
                W += w;
            }
            if (W > 0) {
                triplets.push_back(Eigen::Triplet<double>(i, i, W));
            }
        }
    }
    Eigen::SparseMatrix<double> L(mesh.vertices_.size(), mesh.vertices_.size());
    L.setFromTriplets(triplets.begin(), triplets.end());
    utility::LogDebug(
            "[DeformAsRigidAsPossible] done setting up system matrix L");

    utility::LogDebug("[DeformAsRigidAsPossible] setting up sparse solver");
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(L);
    solver.factorize(L);
    if (solver.info() != Eigen::Success) {
        utility::LogError(
                "[DeformAsRigidAsPossible] Failed to build solver (factorize)");
    } else {
        utility::LogDebug(
                "[DeformAsRigidAsPossible] done setting up sparse solver");
    }

    std::vector<Eigen::VectorXd> b = {Eigen::VectorXd(mesh.vertices_.size()),
                                      Eigen::VectorXd(mesh.vertices_.size()),
                                      Eigen::VectorXd(mesh.vertices_.size())};
    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
            std::swap(Rs, Rs_old);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < int(mesh.vertices_.size()); ++i) {
            // Update rotations
            Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
            Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
            int n_nbs = 0;
            for (int j : prime->adjacency_list_[i]) {
                Eigen::Vector3d e0 = mesh.vertices_[i] - mesh.vertices_[j];
                Eigen::Vector3d e1 = prime->vertices_[i] - prime->vertices_[j];
                double w = edge_weights[GetOrderedEdge(i, j)];
                S += w * (e0 * e1.transpose());
                if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
                    R += Rs_old[j];
                }
                n_nbs++;
            }
            if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed &&
                iter > 0 && n_nbs > 0) {
                S = 2 * S +
                    (4 * smoothed_alpha * surface_area / n_nbs) * R.transpose();
            }
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                    S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Vector3d D(1, 1, (V * U.transpose()).determinant());
            // ensure rotation:
            // http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
            Rs[i] = V * D.asDiagonal() * U.transpose();
            if (Rs[i].determinant() <= 0) {
                utility::LogError(
                        "[DeformAsRigidAsPossible] something went wrong with "
                        "updating R");
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < int(mesh.vertices_.size()); ++i) {
            // Update Positions
            Eigen::Vector3d bi(0, 0, 0);
            if (constraints.count(i) > 0) {
                bi = constraints[i];
            } else {
                for (int j : prime->adjacency_list_[i]) {
                    double w = edge_weights[GetOrderedEdge(i, j)];
                    bi += w / 2 *
                          ((Rs[i] + Rs[j]) *
                           (mesh.vertices_[i] - mesh.vertices_[j]));
                }
            }
            b[0](i) = bi(0);
            b[1](i) = bi(1);
            b[2](i) = bi(2);
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int comp = 0; comp < 3; ++comp) {
            Eigen::VectorXd p_prime = solver.solve(b[comp]);
            if (solver.info() != Eigen::Success) {
                utility::LogError(
                        "[DeformAsRigidAsPossible] Cholesky solve failed");
            }
            for (int i = 0; i < int(mesh.vertices_.size()); ++i) {
                prime->vertices_[i](comp) = p_prime(i);
            }
        }

        // Compute energy and log
        double energy = 0;
        double reg = 0;
        for (int i = 0; i < int(mesh.vertices_.size()); ++i) {
            for (int j : prime->adjacency_list_[i]) {
                double w = edge_weights[GetOrderedEdge(i, j)];
                Eigen::Vector3d e0 = mesh.vertices_[i] - mesh.vertices_[j];
                Eigen::Vector3d e1 = prime->vertices_[i] - prime->vertices_[j];
                Eigen::Vector3d diff = e1 - Rs[i] * e0;
                energy += w * diff.squaredNorm();
                if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
                    reg += (Rs[i] - Rs[j]).squaredNorm();
                }
            }
        }
        if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
            energy = energy + smoothed_alpha * surface_area * reg;
        }
        utility::LogDebug("[DeformAsRigidAsPossible] iter={}, energy={:e}",
                          iter, energy);
    }

    return prime;
}

}  // namespace mesh_deform
}  // namespace pipelines
}  // namespace open3d
