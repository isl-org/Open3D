// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <Eigen/Eigenvalues>
#include <queue>
#include <tuple>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TetraMesh.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Eigen.h"

namespace open3d {

namespace {
using namespace geometry;

Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d &A, double eval0) {
    Eigen::Vector3d row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3d row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3d row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3d r0xr1 = row0.cross(row1);
    Eigen::Vector3d r0xr2 = row0.cross(row2);
    Eigen::Vector3d r1xr2 = row1.cross(row2);
    double d0 = r0xr1.dot(r0xr1);
    double d1 = r0xr2.dot(r0xr2);
    double d2 = r1xr2.dot(r1xr2);

    double dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }

    if (imax == 0) {
        return r0xr1 / std::sqrt(d0);
    } else if (imax == 1) {
        return r0xr2 / std::sqrt(d1);
    } else {
        return r1xr2 / std::sqrt(d2);
    }
}

Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d &A,
                                    const Eigen::Vector3d &evec0,
                                    double eval1) {
    Eigen::Vector3d U, V;
    if (std::abs(evec0(0)) > std::abs(evec0(1))) {
        double inv_length =
                1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
        U << -evec0(2) * inv_length, 0, evec0(0) * inv_length;
    } else {
        double inv_length =
                1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
        U << 0, evec0(2) * inv_length, -evec0(1) * inv_length;
    }
    V = evec0.cross(U);

    Eigen::Vector3d AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                       A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                       A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

    Eigen::Vector3d AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};

    double m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    double m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    double m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

    double absM00 = std::abs(m00);
    double absM01 = std::abs(m01);
    double absM11 = std::abs(m11);
    double max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = std::max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / std::sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            return m01 * U - m00 * V;
        } else {
            return U;
        }
    } else {
        max_abs_comp = std::max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / std::sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            return m11 * U - m01 * V;
        } else {
            return U;
        }
    }
}

Eigen::Vector3d FastEigen3x3(Eigen::Matrix3d &A) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    double max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return Eigen::Vector3d::Zero();
    }
    A /= max_coeff;

    double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3d eval;
        Eigen::Vector3d evec0;
        Eigen::Vector3d evec1;
        Eigen::Vector3d evec2;

        double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        double b00 = A(0, 0) - q;
        double b11 = A(1, 1) - q;
        double b22 = A(2, 2) - q;

        double p =
                std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        double c00 = b11 * b22 - A(1, 2) * A(1, 2);
        double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        double half_det = det * 0.5;
        half_det = std::min(std::max(half_det, -1.0), 1.0);

        double angle = std::acos(half_det) / (double)3;
        double const two_thirds_pi = 2.09439510239319549;
        double beta2 = std::cos(angle) * 2;
        double beta0 = std::cos(angle + two_thirds_pi) * 2;
        double beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < eval(0) && eval(2) < eval(1)) {
                A *= max_coeff;
                return evec2;
            }
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec0 = evec1.cross(evec2);
            return evec0;
        } else {
            evec0 = ComputeEigenvector0(A, eval(0));
            if (eval(0) < eval(1) && eval(0) < eval(2)) {
                A *= max_coeff;
                return evec0;
            }
            evec1 = ComputeEigenvector1(A, evec0, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec2 = evec0.cross(evec1);
            return evec2;
        }
    } else {
        A *= max_coeff;
        if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) {
            return Eigen::Vector3d(1, 0, 0);
        } else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) {
            return Eigen::Vector3d(0, 1, 0);
        } else {
            return Eigen::Vector3d(0, 0, 1);
        }
    }
}

Eigen::Vector3d ComputeNormal(const PointCloud &cloud,
                              const std::vector<int> &indices,
                              bool fast_normal_computation) {
    if (indices.size() == 0) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Matrix3d covariance =
            utility::ComputeCovariance(cloud.points_, indices);

    if (fast_normal_computation) {
        return FastEigen3x3(covariance);
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        solver.compute(covariance, Eigen::ComputeEigenvectors);
        return solver.eigenvectors().col(0);
    }
}

// Disjoint set data structure to find cycles in graphs
class DisjointSet {
public:
    DisjointSet(size_t size) : parent_(size), size_(size) {
        for (size_t idx = 0; idx < size; idx++) {
            parent_[idx] = idx;
            size_[idx] = 0;
        }
    }

    // find representative element for given x
    // using path compression
    size_t Find(size_t x) {
        if (x != parent_[x]) {
            parent_[x] = Find(parent_[x]);
        }
        return parent_[x];
    }

    // combine two sets using size of sets
    void Union(size_t x, size_t y) {
        x = Find(x);
        y = Find(y);
        if (x != y) {
            if (size_[x] < size_[y]) {
                size_[y] += size_[x];
                parent_[x] = y;
            } else {
                size_[x] += size_[y];
                parent_[y] = x;
            }
        }
    }

private:
    std::vector<size_t> parent_;
    std::vector<size_t> size_;
};

struct WeightedEdge {
    WeightedEdge(size_t v0, size_t v1, double weight)
        : v0_(v0), v1_(v1), weight_(weight) {}
    size_t v0_;
    size_t v1_;
    double weight_;
};

// Minimum Spanning Tree algorithm (Kruskal's algorithm)
std::vector<WeightedEdge> Kruskal(std::vector<WeightedEdge> &edges,
                                  size_t n_vertices) {
    std::sort(edges.begin(), edges.end(),
              [](WeightedEdge &e0, WeightedEdge &e1) {
                  return e0.weight_ < e1.weight_;
              });
    DisjointSet disjoint_set(n_vertices);
    std::vector<WeightedEdge> mst;
    for (size_t eidx = 0; eidx < edges.size(); ++eidx) {
        size_t set0 = disjoint_set.Find(edges[eidx].v0_);
        size_t set1 = disjoint_set.Find(edges[eidx].v1_);
        if (set0 != set1) {
            mst.push_back(edges[eidx]);
            disjoint_set.Union(set0, set1);
        }
    }
    return mst;
}

}  // unnamed namespace

namespace geometry {

void PointCloud::EstimateNormals(
        const KDTreeSearchParam &search_param /* = KDTreeSearchParamKNN()*/,
        bool fast_normal_computation /* = true */) {
    bool has_normal = HasNormals();
    if (!has_normal) {
        normals_.resize(points_.size());
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)points_.size(); i++) {
        std::vector<int> indices;
        std::vector<double> distance2;
        Eigen::Vector3d normal;
        if (kdtree.Search(points_[i], search_param, indices, distance2) >= 3) {
            normal = ComputeNormal(*this, indices, fast_normal_computation);
            if (normal.norm() == 0.0) {
                if (has_normal) {
                    normal = normals_[i];
                } else {
                    normal = Eigen::Vector3d(0.0, 0.0, 1.0);
                }
            }
            if (has_normal && normal.dot(normals_[i]) < 0.0) {
                normal *= -1.0;
            }
            normals_[i] = normal;
        } else {
            normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
        }
    }
}

void PointCloud::OrientNormalsToAlignWithDirection(
        const Eigen::Vector3d &orientation_reference
        /* = Eigen::Vector3d(0.0, 0.0, 1.0)*/) {
    if (!HasNormals()) {
        utility::LogError(
                "[OrientNormalsToAlignWithDirection] No normals in the "
                "PointCloud. Call EstimateNormals() first.");
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)points_.size(); i++) {
        auto &normal = normals_[i];
        if (normal.norm() == 0.0) {
            normal = orientation_reference;
        } else if (normal.dot(orientation_reference) < 0.0) {
            normal *= -1.0;
        }
    }
}

void PointCloud::OrientNormalsTowardsCameraLocation(
        const Eigen::Vector3d &camera_location /* = Eigen::Vector3d::Zero()*/) {
    if (!HasNormals()) {
        utility::LogError(
                "[OrientNormalsTowardsCameraLocation] No normals in the "
                "PointCloud. Call EstimateNormals() first.");
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)points_.size(); i++) {
        Eigen::Vector3d orientation_reference = camera_location - points_[i];
        auto &normal = normals_[i];
        if (normal.norm() == 0.0) {
            normal = orientation_reference;
            if (normal.norm() == 0.0) {
                normal = Eigen::Vector3d(0.0, 0.0, 1.0);
            } else {
                normal.normalize();
            }
        } else if (normal.dot(orientation_reference) < 0.0) {
            normal *= -1.0;
        }
    }
}

void PointCloud::OrientNormalsConsistentTangentPlane(size_t k) {
    if (!HasNormals()) {
        utility::LogError(
                "[OrientNormalsConsistentTangentPlane] No normals in the "
                "PointCloud. Call EstimateNormals() first.");
    }

    // Create Riemannian graph (Euclidian MST + kNN)
    // Euclidian MST is subgraph of Delaunay triangulation
    std::shared_ptr<TetraMesh> delaunay_mesh;
    std::vector<size_t> pt_map;
    std::tie(delaunay_mesh, pt_map) = TetraMesh::CreateFromPointCloud(*this);
    std::vector<WeightedEdge> delaunay_graph;
    std::unordered_set<size_t> graph_edges;
    auto EdgeIndex = [&](size_t v0, size_t v1) -> size_t {
        return std::min(v0, v1) * points_.size() + std::max(v0, v1);
    };
    auto AddEdgeToDelaunayGraph = [&](size_t v0, size_t v1) {
        v0 = pt_map[v0];
        v1 = pt_map[v1];
        size_t edge = EdgeIndex(v0, v1);
        if (graph_edges.count(edge) == 0) {
            double dist = (points_[v0] - points_[v1]).squaredNorm();
            delaunay_graph.push_back(WeightedEdge(v0, v1, dist));
            graph_edges.insert(edge);
        }
    };
    for (const Eigen::Vector4i &tetra : delaunay_mesh->tetras_) {
        AddEdgeToDelaunayGraph(tetra[0], tetra[1]);
        AddEdgeToDelaunayGraph(tetra[0], tetra[2]);
        AddEdgeToDelaunayGraph(tetra[0], tetra[3]);
        AddEdgeToDelaunayGraph(tetra[1], tetra[2]);
        AddEdgeToDelaunayGraph(tetra[1], tetra[3]);
        AddEdgeToDelaunayGraph(tetra[2], tetra[3]);
    }

    std::vector<WeightedEdge> mst = Kruskal(delaunay_graph, points_.size());

    auto NormalWeight = [&](size_t v0, size_t v1) -> double {
        return 1.0 - std::abs(normals_[v0].dot(normals_[v1]));
    };
    for (auto &edge : mst) {
        edge.weight_ = NormalWeight(edge.v0_, edge.v1_);
    }

    // Add k nearest neighbors to Riemannian graph
    KDTreeFlann kdtree(*this);
    for (size_t v0 = 0; v0 < points_.size(); ++v0) {
        std::vector<int> neighbors;
        std::vector<double> dists2;
        kdtree.SearchKNN(points_[v0], int(k), neighbors, dists2);
        for (size_t vidx1 = 0; vidx1 < neighbors.size(); ++vidx1) {
            size_t v1 = size_t(neighbors[vidx1]);
            if (v0 == v1) {
                continue;
            }
            size_t edge = EdgeIndex(v0, v1);
            if (graph_edges.count(edge) == 0) {
                double weight = NormalWeight(v0, v1);
                mst.push_back(WeightedEdge(v0, v1, weight));
                graph_edges.insert(edge);
            }
        }
    }

    // extract MST from Riemannian graph
    mst = Kruskal(mst, points_.size());

    // convert list of edges to graph
    std::vector<std::unordered_set<size_t>> mst_graph(points_.size());
    for (const auto &edge : mst) {
        size_t v0 = edge.v0_;
        size_t v1 = edge.v1_;
        mst_graph[v0].insert(v1);
        mst_graph[v1].insert(v0);
    }

    // find start node for tree traversal
    // init with node that maximizes z
    double max_z = std::numeric_limits<double>::lowest();
    size_t v0;
    for (size_t vidx = 0; vidx < points_.size(); ++vidx) {
        const Eigen::Vector3d &v = points_[vidx];
        if (v(2) > max_z) {
            max_z = v(2);
            v0 = vidx;
        }
    }

    // traverse MST and orient normals consistently
    std::queue<size_t> traversal_queue;
    std::vector<bool> visited(points_.size(), false);
    traversal_queue.push(v0);
    auto TestAndOrientNormal = [&](const Eigen::Vector3d &n0,
                                   Eigen::Vector3d &n1) {
        if (n0.dot(n1) < 0) {
            n1 *= -1;
        }
    };
    TestAndOrientNormal(Eigen::Vector3d(0, 0, 1), normals_[v0]);
    while (!traversal_queue.empty()) {
        v0 = traversal_queue.front();
        traversal_queue.pop();
        visited[v0] = true;
        for (size_t v1 : mst_graph[v0]) {
            if (!visited[v1]) {
                traversal_queue.push(v1);
                TestAndOrientNormal(normals_[v0], normals_[v1]);
            }
        }
    }
}

}  // namespace geometry
}  // namespace open3d
