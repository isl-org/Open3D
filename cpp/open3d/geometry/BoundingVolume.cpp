// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "open3d/geometry/BoundingVolume.h"

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

OrientedBoundingBox& OrientedBoundingBox::Clear() {
    center_.setZero();
    extent_.setZero();
    R_ = Eigen::Matrix3d::Identity();
    color_.setOnes();
    return *this;
}

bool OrientedBoundingBox::IsEmpty() const { return Volume() <= 0; }

Eigen::Vector3d OrientedBoundingBox::GetMinBound() const {
    auto points = GetBoxPoints();
    return ComputeMinBound(points);
}

Eigen::Vector3d OrientedBoundingBox::GetMaxBound() const {
    auto points = GetBoxPoints();
    return ComputeMaxBound(points);
}

Eigen::Vector3d OrientedBoundingBox::GetCenter() const { return center_; }

AxisAlignedBoundingBox OrientedBoundingBox::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetBoxPoints());
}

OrientedBoundingBox OrientedBoundingBox::GetOrientedBoundingBox(bool) const {
    return *this;
}

OrientedBoundingBox OrientedBoundingBox::GetMinimalOrientedBoundingBox(
        bool) const {
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of an OrientedBoundingBox is not implemented. "
            "Call Translate, Scale, and Rotate.");
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        center_ += translation;
    } else {
        center_ = translation;
    }
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Scale(const double scale,
                                                const Eigen::Vector3d& center) {
    extent_ *= scale;
    center_ = scale * (center_ - center) + center;
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Rotate(
        const Eigen::Matrix3d& R, const Eigen::Vector3d& center) {
    R_ = R * R_;
    center_ = R * (center_ - center) + center;
    return *this;
}

double OrientedBoundingBox::Volume() const {
    return extent_(0) * extent_(1) * extent_(2);
}

std::vector<Eigen::Vector3d> OrientedBoundingBox::GetBoxPoints() const {
    Eigen::Vector3d x_axis = R_ * Eigen::Vector3d(extent_(0) / 2, 0, 0);
    Eigen::Vector3d y_axis = R_ * Eigen::Vector3d(0, extent_(1) / 2, 0);
    Eigen::Vector3d z_axis = R_ * Eigen::Vector3d(0, 0, extent_(2) / 2);
    std::vector<Eigen::Vector3d> points(8);
    points[0] = center_ - x_axis - y_axis - z_axis;
    points[1] = center_ + x_axis - y_axis - z_axis;
    points[2] = center_ - x_axis + y_axis - z_axis;
    points[3] = center_ - x_axis - y_axis + z_axis;
    points[4] = center_ + x_axis + y_axis + z_axis;
    points[5] = center_ - x_axis + y_axis + z_axis;
    points[6] = center_ + x_axis - y_axis + z_axis;
    points[7] = center_ + x_axis + y_axis - z_axis;
    return points;
}

std::vector<size_t> OrientedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<Eigen::Vector3d>& points) const {
    std::vector<size_t> indices;
    Eigen::Vector3d dx = R_ * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d dy = R_ * Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d dz = R_ * Eigen::Vector3d(0, 0, 1);
    for (size_t idx = 0; idx < points.size(); idx++) {
        Eigen::Vector3d d = points[idx] - center_;
        if (std::abs(d.dot(dx)) <= extent_(0) / 2 &&
            std::abs(d.dot(dy)) <= extent_(1) / 2 &&
            std::abs(d.dot(dz)) <= extent_(2) / 2) {
            indices.push_back(idx);
        }
    }
    return indices;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox& aabox) {
    OrientedBoundingBox obox;
    obox.center_ = aabox.GetCenter();
    obox.extent_ = aabox.GetExtent();
    obox.R_ = Eigen::Matrix3d::Identity();
    return obox;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    PointCloud hull_pcd;
    std::vector<size_t> hull_point_indices;
    {
        std::shared_ptr<TriangleMesh> mesh;
        std::tie(mesh, hull_point_indices) =
                Qhull::ComputeConvexHull(points, robust);
        hull_pcd.points_ = mesh->vertices_;
    }

    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    std::tie(mean, cov) = hull_pcd.ComputeMeanAndCovariance();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d evals = es.eigenvalues();
    Eigen::Matrix3d R = es.eigenvectors();

    if (evals(1) > evals(0)) {
        std::swap(evals(1), evals(0));
        Eigen::Vector3d tmp = R.col(1);
        R.col(1) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(0)) {
        std::swap(evals(2), evals(0));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(1)) {
        std::swap(evals(2), evals(1));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(1);
        R.col(1) = tmp;
    }
    R.col(0) /= R.col(0).norm();
    R.col(1) /= R.col(1).norm();
    R.col(2) = R.col(0).cross(R.col(1));

    for (size_t i = 0; i < hull_point_indices.size(); ++i) {
        hull_pcd.points_[i] =
                R.transpose() * (points[hull_point_indices[i]] - mean);
    }

    const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();

    OrientedBoundingBox obox;
    obox.center_ = R * aabox.GetCenter() + mean;
    obox.R_ = R;
    obox.extent_ = aabox.GetExtent();

    return obox;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPointsMinimalApprox(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    std::shared_ptr<TriangleMesh> mesh;
    std::tie(mesh, std::ignore) = Qhull::ComputeConvexHull(points, robust);
    double min_vol = -1;
    OrientedBoundingBox min_box;
    PointCloud hull_pcd;
    for (auto& tri : mesh->triangles_) {
        hull_pcd.points_ = mesh->vertices_;
        Eigen::Vector3d a = mesh->vertices_[tri(0)];
        Eigen::Vector3d b = mesh->vertices_[tri(1)];
        Eigen::Vector3d c = mesh->vertices_[tri(2)];
        Eigen::Vector3d u = b - a;
        Eigen::Vector3d v = c - a;
        Eigen::Vector3d w = u.cross(v);
        v = w.cross(u);
        u = u / u.norm();
        v = v / v.norm();
        w = w / w.norm();
        Eigen::Matrix3d m_rot;
        m_rot << u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2];
        hull_pcd.Rotate(m_rot.inverse(), a);

        const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();
        double volume = aabox.Volume();
        if (min_vol == -1. || volume < min_vol) {
            min_vol = volume;
            min_box = aabox.GetOrientedBoundingBox();
            min_box.Rotate(m_rot, a);
        }
    }
    return min_box;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPointsMinimal(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    // ------------------------------------------------------------
    // 0) Compute the convex hull of the input point cloud
    // ------------------------------------------------------------
    if (points.empty()) {
        utility::LogError("CreateFromPointsMinimal: Input point set is empty.");
        return OrientedBoundingBox();
    }
    std::shared_ptr<TriangleMesh> hullMesh;
    std::tie(hullMesh, std::ignore) = Qhull::ComputeConvexHull(points, robust);
    if (!hullMesh) {
        utility::LogError("Failed to compute convex hull.");
        return OrientedBoundingBox();
    }

    // Get convex hull vertices and triangles
    const std::vector<Eigen::Vector3d>& hullV = hullMesh->vertices_;
    const std::vector<Eigen::Vector3i>& hullT = hullMesh->triangles_;
    int numVertices = static_cast<int>(hullV.size());
    int numTriangles = static_cast<int>(hullT.size());

    OrientedBoundingBox minOBB;
    double minVolume = std::numeric_limits<double>::max();

    // Handle degenerate planar cases up front.
    if (numVertices <= 3 || numTriangles < 1) {  // Handle degenerate case
        utility::LogError("Convex hull is degenerate.");
        return OrientedBoundingBox();
    }

    auto mapOBBToClosestIdentity = [&](OrientedBoundingBox& obb) {
        Eigen::Matrix3d& R = obb.R_;
        Eigen::Vector3d& extent = obb.extent_;
        Eigen::Vector3d col[3] = {R.col(0), R.col(1), R.col(2)};
        Eigen::Vector3d ext = extent;
        double best_score = -1e9;
        Eigen::Matrix3d best_R;
        Eigen::Vector3d best_extent;
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
                    best_extent(0) = ext(p[0]);
                    best_extent(1) = ext(p[1]);
                    best_extent(2) = ext(p[2]);
                }
            }
        }

        // Update the OBB with the best orientation found
        obb.R_ = best_R;
        obb.extent_ = best_extent;

        // Note: obb.center_ and obb.color_ remain unchanged.
    };

    // --------------------------------------------------------------------
    // 1) Precompute vertex adjacency data, face normals, and edge data
    // --------------------------------------------------------------------
    std::vector<std::vector<int>> adjacencyData;
    adjacencyData.reserve(numVertices);
    adjacencyData.insert(adjacencyData.end(), numVertices, std::vector<int>());

    std::vector<Eigen::Vector3d> faceNormals;
    faceNormals.reserve(numTriangles);

    // Each edge is stored as (v0, v1).
    std::vector<std::pair<int, int>> edges;
    edges.reserve(numVertices * 2);

    // Each edge knows which two faces it belongs to: (f0, f1).
    std::vector<std::pair<int, int>> facesForEdge;
    facesForEdge.reserve(numVertices * 2);

    constexpr unsigned int emptyEdge = std::numeric_limits<unsigned int>::max();
    std::vector<unsigned int> vertexPairsToEdges(numVertices * numVertices,
                                                 emptyEdge);

    for (int i = 0; i < numTriangles; ++i) {
        const Eigen::Vector3i& tri = hullT[i];
        int t0 = tri(0), t1 = tri(1), t2 = tri(2);
        int v0 = t2, v1 = t0;

        for (int j = 0; j < 3; ++j) {
            v1 = tri(j);

            // Build Adjacency Data (vertex -> adjacent vertices)
            adjacencyData[v0].push_back(v1);

            // Register Edges (edge -> neighbouring faces)
            unsigned int& refIdx1 = vertexPairsToEdges[v0 * numVertices + v1];
            unsigned int& refIdx2 = vertexPairsToEdges[v1 * numVertices + v0];
            if (refIdx1 == emptyEdge) {
                // Not registered yet
                unsigned int newIdx = static_cast<unsigned int>(edges.size());
                refIdx1 = newIdx;
                refIdx2 = newIdx;
                edges.emplace_back(v0, v1);
                facesForEdge.emplace_back(i, -1);
            } else {
                // Already existing, update the second face index
                facesForEdge[refIdx1].second = i;
            }

            v0 = v1;
        }
        // Compute Face Normal
        auto n = (hullV[t1] - hullV[t0]).cross(hullV[t2] - hullV[t0]);
        faceNormals.push_back(n.normalized());
    }

    // ------------------------------------------------------------
    // 2) Precompute "antipodal vertices" for each edge of the hull
    // ------------------------------------------------------------

    // Throughout the algorithm, internal edges can all be discarded.
    auto isInternalEdge = [&](std::size_t iEdge) noexcept {
        return (faceNormals[facesForEdge[iEdge].first].dot(
                        faceNormals[facesForEdge[iEdge].second]) > 1.0 - 1e-4);
    };

    // Throughout the whole algorithm, this array stores an auxiliary structure
    // for performing graph searches on the vertices of the convex hull.
    // Conceptually each index of the array stores a boolean whether we have
    // visited that vertex or not during the current search. However storing
    // such booleans is slow, since we would have to perform a linear-time scan
    // through this array before next search to reset each boolean to unvisited
    // false state. Instead, store a number, called a "color" for each vertex to
    // specify whether that vertex has been visited, and manage a global color
    // counter floodFillVisitColor that represents the visited vertices. At any
    // given time, the vertices that have already been visited have the value
    // floodFillVisited[i] == floodFillVisitColor in them. This gives a win that
    // we can perform constant-time clears of the floodFillVisited array, by
    // simply incrementing the "color" counter to clear the array.

    int edgeSize = edges.size();
    std::vector<std::vector<int>> antipodalPointsForEdge(edgeSize);
    antipodalPointsForEdge.reserve(edgeSize);

    std::vector<unsigned int> floodFillVisited(numVertices, 0u);
    unsigned int floodFillVisitColor = 1u;

    auto markVertexVisited = [&](int v) {
        floodFillVisited[v] = floodFillVisitColor;
    };

    auto haveVisitedVertex = [&](int v) -> bool {
        return floodFillVisited[v] == floodFillVisitColor;
    };

    auto clearGraphSearch = [&]() { ++floodFillVisitColor; };

    auto isVertexAntipodalToEdge =
            [&](int vi, const std::vector<int>& neighbors,
                const Eigen::Vector3d& f1a,
                const Eigen::Vector3d& f1b) noexcept -> bool {
        constexpr double epsilon = 1e-4;
        constexpr double degenerateThreshold = -5e-2;
        double tMin = 0.0;
        double tMax = 1.0;

        // Precompute values outside the loop for efficiency.
        const auto& v = hullV[vi];
        Eigen::Vector3d f1b_f1a = f1b - f1a;

        // Iterate over each neighbor.
        for (int neighborIndex : neighbors) {
            const auto& neighbor = hullV[neighborIndex];

            // Compute edge vector e = neighbor - v.
            Eigen::Vector3d e = neighbor - v;

            // Compute dot products manually for efficiency.
            double s = f1b_f1a.dot(e);
            double n = f1b.dot(e);

            // Adjust tMin and tMax based on the value of s.
            if (s > epsilon) {
                tMax = std::min(tMax, n / s);
            } else if (s < -epsilon) {
                tMin = std::max(tMin, n / s);
            } else if (n < -epsilon) {
                // No feasible t if n is negative when s is nearly zero.
                return false;
            }

            // If the valid interval for t has degenerated, exit early.
            if ((tMax - tMin) < degenerateThreshold) {
                return false;
            }
        }
        return true;
    };

    auto extremeVertexConvex = [&](auto& self, const Eigen::Vector3d& direction,
                                   std::vector<unsigned int>& floodFillVisited,
                                   unsigned int floodFillVisitColor,
                                   double& mostExtremeDistance,
                                   int startingVertex) -> int {
        // Compute dot product for the starting vertex.
        double curD = direction.dot(hullV[startingVertex]);

        // Cache neighbor list for the starting vertex.
        const int* neighbors = &adjacencyData[startingVertex][0];
        const int* neighborsEnd =
                neighbors + adjacencyData[startingVertex].size();

        // Mark starting vertex as visited.
        floodFillVisited[startingVertex] = floodFillVisitColor;

        // Traverse neighbors to find more extreme vertices.
        int secondBest = -1;
        double secondBestD = curD - 1e-3;
        while (neighbors != neighborsEnd) {
            int n = *neighbors++;
            if (floodFillVisited[n] != floodFillVisitColor) {
                double d = direction.dot(hullV[n]);
                if (d > curD) {
                    // Found a new vertex with higher dot product.
                    startingVertex = n;
                    curD = d;
                    floodFillVisited[startingVertex] = floodFillVisitColor;
                    neighbors = &adjacencyData[startingVertex][0];
                    neighborsEnd =
                            neighbors + adjacencyData[startingVertex].size();
                    secondBest = -1;
                    secondBestD = curD - 1e-3;
                } else if (d > secondBestD) {
                    // Update second-best candidate.
                    secondBest = n;
                    secondBestD = d;
                }
            }
        }

        // Explore second-best neighbor recursively if valid.
        if (secondBest != -1 &&
            floodFillVisited[secondBest] != floodFillVisitColor) {
            double secondMostExtreme = -std::numeric_limits<double>::infinity();
            int secondTry =
                    self(self, direction, floodFillVisited, floodFillVisitColor,
                         secondMostExtreme, secondBest);

            if (secondMostExtreme > curD) {
                mostExtremeDistance = secondMostExtreme;
                return secondTry;
            }
        }

        mostExtremeDistance = curD;
        return startingVertex;
    };

    // The currently best variant for establishing a spatially coherent
    // traversal order.
    std::vector<int> spatialFaceOrder;
    spatialFaceOrder.reserve(numTriangles);
    std::vector<int> spatialEdgeOrder;
    spatialEdgeOrder.reserve(edgeSize);

    // Initialize random number generator
    std::random_device rd;   // Obtain a random number from hardware
    std::mt19937 rng(rd());  // Seed the generator
    {  // Explicit scope for variables that are not needed after this.

        std::vector<unsigned int> visitedEdges(edgeSize, 0u);
        std::vector<unsigned int> visitedFaces(numTriangles, 0u);

        std::vector<std::pair<int, int>> traverseStackEdges;
        traverseStackEdges.reserve(edgeSize);
        traverseStackEdges.emplace_back(0, adjacencyData[0].front());
        while (!traverseStackEdges.empty()) {
            auto e = traverseStackEdges.back();
            traverseStackEdges.pop_back();

            // Find edge index
            int edgeIdx = vertexPairsToEdges[e.first * numVertices + e.second];
            if (visitedEdges[edgeIdx]) continue;
            visitedEdges[edgeIdx] = 1;
            auto& ff = facesForEdge[edgeIdx];
            if (!visitedFaces[ff.first]) {
                visitedFaces[ff.first] = 1;
                spatialFaceOrder.push_back(ff.first);
            }
            if (!visitedFaces[ff.second]) {
                visitedFaces[ff.second] = 1;
                spatialFaceOrder.push_back(ff.second);
            }

            // If not an internal edge, keep it
            if (!isInternalEdge(edgeIdx)) {
                spatialEdgeOrder.push_back(edgeIdx);
            }

            int v0 = e.second;
            size_t sizeBefore = traverseStackEdges.size();
            for (int v1 : adjacencyData[v0]) {
                int e1 = vertexPairsToEdges[v0 * numVertices + v1];
                if (visitedEdges[e1]) continue;
                traverseStackEdges.push_back(std::make_pair(v0, v1));
            }

            // Randomly shuffle newly added edges
            int nNewEdges =
                    static_cast<int>(traverseStackEdges.size() - sizeBefore);
            if (nNewEdges > 0) {
                std::uniform_int_distribution<> distr(0, nNewEdges - 1);
                int r = distr(rng);
                std::swap(traverseStackEdges.back(),
                          traverseStackEdges[sizeBefore + r]);
            }
        }
    }

    // --------------------------------------------------------------------
    // 3) Precompute "sidepodal edges" for each edge of the hull
    // --------------------------------------------------------------------

    // Stores a memory of yet unvisited vertices for current graph search.
    std::vector<int> traverseStack;

    // Since we do several extreme vertex searches, and the search directions
    // have a lot of spatial locality, always start the search for the next
    // extreme vertex from the extreme vertex that was found during the previous
    // iteration for the previous edge. This has been profiled to improve
    // overall performance by as much as 15-25%.
    int startVertex = 0;

    // Precomputation: for each edge, we need to compute the list of potential
    // antipodal points (points on the opposing face of an enclosing OBB of the
    // face that is flush with the given edge of the polyhedron).
    for (int edgeI : spatialEdgeOrder) {
        auto [faceI_a, faceI_b] = facesForEdge[edgeI];
        const Eigen::Vector3d& f1a = faceNormals[faceI_a];
        const Eigen::Vector3d& f1b = faceNormals[faceI_b];

        double dummy;
        clearGraphSearch();
        startVertex =
                extremeVertexConvex(extremeVertexConvex, -f1a, floodFillVisited,
                                    floodFillVisitColor, dummy, startVertex);
        clearGraphSearch();

        traverseStack.push_back(startVertex);
        markVertexVisited(startVertex);
        while (!traverseStack.empty()) {
            int v = traverseStack.back();
            traverseStack.pop_back();
            const auto& neighbors = adjacencyData[v];
            if (isVertexAntipodalToEdge(v, neighbors, f1a, f1b)) {
                if (edges[edgeI].first == v || edges[edgeI].second == v) {
                    return OrientedBoundingBox();
                }
                antipodalPointsForEdge[edgeI].push_back(v);
                for (size_t j = 0; j < neighbors.size(); ++j) {
                    if (!haveVisitedVertex(neighbors[j])) {
                        traverseStack.push_back(neighbors[j]);
                        markVertexVisited(neighbors[j]);
                    }
                }
            }
        }

        // Robustness: If the above search did not find any antipodal points,
        // add the first found extreme point at least, since it is always an
        // antipodal point. This is known to occur very rarely due to numerical
        // imprecision in the above loop over adjacent edges.
        if (antipodalPointsForEdge[edgeI].empty()) {
            // Getting here is most likely a bug. Fall back to linear scan,
            // which is very slow.
            for (int j = 0; j < numVertices; ++j) {
                if (isVertexAntipodalToEdge(j, adjacencyData[j], f1a, f1b)) {
                    antipodalPointsForEdge[edgeI].push_back(j);
                }
            }
        }
    }

    // Data structure for sidepodal vertices.
    std::vector<unsigned char> sidepodalVertices(edgeSize * numVertices, 0);

    // Stores for each edge i the list of all sidepodal edge indices j that it
    // can form an OBB with.
    std::vector<std::vector<int>> compatibleEdges(edgeSize);
    compatibleEdges.reserve(edgeSize);

    // Compute all sidepodal edges for each edge by performing a graph search.
    // The set of sidepodal edges is connected in the graph, which lets us avoid
    // having to iterate over each edge pair of the convex hull.
    for (int edgeI : spatialEdgeOrder) {
        auto [faceI_a, faceI_b] = facesForEdge[edgeI];
        const Eigen::Vector3d& f1a = faceNormals[faceI_a];
        const Eigen::Vector3d& f1b = faceNormals[faceI_b];

        // Pixar orthonormal basis code:
        // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        Eigen::Vector3d deadDirection = (f1a + f1b) * 0.5;
        Eigen::Vector3d basis1, basis2;
        double sign = std::copysign(1.0, deadDirection.z());
        const double a = -1.0 / (sign + deadDirection.z());
        const double b = deadDirection.x() * deadDirection.y() * a;
        basis1 = Eigen::Vector3d(
                1.0 + sign * deadDirection.x() * deadDirection.x() * a,
                sign * b, -sign * deadDirection.x());
        basis2 = Eigen::Vector3d(
                b, sign + deadDirection.y() * deadDirection.y() * a,
                -deadDirection.y());

        double dummy;
        Eigen::Vector3d dir =
                (f1a.cross(Eigen::Vector3d(0, 1, 0))).normalized();
        if (dir.norm() < 1e-4) {
            dir = Eigen::Vector3d(0, 0, 1);  // If f1a is parallel to y-axis
        }
        clearGraphSearch();
        startVertex =
                extremeVertexConvex(extremeVertexConvex, dir, floodFillVisited,
                                    floodFillVisitColor, dummy, startVertex);
        clearGraphSearch();
        traverseStack.push_back(startVertex);
        while (!traverseStack.empty()) {
            int v = traverseStack.back();
            traverseStack.pop_back();

            if (haveVisitedVertex(v)) continue;
            markVertexVisited(v);

            // const auto& neighbors = adjacencyData[v];
            for (int vAdj : adjacencyData[v]) {
                if (haveVisitedVertex(vAdj)) continue;
                int edge = vertexPairsToEdges[v * numVertices + vAdj];
                auto [faceI_a, faceI_b] = facesForEdge[edge];
                Eigen::Vector3d f1a_f1b = f1a - f1b;
                Eigen::Vector3d f2a_f2b =
                        faceNormals[faceI_a] - faceNormals[faceI_b];

                double a2 = f1b.dot(faceNormals[faceI_b]);
                double b2 = f1a_f1b.dot(faceNormals[faceI_b]);
                double c2 = f2a_f2b.dot(f1b);
                double d2 = f1a_f1b.dot(f2a_f2b);
                double ab = a2 + b2;
                double ac = a2 + c2;
                double abcd = ab + c2 + d2;
                double minVal = std::min({a2, ab, ac, abcd});
                double maxVal = std::max({a2, ab, ac, abcd});
                bool are_edges_compatible_for_obb =
                        (minVal <= 0.0 && maxVal >= 0.0);

                if (are_edges_compatible_for_obb) {
                    if (edgeI <= edge) {
                        if (!isInternalEdge(edge)) {
                            compatibleEdges[edgeI].push_back(edge);
                        }

                        sidepodalVertices[edgeI * numVertices +
                                          edges[edge].first] = 1;
                        sidepodalVertices[edgeI * numVertices +
                                          edges[edge].second] = 1;
                        if (edgeI != edge) {
                            if (!isInternalEdge(edge)) {
                                compatibleEdges[edge].push_back(edgeI);
                            }
                            sidepodalVertices[edge * numVertices +
                                              edges[edgeI].first] = 1;
                            sidepodalVertices[edge * numVertices +
                                              edges[edgeI].second] = 1;
                        }
                    }
                    traverseStack.push_back(vAdj);
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // 4) Test configurations where all three edges are on adjacent faces.
    // --------------------------------------------------------------------

    // Take advantage of spatial locality: start the search for the extreme
    // vertex from the extreme vertex that was found during the previous
    // iteration for the previous edge. This speeds up the search since edge
    // directions have some amount of spatial locality and the next extreme
    // vertex is often close to the previous one. Track two hint variables since
    // we are performing extreme vertex searches to two opposing directions at
    // the same time.
    int vHint1 = 0;
    int vHint2 = 0;
    int vHint3 = 0;
    int vHint4 = 0;
    int vHint1_b = 0;
    int vHint2_b = 0;
    int vHint3_b = 0;

    // Stores a memory of yet unvisited vertices that are common sidepodal
    // vertices to both currently chosen edges for current graph search.
    std::vector<int> traverseStackCommonSidepodals;
    traverseStackCommonSidepodals.reserve(numVertices);
    for (int edgeI : spatialEdgeOrder) {
        auto [faceI_a, faceI_b] = facesForEdge[edgeI];
        const Eigen::Vector3d& f1a = faceNormals[faceI_a];
        const Eigen::Vector3d& f1b = faceNormals[faceI_b];

        const auto& compatibleEdgesI = compatibleEdges[edgeI];
        Eigen::Vector3d baseDir = 0.5 * (f1a + f1b);

        for (int edgeJ : compatibleEdgesI) {
            if (edgeJ <= edgeI) continue;  // Remove symmetry.
            auto [faceJ_a, faceJ_b] = facesForEdge[edgeJ];
            const Eigen::Vector3d& f2a = faceNormals[faceJ_a];
            const Eigen::Vector3d& f2b = faceNormals[faceJ_b];

            // Compute search direction
            Eigen::Vector3d deadDir = 0.5 * (f2a + f2b);
            Eigen::Vector3d searchDir = baseDir.cross(deadDir);
            searchDir = searchDir.normalized();
            if (searchDir.norm() < 1e-9) {
                searchDir = f1a.cross(f2a);
                searchDir = searchDir.normalized();
                if (searchDir.norm() < 1e-9) {
                    searchDir =
                            (f1a.cross(Eigen::Vector3d(0, 1, 0))).normalized();
                }
            }

            double dummy;
            clearGraphSearch();
            vHint1 = extremeVertexConvex(extremeVertexConvex, searchDir,
                                         floodFillVisited, floodFillVisitColor,
                                         dummy, vHint1);
            clearGraphSearch();
            vHint2 = extremeVertexConvex(extremeVertexConvex, -searchDir,
                                         floodFillVisited, floodFillVisitColor,
                                         dummy, vHint2);

            int secondSearch = -1;
            if (sidepodalVertices[edgeJ * numVertices + vHint1]) {
                traverseStackCommonSidepodals.push_back(vHint1);
            } else {
                traverseStack.push_back(vHint1);
            }
            if (sidepodalVertices[edgeJ * numVertices + vHint2]) {
                traverseStackCommonSidepodals.push_back(vHint2);
            } else {
                secondSearch = vHint2;
            }

            // Bootstrap to a good vertex that is sidepodal to both edges.
            clearGraphSearch();
            while (!traverseStack.empty()) {
                int v = traverseStack.front();
                traverseStack.erase(traverseStack.begin());
                if (haveVisitedVertex(v)) continue;
                markVertexVisited(v);
                const auto& neighbors = adjacencyData[v];
                for (int vAdj : neighbors) {
                    if (!haveVisitedVertex(vAdj) &&
                        sidepodalVertices[edgeI * numVertices + vAdj]) {
                        if (sidepodalVertices[edgeJ * numVertices + vAdj]) {
                            traverseStack.clear();
                            if (secondSearch != -1) {
                                traverseStack.push_back(secondSearch);
                                secondSearch = -1;
                                markVertexVisited(vAdj);
                            }
                            traverseStackCommonSidepodals.push_back(vAdj);
                            break;
                        } else {
                            traverseStack.push_back(vAdj);
                        }
                    }
                }
            }

            clearGraphSearch();
            while (!traverseStackCommonSidepodals.empty()) {
                int v = traverseStackCommonSidepodals.back();
                traverseStackCommonSidepodals.pop_back();
                if (haveVisitedVertex(v)) continue;
                markVertexVisited(v);
                const auto& neighbors = adjacencyData[v];
                for (int vAdj : neighbors) {
                    int edgeK = vertexPairsToEdges[v * numVertices + vAdj];
                    int idxI = edgeI * numVertices + vAdj;
                    int idxJ = edgeJ * numVertices + vAdj;

                    if (isInternalEdge(edgeK)) continue;

                    if (sidepodalVertices[idxI] && sidepodalVertices[idxJ]) {
                        if (!haveVisitedVertex(vAdj)) {
                            traverseStackCommonSidepodals.push_back(vAdj);
                        }
                        if (edgeJ < edgeK) {
                            auto [faceK_a, faceK_b] = facesForEdge[edgeK];
                            const Eigen::Vector3d& f3a = faceNormals[faceK_a];
                            const Eigen::Vector3d& f3b = faceNormals[faceK_b];

                            std::vector<Eigen::Vector3d> n1 = {
                                    Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero()};
                            std::vector<Eigen::Vector3d> n2 = {
                                    Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero()};
                            std::vector<Eigen::Vector3d> n3 = {
                                    Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero()};

                            constexpr double eps = 1e-4;
                            constexpr double angleEps = 1e-3;
                            int nSolutions = 0;

                            {
                                // Precompute intermediate vectors for
                                // polynomial coefficients.
                                Eigen::Vector3d a = f1b;
                                Eigen::Vector3d b = f1a - f1b;
                                Eigen::Vector3d c = f2b;
                                Eigen::Vector3d d = f2a - f2b;
                                Eigen::Vector3d e = f3b;
                                Eigen::Vector3d f = f3a - f3b;

                                // Compute polynomial coefficients.
                                double g = a.dot(c) * d.dot(e) -
                                           a.dot(d) * c.dot(e);
                                double h = a.dot(c) * d.dot(f) -
                                           a.dot(d) * c.dot(f);
                                double i = b.dot(c) * d.dot(e) -
                                           b.dot(d) * c.dot(e);
                                double j = b.dot(c) * d.dot(f) -
                                           b.dot(d) * c.dot(f);
                                double k = g * b.dot(e) - a.dot(e) * i;
                                double l = h * b.dot(e) + g * b.dot(f) -
                                           a.dot(f) * i - a.dot(e) * j;
                                double m = h * b.dot(f) - a.dot(f) * j;
                                double s = l * l - 4 * m * k;

                                // Handle degenerate or linear case.
                                if (std::abs(m) < 1e-5 || std::abs(s) < 1e-5) {
                                    double v = -k / l;
                                    double t = -(g + h * v) / (i + j * v);
                                    double u = -(c.dot(e) + c.dot(f) * v) /
                                               (d.dot(e) + d.dot(f) * v);
                                    nSolutions = 0;

                                    // If we happened to divide by zero above,
                                    // the following checks handle them.
                                    if (v >= -eps && t >= -eps && u >= -eps &&
                                        v <= 1.0 + eps && t <= 1.0 + eps &&
                                        u <= 1.0 + eps) {
                                        n1[0] = (a + b * t).normalized();
                                        n2[0] = (c + d * u).normalized();
                                        n3[0] = (e + f * v).normalized();
                                        if (std::abs(n1[0].dot(n2[0])) <
                                                    angleEps &&
                                            std::abs(n1[0].dot(n3[0])) <
                                                    angleEps &&
                                            std::abs(n2[0].dot(n3[0])) <
                                                    angleEps) {
                                            nSolutions = 1;
                                        } else {
                                            nSolutions = 0;
                                        }
                                    }
                                } else {
                                    // Discriminant negative: no solutions for v
                                    if (s < 0.0) {
                                        nSolutions = 0;
                                    } else {
                                        double sgnL = l < 0 ? -1.0 : 1.0;
                                        double V1 = -(l + sgnL * std::sqrt(s)) /
                                                    (2.0 * m);
                                        double V2 = k / (m * V1);
                                        double T1 =
                                                -(g + h * V1) / (i + j * V1);
                                        double T2 =
                                                -(g + h * V2) / (i + j * V2);
                                        double U1 =
                                                -(c.dot(e) + c.dot(f) * V1) /
                                                (d.dot(e) + d.dot(f) * V1);
                                        double U2 =
                                                -(c.dot(e) + c.dot(f) * V2) /
                                                (d.dot(e) + d.dot(f) * V2);

                                        if (V1 >= -eps && T1 >= -eps &&
                                            U1 >= -eps && V1 <= 1.0 + eps &&
                                            T1 <= 1.0 + eps &&
                                            U1 <= 1.0 + eps) {
                                            n1[nSolutions] =
                                                    (a + b * T1).normalized();
                                            n2[nSolutions] =
                                                    (c + d * U1).normalized();
                                            n3[nSolutions] =
                                                    (e + f * V1).normalized();

                                            if (std::abs(n1[nSolutions].dot(
                                                        n2[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n1[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n2[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps)
                                                ++nSolutions;
                                        }
                                        if (V2 >= -eps && T2 >= -eps &&
                                            U2 >= -eps && V2 <= 1.0 + eps &&
                                            T2 <= 1.0 + eps &&
                                            U2 <= 1.0 + eps) {
                                            n1[nSolutions] =
                                                    (a + b * T2).normalized();
                                            n2[nSolutions] =
                                                    (c + d * U2).normalized();
                                            n3[nSolutions] =
                                                    (e + f * V2).normalized();
                                            if (std::abs(n1[nSolutions].dot(
                                                        n2[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n1[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n2[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps)
                                                ++nSolutions;
                                        }
                                        if (s < 1e-4 && nSolutions == 2) {
                                            nSolutions = 1;
                                        }
                                    }
                                }
                            }

                            for (int s = 0; s < nSolutions; ++s) {
                                const auto& hullVi = hullV[edges[edgeI].first];
                                const auto& hullVj = hullV[edges[edgeJ].first];
                                const auto& hullVk = hullV[edges[edgeK].first];
                                const auto& n1_ = n1[s];
                                const auto& n2_ = n2[s];
                                const auto& n3_ = n3[s];

                                // Compute the most extreme points in each
                                // direction.
                                double maxN1 = n1_.dot(hullVi);
                                double maxN2 = n2_.dot(hullVj);
                                double maxN3 = n3_.dot(hullVk);
                                double minN1 =
                                        std::numeric_limits<double>::infinity();
                                double minN2 =
                                        std::numeric_limits<double>::infinity();
                                double minN3 =
                                        std::numeric_limits<double>::infinity();

                                const auto& antipodalI =
                                        antipodalPointsForEdge[edgeI];
                                const auto& antipodalJ =
                                        antipodalPointsForEdge[edgeJ];
                                const auto& antipodalK =
                                        antipodalPointsForEdge[edgeK];

                                // Determine the minimum projections along each
                                // axis over respective antipodal sets.
                                for (int vIdx : antipodalI) {
                                    minN1 = std::min(minN1,
                                                     n1_.dot(hullV[vIdx]));
                                }
                                for (int vIdx : antipodalJ) {
                                    minN2 = std::min(minN2,
                                                     n2_.dot(hullV[vIdx]));
                                }
                                for (int vIdx : antipodalK) {
                                    minN3 = std::min(minN3,
                                                     n3_.dot(hullV[vIdx]));
                                }

                                // Compute volume based on extents in the three
                                // principal directions.
                                double extent0 = maxN1 - minN1;
                                double extent1 = maxN2 - minN2;
                                double extent2 = maxN3 - minN3;
                                double volume = extent0 * extent1 * extent2;

                                // Update the minimum oriented bounding box if a
                                // smaller volume is found.
                                if (volume < minVolume) {
                                    // Update rotation matrix columns.
                                    minOBB.R_.col(0) = n1_;
                                    minOBB.R_.col(1) = n2_;
                                    minOBB.R_.col(2) = n3_;

                                    // Update extents.
                                    minOBB.extent_(0) = extent0;
                                    minOBB.extent_(1) = extent1;
                                    minOBB.extent_(2) = extent2;

                                    // Compute the center of the OBB using
                                    // midpoints along each axis.
                                    minOBB.center_ =
                                            (minN1 + 0.5 * extent0) * n1_ +
                                            (minN2 + 0.5 * extent1) * n2_ +
                                            (minN3 + 0.5 * extent2) * n3_;

                                    minVolume = volume;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // 5) Test all configurations where two edges are on opposing faces,
    //    ,and the third one is on a face adjacent to the two.
    // --------------------------------------------------------------------

    {
        std::vector<int> antipodalEdges;
        antipodalEdges.reserve(128);
        std::vector<Eigen::Vector3d> antipodalEdgeNormals;
        antipodalEdgeNormals.reserve(128);

        // Iterate over each edgeI in spatialEdgeOrder.
        for (int edgeI : spatialEdgeOrder) {
            // Cache face indices and normals for edgeI.
            auto [faceI_a, faceI_b] = facesForEdge[edgeI];
            const Eigen::Vector3d& f1a = faceNormals[faceI_a];
            const Eigen::Vector3d& f1b = faceNormals[faceI_b];

            antipodalEdges.clear();
            antipodalEdgeNormals.clear();

            // Iterate over vertices antipodal to edgeI.
            const auto& antipodalsForI = antipodalPointsForEdge[edgeI];
            for (int antipodalVertex : antipodalsForI) {
                const auto& neighbors = adjacencyData[antipodalVertex];
                for (int vAdj : neighbors) {
                    if (vAdj < antipodalVertex) continue;

                    int edgeIndex = antipodalVertex * numVertices + vAdj;
                    int edge = vertexPairsToEdges[edgeIndex];

                    if (edgeI > edge) continue;
                    if (isInternalEdge(edge)) continue;

                    auto [faceJ_a, faceJ_b] = facesForEdge[edge];
                    const Eigen::Vector3d& f2a = faceNormals[faceJ_a];
                    const Eigen::Vector3d& f2b = faceNormals[faceJ_b];

                    Eigen::Vector3d n;

                    bool areCompatibleOpposingEdges = false;
                    constexpr double tooCloseToFaceEpsilon = 1e-4;

                    Eigen::Matrix3d A;
                    A.row(0) = f2b;
                    A.row(1) = f1a - f1b;
                    A.row(2) = f2a - f2b;
                    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> solver(A);
                    Eigen::Vector3d x = solver.solve(-f1b);
                    double c = x(0);
                    double t = x(1);
                    double cu = x(2);

                    if (c <= 0.0 || t < 0.0 || t > 1.0) {
                        areCompatibleOpposingEdges = false;
                    } else {
                        double u = cu / c;
                        if (t < tooCloseToFaceEpsilon ||
                            t > 1.0 - tooCloseToFaceEpsilon ||
                            u < tooCloseToFaceEpsilon ||
                            u > 1.0 - tooCloseToFaceEpsilon) {
                            areCompatibleOpposingEdges = false;
                        } else {
                            if (cu < 0.0 || cu > c) {
                                areCompatibleOpposingEdges = false;
                            } else {
                                n = f1b + (f1a - f1b) * t;
                                areCompatibleOpposingEdges = true;
                            }
                        }
                    }

                    if (areCompatibleOpposingEdges) {
                        antipodalEdges.push_back(edge);
                        antipodalEdgeNormals.push_back(n.normalized());
                    }
                }
            }

            auto moveSign = [](double& dst, double& src) {
                if (src < 0.0) {
                    dst = -dst;
                    src = -src;
                }
            };

            const auto& compatibleEdgesI = compatibleEdges[edgeI];
            for (int edgeJ : compatibleEdgesI) {
                for (size_t k = 0; k < antipodalEdges.size(); ++k) {
                    int edgeK = antipodalEdges[k];

                    const Eigen::Vector3d& n1 = antipodalEdgeNormals[k];
                    double minN1 = n1.dot(hullV[edges[edgeK].first]);
                    double maxN1 = n1.dot(hullV[edges[edgeI].first]);

                    // Test all mutual compatible edges.
                    auto [faceK_a, faceK_b] = facesForEdge[edgeJ];
                    const Eigen::Vector3d& f3a = faceNormals[faceK_a];
                    const Eigen::Vector3d& f3b = faceNormals[faceK_b];

                    double num = n1.dot(f3b);
                    double den = n1.dot(f3b - f3a);
                    moveSign(num, den);

                    constexpr double epsilon = 1e-4;
                    if (den < epsilon) {
                        num = (std::abs(num) < 1e-4) ? 0.0 : -1.0;
                        den = 1.0;
                    }

                    if (num >= den * -epsilon && num <= den * (1.0 + epsilon)) {
                        double v = num / den;
                        Eigen::Vector3d n3 =
                                (f3b + (f3a - f3b) * v).normalized();
                        Eigen::Vector3d n2 = n3.cross(n1).normalized();

                        double maxN2, minN2;
                        clearGraphSearch();
                        int hint = extremeVertexConvex(
                                extremeVertexConvex, n2, floodFillVisited,
                                floodFillVisitColor, maxN2,
                                (k == 0) ? vHint1 : vHint1_b);
                        if (k == 0) {
                            vHint1 = vHint1_b = hint;
                        } else {
                            vHint1_b = hint;
                        }

                        clearGraphSearch();
                        hint = extremeVertexConvex(
                                extremeVertexConvex, -n2, floodFillVisited,
                                floodFillVisitColor, minN2,
                                (k == 0) ? vHint2 : vHint2_b);
                        if (k == 0) {
                            vHint2 = vHint2_b = hint;
                        } else {
                            vHint2_b = hint;
                        }

                        minN2 = -minN2;

                        double maxN3 = n3.dot(hullV[edges[edgeJ].first]);
                        double minN3 = std::numeric_limits<double>::infinity();

                        // If there are very few antipodal vertices, do a
                        // very tight loop and just iterate over each.
                        const auto& antipodalsEdge =
                                antipodalPointsForEdge[edgeJ];
                        if (antipodalsEdge.size() < 20) {
                            for (int vIdx : antipodalsEdge) {
                                minN3 = std::min(minN3, n3.dot(hullV[vIdx]));
                            }
                        } else {
                            // Otherwise perform a spatial locality
                            // exploiting graph search.
                            clearGraphSearch();
                            hint = extremeVertexConvex(
                                    extremeVertexConvex, -n3, floodFillVisited,
                                    floodFillVisitColor, minN3,
                                    (k == 0) ? vHint3 : vHint3_b);

                            if (k == 0) {
                                vHint3 = vHint3_b = hint;
                            } else {
                                vHint3_b = hint;
                            }

                            minN3 = -minN3;
                        }

                        double volume = (maxN1 - minN1) * (maxN2 - minN2) *
                                        (maxN3 - minN3);
                        if (volume < minVolume) {
                            minOBB.R_.col(0) = n1;
                            minOBB.R_.col(1) = n2;
                            minOBB.R_.col(2) = n3;
                            minOBB.extent_(0) = (maxN1 - minN1);
                            minOBB.extent_(1) = (maxN2 - minN2);
                            minOBB.extent_(2) = (maxN3 - minN3);
                            minOBB.center_ = 0.5 * ((minN1 + maxN1) * n1 +
                                                    (minN2 + maxN2) * n2 +
                                                    (minN3 + maxN3) * n3);
                            minVolume = volume;
                        }
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // 6) Test all configurations where two edges are on the same face (OBB
    //    aligns with a face of the convex hull)
    // --------------------------------------------------------------------
    {
        // Preallocate vectors to avoid frequent reallocations.
        std::vector<int> antipodalEdges;
        antipodalEdges.reserve(128);
        std::vector<Eigen::Vector3d> antipodalEdgeNormals;
        antipodalEdgeNormals.reserve(128);

        for (int faceIdx : spatialFaceOrder) {
            const Eigen::Vector3d& n1 = faceNormals[faceIdx];

            // Find two edges on the face. Since we have flexibility to
            // choose from multiple edges of the same face, choose two that
            // are possibly most opposing to each other, in the hope that
            // their sets of sidepodal edges are most mutually exclusive as
            // possible, speeding up the search below.
            int e1 = -1;
            const auto& tri = hullT[faceIdx];
            int v0 = tri(2);
            for (int j = 0; j < 3; ++j) {
                int v1 = tri(j);
                int e = vertexPairsToEdges[v0 * numVertices + v1];
                if (!isInternalEdge(e)) {
                    e1 = e;
                    break;
                }
                v0 = v1;
            }

            if (e1 == -1) continue;

            // Compute minN1 either by scanning antipodal points or using
            // ExtremeVertexConvex.
            double maxN1 = n1.dot(hullV[edges[e1].first]);
            double minN1 = std::numeric_limits<double>::infinity();
            const auto& antipodals = antipodalPointsForEdge[e1];
            if (antipodals.size() < 20) {
                minN1 = std::numeric_limits<double>::infinity();
                for (int vIdx : antipodals) {
                    minN1 = std::min(minN1, n1.dot(hullV[vIdx]));
                }
            } else {
                clearGraphSearch();
                vHint4 = extremeVertexConvex(
                        extremeVertexConvex, -n1, floodFillVisited,
                        floodFillVisitColor, minN1, vHint4);
                minN1 = -minN1;
            }

            // Check edges compatible with e1.
            const auto& compatibleEdgesI = compatibleEdges[e1];
            for (int edgeK : compatibleEdgesI) {
                auto [faceK_a, faceK_b] = facesForEdge[edgeK];
                const Eigen::Vector3d& f3a = faceNormals[faceK_a];
                const Eigen::Vector3d& f3b = faceNormals[faceK_b];

                // Is edge3 compatible with direction n?
                double num = n1.dot(f3b);
                double den = n1.dot(f3b - f3a);
                double v;
                constexpr double epsilon = 1e-4;
                if (std::abs(den) >= epsilon) {
                    v = num / den;
                } else {
                    v = (std::abs(num) < epsilon) ? 0.0 : -1.0;
                }

                if (v >= -epsilon && v <= 1.0 + epsilon) {
                    Eigen::Vector3d n3 = (f3b + (f3a - f3b) * v).normalized();
                    Eigen::Vector3d n2 = n3.cross(n1).normalized();

                    double maxN2, minN2;
                    clearGraphSearch();
                    vHint1 = extremeVertexConvex(
                            extremeVertexConvex, n2, floodFillVisited,
                            floodFillVisitColor, maxN2, vHint1);
                    clearGraphSearch();
                    vHint2 = extremeVertexConvex(
                            extremeVertexConvex, -n2, floodFillVisited,
                            floodFillVisitColor, minN2, vHint2);
                    minN2 = -minN2;

                    double maxN3 = n3.dot(hullV[edges[edgeK].first]);
                    double minN3 = std::numeric_limits<double>::infinity();

                    // If there are very few antipodal vertices, do a very
                    // tight loop and just iterate over each.
                    const auto& antipodalsEdge = antipodalPointsForEdge[edgeK];
                    if (antipodalsEdge.size() < 20) {
                        for (int vIdx : antipodalsEdge) {
                            minN3 = std::min(minN3, n3.dot(hullV[vIdx]));
                        }
                    } else {
                        clearGraphSearch();
                        vHint3 = extremeVertexConvex(
                                extremeVertexConvex, -n3, floodFillVisited,
                                floodFillVisitColor, minN3, vHint3);
                        minN3 = -minN3;
                    }

                    double volume =
                            (maxN1 - minN1) * (maxN2 - minN2) * (maxN3 - minN3);
                    if (volume < minVolume) {
                        minOBB.R_.col(0) = n1;
                        minOBB.R_.col(1) = n2;
                        minOBB.R_.col(2) = n3;
                        minOBB.extent_(0) = (maxN1 - minN1);
                        minOBB.extent_(1) = (maxN2 - minN2);
                        minOBB.extent_(2) = (maxN3 - minN3);
                        minOBB.center_ = 0.5 * ((minN1 + maxN1) * n1 +
                                                (minN2 + maxN2) * n2 +
                                                (minN3 + maxN3) * n3);
                        assert(volume > 0.0);
                        minVolume = volume;
                    }
                }
            }
        }
    }

    // Final check to ensure right-handed coordinate frame
    if (minOBB.R_.col(0).cross(minOBB.R_.col(1)).dot(minOBB.R_.col(2)) < 0.0) {
        minOBB.R_.col(2) = -minOBB.R_.col(2);
    }
    mapOBBToClosestIdentity(minOBB);
    return minOBB;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Clear() {
    min_bound_.setZero();
    max_bound_.setZero();
    return *this;
}

bool AxisAlignedBoundingBox::IsEmpty() const { return Volume() <= 0; }

Eigen::Vector3d AxisAlignedBoundingBox::GetMinBound() const {
    return min_bound_;
}

Eigen::Vector3d AxisAlignedBoundingBox::GetMaxBound() const {
    return max_bound_;
}

Eigen::Vector3d AxisAlignedBoundingBox::GetCenter() const {
    return (min_bound_ + max_bound_) * 0.5;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::GetAxisAlignedBoundingBox()
        const {
    return *this;
}

OrientedBoundingBox AxisAlignedBoundingBox::GetOrientedBoundingBox(
        bool robust) const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

OrientedBoundingBox AxisAlignedBoundingBox::GetMinimalOrientedBoundingBox(
        bool robust) const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const Eigen::Vector3d& min_bound,
                                               const Eigen::Vector3d& max_bound)
    : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
      min_bound_(min_bound),
      max_bound_(max_bound),
      color_(1, 1, 1) {
    if ((max_bound_.array() < min_bound_.array()).any()) {
        open3d::utility::LogWarning(
                "max_bound {} of bounding box is smaller than min_bound {} "
                "in "
                "one or more axes. Fix input values to remove this "
                "warning.",
                max_bound_, min_bound_);
        max_bound_ = max_bound.cwiseMax(min_bound);
        min_bound_ = max_bound.cwiseMin(min_bound);
    }
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of a AxisAlignedBoundingBox would not be "
            "axis "
            "aligned anymore, convert it to a OrientedBoundingBox first");
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        min_bound_ += translation;
        max_bound_ += translation;
    } else {
        const Eigen::Vector3d half_extent = GetHalfExtent();
        min_bound_ = translation - half_extent;
        max_bound_ = translation + half_extent;
    }
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Scale(
        const double scale, const Eigen::Vector3d& center) {
    min_bound_ = center + scale * (min_bound_ - center);
    max_bound_ = center + scale * (max_bound_ - center);
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Rotate(
        const Eigen::Matrix3d& rotation, const Eigen::Vector3d& center) {
    utility::LogError(
            "A rotation of an AxisAlignedBoundingBox would not be "
            "axis-aligned "
            "anymore, convert it to an OrientedBoundingBox first");
    return *this;
}

std::string AxisAlignedBoundingBox::GetPrintInfo() const {
    return fmt::format("[({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, {:.4f})]",
                       min_bound_(0), min_bound_(1), min_bound_(2),
                       max_bound_(0), max_bound_(1), max_bound_(2));
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::operator+=(
        const AxisAlignedBoundingBox& other) {
    if (IsEmpty()) {
        min_bound_ = other.min_bound_;
        max_bound_ = other.max_bound_;
    } else if (!other.IsEmpty()) {
        min_bound_ = min_bound_.array().min(other.min_bound_.array()).matrix();
        max_bound_ = max_bound_.array().max(other.max_bound_.array()).matrix();
    }
    return *this;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points) {
    AxisAlignedBoundingBox box;
    if (points.empty()) {
        utility::LogWarning(
                "The number of points is 0 when creating axis-aligned "
                "bounding "
                "box.");
        box.min_bound_ = Eigen::Vector3d(0.0, 0.0, 0.0);
        box.max_bound_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    } else {
        box.min_bound_ = std::accumulate(
                points.begin(), points.end(), points[0],
                [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                    return a.array().min(b.array()).matrix();
                });
        box.max_bound_ = std::accumulate(
                points.begin(), points.end(), points[0],
                [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                    return a.array().max(b.array()).matrix();
                });
    }
    return box;
}

double AxisAlignedBoundingBox::Volume() const { return GetExtent().prod(); }

std::vector<Eigen::Vector3d> AxisAlignedBoundingBox::GetBoxPoints() const {
    std::vector<Eigen::Vector3d> points(8);
    Eigen::Vector3d extent = GetExtent();
    points[0] = min_bound_;
    points[1] = min_bound_ + Eigen::Vector3d(extent(0), 0, 0);
    points[2] = min_bound_ + Eigen::Vector3d(0, extent(1), 0);
    points[3] = min_bound_ + Eigen::Vector3d(0, 0, extent(2));
    points[4] = max_bound_;
    points[5] = max_bound_ - Eigen::Vector3d(extent(0), 0, 0);
    points[6] = max_bound_ - Eigen::Vector3d(0, extent(1), 0);
    points[7] = max_bound_ - Eigen::Vector3d(0, 0, extent(2));
    return points;
}

std::vector<size_t> AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<Eigen::Vector3d>& points) const {
    std::vector<size_t> indices;
    for (size_t idx = 0; idx < points.size(); idx++) {
        const auto& point = points[idx];
        if (point(0) >= min_bound_(0) && point(0) <= max_bound_(0) &&
            point(1) >= min_bound_(1) && point(1) <= max_bound_(1) &&
            point(2) >= min_bound_(2) && point(2) <= max_bound_(2)) {
            indices.push_back(idx);
        }
    }
    return indices;
}

}  // namespace geometry
}  // namespace open3d
