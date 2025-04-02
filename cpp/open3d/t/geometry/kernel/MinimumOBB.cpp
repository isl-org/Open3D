// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/MinimumOBB.h"

#include "open3d/core/EigenConverter.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/t/geometry/PointCloud.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace minimum_obb {

namespace {

// Helper struct using Eigen datatypes on the stack for OBB computation
struct EigenOBB {
    EigenOBB()
        : R_(Eigen::Matrix3d::Identity()),
          extent_(Eigen::Vector3d::Zero()),
          center_(Eigen::Vector3d::Zero()) {}
    EigenOBB(const OrientedBoundingBox& obb)
        : R_(core::eigen_converter::TensorToEigenMatrixXd(obb.GetRotation())),
          extent_(core::eigen_converter::TensorToEigenMatrixXd(
                  obb.GetExtent().Reshape({3, 1}))),
          center_(core::eigen_converter::TensorToEigenMatrixXd(
                  obb.GetCenter().Reshape({3, 1}))) {}
    double Volume() const { return extent_(0) * extent_(1) * extent_(2); }
    operator OrientedBoundingBox() const {
        OrientedBoundingBox obb;
        obb.SetRotation(core::eigen_converter::EigenMatrixToTensor(R_));
        obb.SetExtent(
                core::eigen_converter::EigenMatrixToTensor(extent_).Reshape(
                        {3}));
        obb.SetCenter(
                core::eigen_converter::EigenMatrixToTensor(center_).Reshape(
                        {3}));
        return obb;
    }
    Eigen::Matrix3d R_;
    Eigen::Vector3d extent_;
    Eigen::Vector3d center_;
};
}  // namespace

OrientedBoundingBox ComputeMinimumOBBJylanki(const core::Tensor& points_,
                                             bool robust) {
    // ------------------------------------------------------------
    // 0) Compute the convex hull of the input point cloud
    // ------------------------------------------------------------
    core::AssertTensorShape(points_, {utility::nullopt, 3});
    if (points_.GetShape(0) == 0) {
        utility::LogError("Input point set is empty.");
        return OrientedBoundingBox();
    }
    if (points_.GetShape(0) < 4) {
        utility::LogError("Input point set has less than 4 points.");
        return OrientedBoundingBox();
    }
    // copy to CPU here
    PointCloud pcd(points_.To(core::Device()));
    auto hull_mesh = pcd.ComputeConvexHull(robust);
    if (hull_mesh.GetVertexPositions().NumElements() == 0) {
        utility::LogError("Failed to compute convex hull.");
        return OrientedBoundingBox();
    }

    // Get convex hull vertices and triangles
    const std::vector<Eigen::Vector3d>& hull_v =
            core::eigen_converter::TensorToEigenVector3dVector(
                    hull_mesh.GetVertexPositions());
    const std::vector<Eigen::Vector3i>& hull_t =
            core::eigen_converter::TensorToEigenVector3iVector(
                    hull_mesh.GetTriangleIndices());
    int num_vertices = static_cast<int>(hull_v.size());
    int num_triangles = static_cast<int>(hull_t.size());

    EigenOBB min_obb(AxisAlignedBoundingBox::CreateFromPoints(
                             hull_mesh.GetVertexPositions())
                             .GetOrientedBoundingBox());
    double min_volume = min_obb.Volume();

    // Handle degenerate planar cases up front.
    if (num_vertices <= 3 || num_triangles < 1) {  // Handle degenerate case
        utility::LogError("Convex hull is degenerate.");
        return OrientedBoundingBox();
    }

    auto mapOBBToClosestIdentity = [&](EigenOBB obb) {
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
    };

    // --------------------------------------------------------------------
    // 1) Precompute vertex adjacency data, face normals, and edge data
    // --------------------------------------------------------------------
    std::vector<std::vector<int>> adjacency_data;
    adjacency_data.reserve(num_vertices);
    adjacency_data.insert(adjacency_data.end(), num_vertices,
                          std::vector<int>());

    std::vector<Eigen::Vector3d> face_normals;
    face_normals.reserve(num_triangles);

    // Each edge is stored as (v0, v1).
    std::vector<std::pair<int, int>> edges;
    edges.reserve(num_vertices * 2);

    // Each edge knows which two faces it belongs to: (f0, f1).
    std::vector<std::pair<int, int>> faces_for_edge;
    faces_for_edge.reserve(num_vertices * 2);

    constexpr unsigned int empty_edge =
            std::numeric_limits<unsigned int>::max();
    std::vector<unsigned int> vertex_pairs_to_edges(num_vertices * num_vertices,
                                                    empty_edge);

    for (int i = 0; i < num_triangles; ++i) {
        const Eigen::Vector3i& tri = hull_t[i];
        int t0 = tri(0), t1 = tri(1), t2 = tri(2);
        int v0 = t2, v1 = t0;

        for (int j = 0; j < 3; ++j) {
            v1 = tri(j);

            // Build Adjacency Data (vertex -> adjacent vertices)
            adjacency_data[v0].push_back(v1);

            // Register Edges (edge -> neighbouring faces)
            unsigned int& ref_idx1 =
                    vertex_pairs_to_edges[v0 * num_vertices + v1];
            unsigned int& ref_idx2 =
                    vertex_pairs_to_edges[v1 * num_vertices + v0];
            if (ref_idx1 == empty_edge) {
                // Not registered yet
                unsigned int new_idx = static_cast<unsigned int>(edges.size());
                ref_idx1 = new_idx;
                ref_idx2 = new_idx;
                edges.emplace_back(v0, v1);
                faces_for_edge.emplace_back(i, -1);
            } else {
                // Already existing, update the second face index
                faces_for_edge[ref_idx1].second = i;
            }

            v0 = v1;
        }
        // Compute Face Normal
        auto n = (hull_v[t1] - hull_v[t0]).cross(hull_v[t2] - hull_v[t0]);
        face_normals.push_back(n.normalized());
    }

    // ------------------------------------------------------------
    // 2) Precompute "antipodal vertices" for each edge of the hull
    // ------------------------------------------------------------

    // Throughout the algorithm, internal edges can all be discarded.
    auto isInternalEdge = [&](std::size_t iEdge) noexcept {
        return (face_normals[faces_for_edge[iEdge].first].dot(
                        face_normals[faces_for_edge[iEdge].second]) >
                1.0 - 1e-4);
    };

    // Throughout the whole algorithm, this array stores an auxiliary structure
    // for performing graph searches on the vertices of the convex hull.
    // Conceptually each index of the array stores a boolean whether we have
    // visited that vertex or not during the current search. However storing
    // such booleans is slow, since we would have to perform a linear-time scan
    // through this array before next search to reset each boolean to unvisited
    // false state. Instead, store a number, called a "color" for each vertex to
    // specify whether that vertex has been visited, and manage a global color
    // counter flood_fill_visit_color that represents the visited vertices. At
    // any given time, the vertices that have already been visited have the
    // value flood_fill_visited[i] == flood_fill_visit_color in them. This gives
    // a win that we can perform constant-time clears of the flood_fill_visited
    // array, by simply incrementing the "color" counter to clear the array.

    int edge_size = edges.size();
    std::vector<std::vector<int>> antipodal_points_for_edge(edge_size);
    antipodal_points_for_edge.reserve(edge_size);

    std::vector<unsigned int> flood_fill_visited(num_vertices, 0u);
    unsigned int flood_fill_visit_color = 1u;

    auto markVertexVisited = [&](int v) {
        flood_fill_visited[v] = flood_fill_visit_color;
    };

    auto haveVisitedVertex = [&](int v) -> bool {
        return flood_fill_visited[v] == flood_fill_visit_color;
    };

    auto clearGraphSearch = [&]() { ++flood_fill_visit_color; };

    auto isVertexAntipodalToEdge =
            [&](int vi, const std::vector<int>& neighbors,
                const Eigen::Vector3d& f1a,
                const Eigen::Vector3d& f1b) noexcept -> bool {
        constexpr double epsilon = 1e-4;
        constexpr double degenerate_threshold = -5e-2;
        double t_min = 0.0;
        double t_max = 1.0;

        // Precompute values outside the loop for efficiency.
        const auto& v = hull_v[vi];
        Eigen::Vector3d f1b_f1a = f1b - f1a;

        // Iterate over each neighbor.
        for (int neighbor_index : neighbors) {
            const auto& neighbor = hull_v[neighbor_index];

            // Compute edge vector e = neighbor - v.
            Eigen::Vector3d e = neighbor - v;

            // Compute dot products manually for efficiency.
            double s = f1b_f1a.dot(e);
            double n = f1b.dot(e);

            // Adjust t_min and t_max based on the value of s.
            if (s > epsilon) {
                t_max = std::min(t_max, n / s);
            } else if (s < -epsilon) {
                t_min = std::max(t_min, n / s);
            } else if (n < -epsilon) {
                // No feasible t if n is negative when s is nearly zero.
                return false;
            }

            // If the valid interval for t has degenerated, exit early.
            if ((t_max - t_min) < degenerate_threshold) {
                return false;
            }
        }
        return true;
    };

    auto extremeVertexConvex =
            [&](auto& self, const Eigen::Vector3d& direction,
                std::vector<unsigned int>& flood_fill_visited,
                unsigned int flood_fill_visit_color,
                double& most_extreme_distance, int starting_vertex) -> int {
        // Compute dot product for the starting vertex.
        double cur_dot = direction.dot(hull_v[starting_vertex]);

        // Cache neighbor list for the starting vertex.
        const int* neighbors = &adjacency_data[starting_vertex][0];
        const int* neighbors_end =
                neighbors + adjacency_data[starting_vertex].size();

        // Mark starting vertex as visited.
        flood_fill_visited[starting_vertex] = flood_fill_visit_color;

        // Traverse neighbors to find more extreme vertices.
        int second_best = -1;
        double second_best_dot = cur_dot - 1e-3;
        while (neighbors != neighbors_end) {
            int n = *neighbors++;
            if (flood_fill_visited[n] != flood_fill_visit_color) {
                double dot = direction.dot(hull_v[n]);
                if (dot > cur_dot) {
                    // Found a new vertex with higher dot product.
                    starting_vertex = n;
                    cur_dot = dot;
                    flood_fill_visited[starting_vertex] =
                            flood_fill_visit_color;
                    neighbors = &adjacency_data[starting_vertex][0];
                    neighbors_end =
                            neighbors + adjacency_data[starting_vertex].size();
                    second_best = -1;
                    second_best_dot = cur_dot - 1e-3;
                } else if (dot > second_best_dot) {
                    // Update second-best candidate.
                    second_best = n;
                    second_best_dot = dot;
                }
            }
        }

        // Explore second-best neighbor recursively if valid.
        if (second_best != -1 &&
            flood_fill_visited[second_best] != flood_fill_visit_color) {
            double second_most_extreme =
                    -std::numeric_limits<double>::infinity();
            int second_try = self(self, direction, flood_fill_visited,
                                  flood_fill_visit_color, second_most_extreme,
                                  second_best);

            if (second_most_extreme > cur_dot) {
                most_extreme_distance = second_most_extreme;
                return second_try;
            }
        }

        most_extreme_distance = cur_dot;
        return starting_vertex;
    };

    // The currently best variant for establishing a spatially coherent
    // traversal order.
    std::vector<int> spatial_face_order;
    spatial_face_order.reserve(num_triangles);
    std::vector<int> spatial_edge_order;
    spatial_edge_order.reserve(edge_size);

    // Initialize random number generator
    std::random_device rd;   // Obtain a random number from hardware
    std::mt19937 rng(rd());  // Seed the generator
    {  // Explicit scope for variables that are not needed after this.

        std::vector<unsigned int> visited_edges(edge_size, 0u);
        std::vector<unsigned int> visited_faces(num_triangles, 0u);

        std::vector<std::pair<int, int>> traverse_stack_edges;
        traverse_stack_edges.reserve(edge_size);
        traverse_stack_edges.emplace_back(0, adjacency_data[0].front());
        while (!traverse_stack_edges.empty()) {
            auto e = traverse_stack_edges.back();
            traverse_stack_edges.pop_back();

            // Find edge index
            int edge_idx =
                    vertex_pairs_to_edges[e.first * num_vertices + e.second];
            if (visited_edges[edge_idx]) continue;
            visited_edges[edge_idx] = 1;
            auto& ff = faces_for_edge[edge_idx];
            if (!visited_faces[ff.first]) {
                visited_faces[ff.first] = 1;
                spatial_face_order.push_back(ff.first);
            }
            if (!visited_faces[ff.second]) {
                visited_faces[ff.second] = 1;
                spatial_face_order.push_back(ff.second);
            }

            // If not an internal edge, keep it
            if (!isInternalEdge(edge_idx)) {
                spatial_edge_order.push_back(edge_idx);
            }

            int v0 = e.second;
            size_t size_before = traverse_stack_edges.size();
            for (int v1 : adjacency_data[v0]) {
                int e1 = vertex_pairs_to_edges[v0 * num_vertices + v1];
                if (visited_edges[e1]) continue;
                traverse_stack_edges.push_back(std::make_pair(v0, v1));
            }

            // Randomly shuffle newly added edges
            int n_new_edges =
                    static_cast<int>(traverse_stack_edges.size() - size_before);
            if (n_new_edges > 0) {
                std::uniform_int_distribution<> distr(0, n_new_edges - 1);
                int r = distr(rng);
                std::swap(traverse_stack_edges.back(),
                          traverse_stack_edges[size_before + r]);
            }
        }
    }

    // --------------------------------------------------------------------
    // 3) Precompute "sidepodal edges" for each edge of the hull
    // --------------------------------------------------------------------

    // Stores a memory of yet unvisited vertices for current graph search.
    std::vector<int> traverse_stack;

    // Since we do several extreme vertex searches, and the search directions
    // have a lot of spatial locality, always start the search for the next
    // extreme vertex from the extreme vertex that was found during the previous
    // iteration for the previous edge. This has been profiled to improve
    // overall performance by as much as 15-25%.
    int start_vertex = 0;

    // Precomputation: for each edge, we need to compute the list of potential
    // antipodal points (points on the opposing face of an enclosing OBB of the
    // face that is flush with the given edge of the polyhedron).
    for (int edge_i : spatial_edge_order) {
        auto [face_i_a, face_i_b] = faces_for_edge[edge_i];
        const Eigen::Vector3d& f1a = face_normals[face_i_a];
        const Eigen::Vector3d& f1b = face_normals[face_i_b];

        double dummy;
        clearGraphSearch();
        start_vertex = extremeVertexConvex(
                extremeVertexConvex, -f1a, flood_fill_visited,
                flood_fill_visit_color, dummy, start_vertex);
        clearGraphSearch();

        traverse_stack.push_back(start_vertex);
        markVertexVisited(start_vertex);
        while (!traverse_stack.empty()) {
            int v = traverse_stack.back();
            traverse_stack.pop_back();
            const auto& neighbors = adjacency_data[v];
            if (isVertexAntipodalToEdge(v, neighbors, f1a, f1b)) {
                if (edges[edge_i].first == v || edges[edge_i].second == v) {
                    return OrientedBoundingBox();
                }
                antipodal_points_for_edge[edge_i].push_back(v);
                for (size_t j = 0; j < neighbors.size(); ++j) {
                    if (!haveVisitedVertex(neighbors[j])) {
                        traverse_stack.push_back(neighbors[j]);
                        markVertexVisited(neighbors[j]);
                    }
                }
            }
        }

        // Robustness: If the above search did not find any antipodal points,
        // add the first found extreme point at least, since it is always an
        // antipodal point. This is known to occur very rarely due to numerical
        // imprecision in the above loop over adjacent edges.
        if (antipodal_points_for_edge[edge_i].empty()) {
            // Getting here is most likely a bug. Fall back to linear scan,
            // which is very slow.
            for (int j = 0; j < num_vertices; ++j) {
                if (isVertexAntipodalToEdge(j, adjacency_data[j], f1a, f1b)) {
                    antipodal_points_for_edge[edge_i].push_back(j);
                }
            }
        }
    }

    // Data structure for sidepodal vertices.
    std::vector<unsigned char> sidepodal_vertices(edge_size * num_vertices, 0);

    // Stores for each edge i the list of all sidepodal edge indices j that it
    // can form an OBB with.
    std::vector<std::vector<int>> compatible_edges(edge_size);
    compatible_edges.reserve(edge_size);

    // Compute all sidepodal edges for each edge by performing a graph search.
    // The set of sidepodal edges is connected in the graph, which lets us avoid
    // having to iterate over each edge pair of the convex hull.
    for (int edge_i : spatial_edge_order) {
        auto [face_i_a, face_i_b] = faces_for_edge[edge_i];
        const Eigen::Vector3d& f1a = face_normals[face_i_a];
        const Eigen::Vector3d& f1b = face_normals[face_i_b];

        // Pixar orthonormal basis code:
        // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        Eigen::Vector3d dead_direction = (f1a + f1b) * 0.5;
        Eigen::Vector3d basis1, basis2;
        double sign = std::copysign(1.0, dead_direction.z());
        const double a = -1.0 / (sign + dead_direction.z());
        const double b = dead_direction.x() * dead_direction.y() * a;
        basis1 = Eigen::Vector3d(
                1.0 + sign * dead_direction.x() * dead_direction.x() * a,
                sign * b, -sign * dead_direction.x());
        basis2 = Eigen::Vector3d(
                b, sign + dead_direction.y() * dead_direction.y() * a,
                -dead_direction.y());

        double dummy;
        Eigen::Vector3d dir =
                (f1a.cross(Eigen::Vector3d(0, 1, 0))).normalized();
        if (dir.norm() < 1e-4) {
            dir = Eigen::Vector3d(0, 0, 1);  // If f1a is parallel to y-axis
        }
        clearGraphSearch();
        start_vertex = extremeVertexConvex(
                extremeVertexConvex, dir, flood_fill_visited,
                flood_fill_visit_color, dummy, start_vertex);
        clearGraphSearch();
        traverse_stack.push_back(start_vertex);
        while (!traverse_stack.empty()) {
            int v = traverse_stack.back();
            traverse_stack.pop_back();

            if (haveVisitedVertex(v)) continue;
            markVertexVisited(v);

            // const auto& neighbors = adjacency_data[v];
            for (int v_adj : adjacency_data[v]) {
                if (haveVisitedVertex(v_adj)) continue;
                int edge = vertex_pairs_to_edges[v * num_vertices + v_adj];
                auto [face_i_a, face_i_b] = faces_for_edge[edge];
                Eigen::Vector3d f1a_f1b = f1a - f1b;
                Eigen::Vector3d f2a_f2b =
                        face_normals[face_i_a] - face_normals[face_i_b];

                double a2 = f1b.dot(face_normals[face_i_b]);
                double b2 = f1a_f1b.dot(face_normals[face_i_b]);
                double c2 = f2a_f2b.dot(f1b);
                double d2 = f1a_f1b.dot(f2a_f2b);
                double ab = a2 + b2;
                double ac = a2 + c2;
                double abcd = ab + c2 + d2;
                double min_val = std::min({a2, ab, ac, abcd});
                double max_val = std::max({a2, ab, ac, abcd});
                bool are_edges_compatible_for_obb =
                        (min_val <= 0.0 && max_val >= 0.0);

                if (are_edges_compatible_for_obb) {
                    if (edge_i <= edge) {
                        if (!isInternalEdge(edge)) {
                            compatible_edges[edge_i].push_back(edge);
                        }

                        sidepodal_vertices[edge_i * num_vertices +
                                           edges[edge].first] = 1;
                        sidepodal_vertices[edge_i * num_vertices +
                                           edges[edge].second] = 1;
                        if (edge_i != edge) {
                            if (!isInternalEdge(edge)) {
                                compatible_edges[edge].push_back(edge_i);
                            }
                            sidepodal_vertices[edge * num_vertices +
                                               edges[edge_i].first] = 1;
                            sidepodal_vertices[edge * num_vertices +
                                               edges[edge_i].second] = 1;
                        }
                    }
                    traverse_stack.push_back(v_adj);
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
    int v_hint1 = 0;
    int v_hint2 = 0;
    int v_hint3 = 0;
    int v_hint4 = 0;
    int v_hint1_b = 0;
    int v_hint2_b = 0;
    int v_hint3_b = 0;

    // Stores a memory of yet unvisited vertices that are common sidepodal
    // vertices to both currently chosen edges for current graph search.
    std::vector<int> traverseStackCommonSidepodals;
    traverseStackCommonSidepodals.reserve(num_vertices);
    for (int edge_i : spatial_edge_order) {
        auto [face_i_a, face_i_b] = faces_for_edge[edge_i];
        const Eigen::Vector3d& f1a = face_normals[face_i_a];
        const Eigen::Vector3d& f1b = face_normals[face_i_b];

        const auto& compatible_edges_i = compatible_edges[edge_i];
        Eigen::Vector3d baseDir = 0.5 * (f1a + f1b);

        for (int edge_j : compatible_edges_i) {
            if (edge_j <= edge_i) continue;  // Remove symmetry.
            auto [faceJ_a, faceJ_b] = faces_for_edge[edge_j];
            const Eigen::Vector3d& f2a = face_normals[faceJ_a];
            const Eigen::Vector3d& f2b = face_normals[faceJ_b];

            // Compute search direction
            Eigen::Vector3d dead_dir = 0.5 * (f2a + f2b);
            Eigen::Vector3d search_dir = baseDir.cross(dead_dir);
            search_dir = search_dir.normalized();
            if (search_dir.norm() < 1e-9) {
                search_dir = f1a.cross(f2a);
                search_dir = search_dir.normalized();
                if (search_dir.norm() < 1e-9) {
                    search_dir =
                            (f1a.cross(Eigen::Vector3d(0, 1, 0))).normalized();
                }
            }

            double dummy;
            clearGraphSearch();
            v_hint1 = extremeVertexConvex(
                    extremeVertexConvex, search_dir, flood_fill_visited,
                    flood_fill_visit_color, dummy, v_hint1);
            clearGraphSearch();
            v_hint2 = extremeVertexConvex(
                    extremeVertexConvex, -search_dir, flood_fill_visited,
                    flood_fill_visit_color, dummy, v_hint2);

            int secondSearch = -1;
            if (sidepodal_vertices[edge_j * num_vertices + v_hint1]) {
                traverseStackCommonSidepodals.push_back(v_hint1);
            } else {
                traverse_stack.push_back(v_hint1);
            }
            if (sidepodal_vertices[edge_j * num_vertices + v_hint2]) {
                traverseStackCommonSidepodals.push_back(v_hint2);
            } else {
                secondSearch = v_hint2;
            }

            // Bootstrap to a good vertex that is sidepodal to both edges.
            clearGraphSearch();
            while (!traverse_stack.empty()) {
                int v = traverse_stack.front();
                traverse_stack.erase(traverse_stack.begin());
                if (haveVisitedVertex(v)) continue;
                markVertexVisited(v);
                const auto& neighbors = adjacency_data[v];
                for (int v_adj : neighbors) {
                    if (!haveVisitedVertex(v_adj) &&
                        sidepodal_vertices[edge_i * num_vertices + v_adj]) {
                        if (sidepodal_vertices[edge_j * num_vertices + v_adj]) {
                            traverse_stack.clear();
                            if (secondSearch != -1) {
                                traverse_stack.push_back(secondSearch);
                                secondSearch = -1;
                                markVertexVisited(v_adj);
                            }
                            traverseStackCommonSidepodals.push_back(v_adj);
                            break;
                        } else {
                            traverse_stack.push_back(v_adj);
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
                const auto& neighbors = adjacency_data[v];
                for (int v_adj : neighbors) {
                    int edge_k =
                            vertex_pairs_to_edges[v * num_vertices + v_adj];
                    int idx_i = edge_i * num_vertices + v_adj;
                    int idx_j = edge_j * num_vertices + v_adj;

                    if (isInternalEdge(edge_k)) continue;

                    if (sidepodal_vertices[idx_i] &&
                        sidepodal_vertices[idx_j]) {
                        if (!haveVisitedVertex(v_adj)) {
                            traverseStackCommonSidepodals.push_back(v_adj);
                        }
                        if (edge_j < edge_k) {
                            auto [faceK_a, faceK_b] = faces_for_edge[edge_k];
                            const Eigen::Vector3d& f3a = face_normals[faceK_a];
                            const Eigen::Vector3d& f3b = face_normals[faceK_b];

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
                            constexpr double angle_eps = 1e-3;
                            int n_solutions = 0;

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
                                    n_solutions = 0;

                                    // If we happened to divide by zero above,
                                    // the following checks handle them.
                                    if (v >= -eps && t >= -eps && u >= -eps &&
                                        v <= 1.0 + eps && t <= 1.0 + eps &&
                                        u <= 1.0 + eps) {
                                        n1[0] = (a + b * t).normalized();
                                        n2[0] = (c + d * u).normalized();
                                        n3[0] = (e + f * v).normalized();
                                        if (std::abs(n1[0].dot(n2[0])) <
                                                    angle_eps &&
                                            std::abs(n1[0].dot(n3[0])) <
                                                    angle_eps &&
                                            std::abs(n2[0].dot(n3[0])) <
                                                    angle_eps) {
                                            n_solutions = 1;
                                        } else {
                                            n_solutions = 0;
                                        }
                                    }
                                } else {
                                    // Discriminant negative: no solutions for v
                                    if (s < 0.0) {
                                        n_solutions = 0;
                                    } else {
                                        double sgn_l = l < 0 ? -1.0 : 1.0;
                                        double V1 =
                                                -(l + sgn_l * std::sqrt(s)) /
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
                                            n1[n_solutions] =
                                                    (a + b * T1).normalized();
                                            n2[n_solutions] =
                                                    (c + d * U1).normalized();
                                            n3[n_solutions] =
                                                    (e + f * V1).normalized();

                                            if (std::abs(n1[n_solutions].dot(
                                                        n2[n_solutions])) <
                                                        angle_eps &&
                                                std::abs(n1[n_solutions].dot(
                                                        n3[n_solutions])) <
                                                        angle_eps &&
                                                std::abs(n2[n_solutions].dot(
                                                        n3[n_solutions])) <
                                                        angle_eps)
                                                ++n_solutions;
                                        }
                                        if (V2 >= -eps && T2 >= -eps &&
                                            U2 >= -eps && V2 <= 1.0 + eps &&
                                            T2 <= 1.0 + eps &&
                                            U2 <= 1.0 + eps) {
                                            n1[n_solutions] =
                                                    (a + b * T2).normalized();
                                            n2[n_solutions] =
                                                    (c + d * U2).normalized();
                                            n3[n_solutions] =
                                                    (e + f * V2).normalized();
                                            if (std::abs(n1[n_solutions].dot(
                                                        n2[n_solutions])) <
                                                        angle_eps &&
                                                std::abs(n1[n_solutions].dot(
                                                        n3[n_solutions])) <
                                                        angle_eps &&
                                                std::abs(n2[n_solutions].dot(
                                                        n3[n_solutions])) <
                                                        angle_eps)
                                                ++n_solutions;
                                        }
                                        if (s < 1e-4 && n_solutions == 2) {
                                            n_solutions = 1;
                                        }
                                    }
                                }
                            }

                            for (int s = 0; s < n_solutions; ++s) {
                                const auto& hull_v_i =
                                        hull_v[edges[edge_i].first];
                                const auto& hull_v_j =
                                        hull_v[edges[edge_j].first];
                                const auto& hull_v_k =
                                        hull_v[edges[edge_k].first];
                                const auto& n1_ = n1[s];
                                const auto& n2_ = n2[s];
                                const auto& n3_ = n3[s];

                                // Compute the most extreme points in each
                                // direction.
                                double max_n1 = n1_.dot(hull_v_i);
                                double max_n2 = n2_.dot(hull_v_j);
                                double max_n3 = n3_.dot(hull_v_k);
                                double min_n1 =
                                        std::numeric_limits<double>::infinity();
                                double min_n2 =
                                        std::numeric_limits<double>::infinity();
                                double min_n3 =
                                        std::numeric_limits<double>::infinity();

                                const auto& antipodal_i =
                                        antipodal_points_for_edge[edge_i];
                                const auto& antipodal_j =
                                        antipodal_points_for_edge[edge_j];
                                const auto& antipodal_k =
                                        antipodal_points_for_edge[edge_k];

                                // Determine the minimum projections along each
                                // axis over respective antipodal sets.
                                for (int v_idx : antipodal_i) {
                                    min_n1 = std::min(min_n1,
                                                      n1_.dot(hull_v[v_idx]));
                                }
                                for (int v_idx : antipodal_j) {
                                    min_n2 = std::min(min_n2,
                                                      n2_.dot(hull_v[v_idx]));
                                }
                                for (int v_idx : antipodal_k) {
                                    min_n3 = std::min(min_n3,
                                                      n3_.dot(hull_v[v_idx]));
                                }

                                // Compute volume based on extents in the three
                                // principal directions.
                                double extent0 = max_n1 - min_n1;
                                double extent1 = max_n2 - min_n2;
                                double extent2 = max_n3 - min_n3;
                                double volume = extent0 * extent1 * extent2;

                                // Update the minimum oriented bounding box if a
                                // smaller volume is found.
                                if (volume < min_volume) {
                                    // Update rotation matrix columns.
                                    min_obb.R_.col(0) = n1_;
                                    min_obb.R_.col(1) = n2_;
                                    min_obb.R_.col(2) = n3_;

                                    // Update extents.
                                    min_obb.extent_(0) = extent0;
                                    min_obb.extent_(1) = extent1;
                                    min_obb.extent_(2) = extent2;

                                    // Compute the center of the OBB using
                                    // midpoints along each axis.
                                    min_obb.center_ =
                                            (min_n1 + 0.5 * extent0) * n1_ +
                                            (min_n2 + 0.5 * extent1) * n2_ +
                                            (min_n3 + 0.5 * extent2) * n3_;

                                    min_volume = volume;
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
        std::vector<int> antipodal_edges;
        antipodal_edges.reserve(128);
        std::vector<Eigen::Vector3d> antipodal_edge_normals;
        antipodal_edge_normals.reserve(128);

        // Iterate over each edge_i in spatial_edge_order.
        for (int edge_i : spatial_edge_order) {
            // Cache face indices and normals for edge_i.
            auto [face_i_a, face_i_b] = faces_for_edge[edge_i];
            const Eigen::Vector3d& f1a = face_normals[face_i_a];
            const Eigen::Vector3d& f1b = face_normals[face_i_b];

            antipodal_edges.clear();
            antipodal_edge_normals.clear();

            // Iterate over vertices antipodal to edge_i.
            const auto& antipodals_for_i = antipodal_points_for_edge[edge_i];
            for (int antipodal_vertex : antipodals_for_i) {
                const auto& neighbors = adjacency_data[antipodal_vertex];
                for (int v_adj : neighbors) {
                    if (v_adj < antipodal_vertex) continue;

                    int edgeIndex = antipodal_vertex * num_vertices + v_adj;
                    int edge = vertex_pairs_to_edges[edgeIndex];

                    if (edge_i > edge) continue;
                    if (isInternalEdge(edge)) continue;

                    auto [faceJ_a, faceJ_b] = faces_for_edge[edge];
                    const Eigen::Vector3d& f2a = face_normals[faceJ_a];
                    const Eigen::Vector3d& f2b = face_normals[faceJ_b];

                    Eigen::Vector3d n;

                    bool areCompatibleOpposingEdges = false;
                    constexpr double tooCloseToFaceEpsilon = 1e-4;

                    Eigen::Matrix3d A;
                    A.col(0) = f2b;
                    A.col(1) = f1a - f1b;
                    A.col(2) = f2a - f2b;
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
                        antipodal_edges.push_back(edge);
                        antipodal_edge_normals.push_back(n.normalized());
                    }
                }
            }

            auto moveSign = [](double& dst, double& src) {
                if (src < 0.0) {
                    dst = -dst;
                    src = -src;
                }
            };

            const auto& compatible_edges_i = compatible_edges[edge_i];
            for (int edge_j : compatible_edges_i) {
                for (size_t k = 0; k < antipodal_edges.size(); ++k) {
                    int edgeK = antipodal_edges[k];

                    const Eigen::Vector3d& n1 = antipodal_edge_normals[k];
                    double min_n1 = n1.dot(hull_v[edges[edgeK].first]);
                    double max_n1 = n1.dot(hull_v[edges[edge_i].first]);

                    // Test all mutual compatible edges.
                    auto [faceK_a, faceK_b] = faces_for_edge[edge_j];
                    const Eigen::Vector3d& f3a = face_normals[faceK_a];
                    const Eigen::Vector3d& f3b = face_normals[faceK_b];

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

                        double max_n2, min_n2;
                        clearGraphSearch();
                        int hint = extremeVertexConvex(
                                extremeVertexConvex, n2, flood_fill_visited,
                                flood_fill_visit_color, max_n2,
                                (k == 0) ? v_hint1 : v_hint1_b);
                        if (k == 0) {
                            v_hint1 = v_hint1_b = hint;
                        } else {
                            v_hint1_b = hint;
                        }

                        clearGraphSearch();
                        hint = extremeVertexConvex(
                                extremeVertexConvex, -n2, flood_fill_visited,
                                flood_fill_visit_color, min_n2,
                                (k == 0) ? v_hint2 : v_hint2_b);
                        if (k == 0) {
                            v_hint2 = v_hint2_b = hint;
                        } else {
                            v_hint2_b = hint;
                        }

                        min_n2 = -min_n2;

                        double max_n3 = n3.dot(hull_v[edges[edge_j].first]);
                        double min_n3 = std::numeric_limits<double>::infinity();

                        // If there are very few antipodal vertices, do a
                        // very tight loop and just iterate over each.
                        const auto& antipodals_edge =
                                antipodal_points_for_edge[edge_j];
                        if (antipodals_edge.size() < 20) {
                            for (int v_idx : antipodals_edge) {
                                min_n3 =
                                        std::min(min_n3, n3.dot(hull_v[v_idx]));
                            }
                        } else {
                            // Otherwise perform a spatial locality
                            // exploiting graph search.
                            clearGraphSearch();
                            hint = extremeVertexConvex(
                                    extremeVertexConvex, -n3,
                                    flood_fill_visited, flood_fill_visit_color,
                                    min_n3, (k == 0) ? v_hint3 : v_hint3_b);

                            if (k == 0) {
                                v_hint3 = v_hint3_b = hint;
                            } else {
                                v_hint3_b = hint;
                            }

                            min_n3 = -min_n3;
                        }

                        double volume = (max_n1 - min_n1) * (max_n2 - min_n2) *
                                        (max_n3 - min_n3);
                        if (volume < min_volume) {
                            min_obb.R_.col(0) = n1;
                            min_obb.R_.col(1) = n2;
                            min_obb.R_.col(2) = n3;
                            min_obb.extent_(0) = (max_n1 - min_n1);
                            min_obb.extent_(1) = (max_n2 - min_n2);
                            min_obb.extent_(2) = (max_n3 - min_n3);
                            min_obb.center_ = 0.5 * ((min_n1 + max_n1) * n1 +
                                                     (min_n2 + max_n2) * n2 +
                                                     (min_n3 + max_n3) * n3);
                            min_volume = volume;
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
        std::vector<int> antipodal_edges;
        antipodal_edges.reserve(128);
        std::vector<Eigen::Vector3d> antipodal_edge_normals;
        antipodal_edge_normals.reserve(128);

        for (int face_idx : spatial_face_order) {
            const Eigen::Vector3d& n1 = face_normals[face_idx];

            // Find two edges on the face. Since we have flexibility to
            // choose from multiple edges of the same face, choose two that
            // are possibly most opposing to each other, in the hope that
            // their sets of sidepodal edges are most mutually exclusive as
            // possible, speeding up the search below.
            int e1 = -1;
            const auto& tri = hull_t[face_idx];
            int v0 = tri(2);
            for (int j = 0; j < 3; ++j) {
                int v1 = tri(j);
                int e = vertex_pairs_to_edges[v0 * num_vertices + v1];
                if (!isInternalEdge(e)) {
                    e1 = e;
                    break;
                }
                v0 = v1;
            }

            if (e1 == -1) continue;

            // Compute min_n1 either by scanning antipodal points or using
            // ExtremeVertexConvex.
            double max_n1 = n1.dot(hull_v[edges[e1].first]);
            double min_n1 = std::numeric_limits<double>::infinity();
            const auto& antipodals = antipodal_points_for_edge[e1];
            if (antipodals.size() < 20) {
                min_n1 = std::numeric_limits<double>::infinity();
                for (int v_idx : antipodals) {
                    min_n1 = std::min(min_n1, n1.dot(hull_v[v_idx]));
                }
            } else {
                clearGraphSearch();
                v_hint4 = extremeVertexConvex(
                        extremeVertexConvex, -n1, flood_fill_visited,
                        flood_fill_visit_color, min_n1, v_hint4);
                min_n1 = -min_n1;
            }

            // Check edges compatible with e1.
            const auto& compatible_edges_i = compatible_edges[e1];
            for (int edge_k : compatible_edges_i) {
                auto [face_k_a, face_k_b] = faces_for_edge[edge_k];
                const Eigen::Vector3d& f3a = face_normals[face_k_a];
                const Eigen::Vector3d& f3b = face_normals[face_k_b];

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

                    double max_n2, min_n2;
                    clearGraphSearch();
                    v_hint1 = extremeVertexConvex(
                            extremeVertexConvex, n2, flood_fill_visited,
                            flood_fill_visit_color, max_n2, v_hint1);
                    clearGraphSearch();
                    v_hint2 = extremeVertexConvex(
                            extremeVertexConvex, -n2, flood_fill_visited,
                            flood_fill_visit_color, min_n2, v_hint2);
                    min_n2 = -min_n2;

                    double max_n3 = n3.dot(hull_v[edges[edge_k].first]);
                    double min_n3 = std::numeric_limits<double>::infinity();

                    // If there are very few antipodal vertices, do a very
                    // tight loop and just iterate over each.
                    const auto& antipodals_edge =
                            antipodal_points_for_edge[edge_k];
                    if (antipodals_edge.size() < 20) {
                        for (int v_idx : antipodals_edge) {
                            min_n3 = std::min(min_n3, n3.dot(hull_v[v_idx]));
                        }
                    } else {
                        clearGraphSearch();
                        v_hint3 = extremeVertexConvex(
                                extremeVertexConvex, -n3, flood_fill_visited,
                                flood_fill_visit_color, min_n3, v_hint3);
                        min_n3 = -min_n3;
                    }

                    double volume = (max_n1 - min_n1) * (max_n2 - min_n2) *
                                    (max_n3 - min_n3);
                    if (volume < min_volume) {
                        min_obb.R_.col(0) = n1;
                        min_obb.R_.col(1) = n2;
                        min_obb.R_.col(2) = n3;
                        min_obb.extent_(0) = (max_n1 - min_n1);
                        min_obb.extent_(1) = (max_n2 - min_n2);
                        min_obb.extent_(2) = (max_n3 - min_n3);
                        min_obb.center_ = 0.5 * ((min_n1 + max_n1) * n1 +
                                                 (min_n2 + max_n2) * n2 +
                                                 (min_n3 + max_n3) * n3);
                        assert(volume > 0.0);
                        min_volume = volume;
                    }
                }
            }
        }
    }

    // Final check to ensure right-handed coordinate frame
    if (min_obb.R_.col(0).cross(min_obb.R_.col(1)).dot(min_obb.R_.col(2)) <
        0.0) {
        min_obb.R_.col(2) = -min_obb.R_.col(2);
    }
    mapOBBToClosestIdentity(min_obb);
    return static_cast<OrientedBoundingBox>(min_obb).To(points_.GetDevice());
}

OrientedBoundingBox ComputeMinimumOBBApprox(const core::Tensor& points,
                                            bool robust) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    if (points.GetShape(0) == 0) {
        utility::LogError("Input point set is empty.");
        return OrientedBoundingBox();
    }
    if (points.GetShape(0) < 4) {
        utility::LogError("Input point set has less than 4 points.");
        return OrientedBoundingBox();
    }

    // copy to cpu here
    PointCloud pcd(points.To(core::Device()));
    auto hull_mesh = pcd.ComputeConvexHull(robust);
    if (hull_mesh.GetVertexPositions().NumElements() == 0) {
        utility::LogError("Failed to compute convex hull.");
        return OrientedBoundingBox();
    }

    // Get convex hull vertices and triangles
    const std::vector<Eigen::Vector3d>& hull_v =
            core::eigen_converter::TensorToEigenVector3dVector(
                    hull_mesh.GetVertexPositions());
    const std::vector<Eigen::Vector3i>& hull_t =
            core::eigen_converter::TensorToEigenVector3iVector(
                    hull_mesh.GetTriangleIndices());

    OrientedBoundingBox min_box(
            AxisAlignedBoundingBox::CreateFromPoints(
                    hull_mesh.GetVertexPositions().To(core::Float32))
                    .GetOrientedBoundingBox());
    double min_vol = min_box.Volume();

    PointCloud hull_pcd(hull_mesh.GetVertexPositions().Clone());
    for (auto& tri : hull_t) {
        hull_pcd.GetPointPositions().CopyFrom(hull_mesh.GetVertexPositions());
        Eigen::Vector3d a = hull_v[tri(0)];
        Eigen::Vector3d b = hull_v[tri(1)];
        Eigen::Vector3d c = hull_v[tri(2)];
        Eigen::Vector3d u = b - a;
        Eigen::Vector3d v = c - a;
        Eigen::Vector3d w = u.cross(v);
        v = w.cross(u);
        u = u / u.norm();
        v = v / v.norm();
        w = w / w.norm();
        Eigen::Matrix3d m_rot;
        m_rot << u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2];
        auto rot_inv =
                core::eigen_converter::EigenMatrixToTensor(m_rot.inverse());
        auto center =
                core::eigen_converter::EigenMatrixToTensor(a).Reshape({3});
        hull_pcd.Rotate(rot_inv, center);

        const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();
        double volume = aabox.Volume();
        if (volume < min_vol) {
            min_vol = volume;
            min_box = aabox.GetOrientedBoundingBox();
            auto rot = core::eigen_converter::EigenMatrixToTensor(m_rot);
            min_box.Rotate(rot, center);
        }
    }
    auto device = points.GetDevice();
    auto dtype = points.GetDtype();
    return OrientedBoundingBox(min_box.GetCenter().To(device, dtype),
                               min_box.GetRotation().To(device, dtype),
                               min_box.GetExtent().To(device, dtype));
}

}  // namespace minimum_obb
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d