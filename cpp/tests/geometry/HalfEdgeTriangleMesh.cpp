// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/HalfEdgeTriangleMesh.h"

#include <iostream>
#include <string>

#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/Helper.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

// [0: (-1, 2)]__________[1: (1, 2)]
//             |        /|
//              |  (0) /  |
//               |    / (1)|
//                |  /      |
//      [2: (0, 0)]|/________|[3: (2, 0)]
geometry::TriangleMesh get_mesh_two_triangles() {
    std::vector<Eigen::Vector3d> vertices{
            Eigen::Vector3d(-1, 2, 0), Eigen::Vector3d(1, 2, 0),
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(2, 0, 0)};
    std::vector<Eigen::Vector3i> triangles{Eigen::Vector3i(0, 2, 1),
                                           Eigen::Vector3i(1, 2, 3)};
    geometry::TriangleMesh mesh;
    mesh.vertices_ = vertices;
    mesh.triangles_ = triangles;
    return mesh;
}

// [0: (-1, 2)]__________[1: (1, 2)]     [4: (9, 2)]__________[5: (11, 2)]
//             |        /|                           |        /|
//              |  (0) /  |                           |  (0) /  |
//               |    / (1)|                           |    / (1)|
//                |  /      |                           |  /      |
//      [2: (0, 0)]|/________|[3: (2, 0)]    [6: (10, 0)]|/________|[7: (12, 0)]
geometry::TriangleMesh get_mesh_four_triangles_disconnect() {
    std::vector<Eigen::Vector3d> vertices{
            Eigen::Vector3d(-1, 2, 0), Eigen::Vector3d(1, 2, 0),
            Eigen::Vector3d(0, 0, 0),  Eigen::Vector3d(2, 0, 0),
            Eigen::Vector3d(9, 2, 0),  Eigen::Vector3d(11, 2, 0),
            Eigen::Vector3d(10, 0, 0), Eigen::Vector3d(12, 0, 0)};
    std::vector<Eigen::Vector3i> triangles{
            Eigen::Vector3i(0, 2, 1), Eigen::Vector3i(1, 2, 3),
            Eigen::Vector3i(4, 6, 5), Eigen::Vector3i(5, 6, 7)};
    geometry::TriangleMesh mesh;
    mesh.vertices_ = vertices;
    mesh.triangles_ = triangles;
    return mesh;
}

// [0: (-1, 2)]__________[1: (1, 2)]
//             |        /|
//              |  (0) /  |
//               |    / (1)|
//                |  /      |
//      [2: (0, 0)]|/________|[3: (2, 0)]
//
// Non-manifold: triangle (1) is flipped
geometry::TriangleMesh get_mesh_two_triangles_flipped() {
    std::vector<Eigen::Vector3d> vertices{
            Eigen::Vector3d(-1, 2, 0), Eigen::Vector3d(1, 2, 0),
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(2, 0, 0)};
    std::vector<Eigen::Vector3i> triangles{Eigen::Vector3i(0, 2, 1),
                                           Eigen::Vector3i(1, 3, 2)};
    geometry::TriangleMesh mesh;
    mesh.vertices_ = vertices;
    mesh.triangles_ = triangles;
    return mesh;
}

//  [0: (-1, 2)]__________[1: (1, 2)]
//              |        /
//               |  (0) /
//                |    /
//                 |  /
//                  |/ [2: (0, 0)]
//                  /|
//                 /  |
//                /    |
//               /  (1) |
//              /________|
// [3: (-1, -2)]          [4: (1, -2)]
//
// Non-manifold
geometry::TriangleMesh get_mesh_two_triangles_invalid_vertex() {
    std::vector<Eigen::Vector3d> vertices{
            Eigen::Vector3d(-1, 2, 0), Eigen::Vector3d(1, 2, 0),
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(-1, -2, 0),
            Eigen::Vector3d(1, -2, 0)};
    std::vector<Eigen::Vector3i> triangles{Eigen::Vector3i(0, 2, 1),
                                           Eigen::Vector3i(2, 3, 4)};
    geometry::TriangleMesh mesh;
    mesh.vertices_ = vertices;
    mesh.triangles_ = triangles;
    return mesh;
}

//          [0: (-1, 2)]__________[1: (1, 2)]
//                     /|        /|
//                    /  |  (1) /  |
//                   / (0)|    / (2)|
//                  /      |  /      |
//     [2: (-2, 0)]/____[3: (O, 0)]___|[4: (2, 0)]
//                 |        /|        /
//                  |  (3) /  |  (5) /
//                   |    /    |    /
//                    |  /  (4) |  /
//                     |/________|/
//         [5: (-1, -2)]          [6: (1, -2)]
geometry::TriangleMesh get_mesh_hexagon() {
    std::vector<Eigen::Vector3d> vertices{
            Eigen::Vector3d(-1, 2, 0), Eigen::Vector3d(1, 2, 0),
            Eigen::Vector3d(-2, 0, 0), Eigen::Vector3d(0, 0, 0),
            Eigen::Vector3d(2, 0, 0),  Eigen::Vector3d(-1, -2, 0),
            Eigen::Vector3d(1, -2, 0)};
    std::vector<Eigen::Vector3i> triangles{
            Eigen::Vector3i(0, 2, 3), Eigen::Vector3i(0, 3, 1),
            Eigen::Vector3i(1, 3, 4), Eigen::Vector3i(2, 5, 3),
            Eigen::Vector3i(3, 5, 6), Eigen::Vector3i(3, 6, 4)};
    geometry::TriangleMesh mesh;
    mesh.vertices_ = vertices;
    mesh.triangles_ = triangles;
    return mesh;
}

//          [0: (-1, 2)]__________[1: (1, 2)]
//                     /|        /|
//                    /  |  (1) /  |
//                   / (0)|    / (2)|
//                  /      |  /      |
//     [2: (-2, 0)]/____[3: (O, 0)]___|[4: (2, 0)]
//                 |        /|
//                  |  (3) /  |
//                   |    /    |
//                    |  /  (4) |
//                     |/________|
//         [5: (-1, -2)]          [6: (1, -2)]
geometry::TriangleMesh get_mesh_partial_hexagon() {
    std::vector<Eigen::Vector3d> vertices{
            Eigen::Vector3d(-1, 2, 0), Eigen::Vector3d(1, 2, 0),
            Eigen::Vector3d(-2, 0, 0), Eigen::Vector3d(0, 0, 0),
            Eigen::Vector3d(2, 0, 0),  Eigen::Vector3d(-1, -2, 0),
            Eigen::Vector3d(1, -2, 0)};
    std::vector<Eigen::Vector3i> triangles{
            Eigen::Vector3i(0, 2, 3), Eigen::Vector3i(0, 3, 1),
            Eigen::Vector3i(1, 3, 4), Eigen::Vector3i(2, 5, 3),
            Eigen::Vector3i(3, 5, 6)};
    geometry::TriangleMesh mesh;
    mesh.vertices_ = vertices;
    mesh.triangles_ = triangles;
    return mesh;
}

void assert_ordreded_neighbor(
        const std::shared_ptr<geometry::HalfEdgeTriangleMesh>& het_mesh,
        int vertex_index,
        const std::vector<int>& expected_ordered_neighbors,
        bool allow_rotation = false) {
    std::vector<int> actual_ordered_neighbors;
    for (int half_edge_index :
         het_mesh->ordered_half_edge_from_vertex_[vertex_index]) {
        actual_ordered_neighbors.push_back(
                het_mesh->half_edges_[half_edge_index].vertex_indices_[1]);
    }

    if (expected_ordered_neighbors.size() == 0) {
        EXPECT_EQ(actual_ordered_neighbors.size(), 0u);
        return;
    }

    if (allow_rotation) {
        // E.g. Actual 0, 1, 2, 3, 4
        //      Expect 2, 3, 4, 0, 1
        // Then left-rotate actual by 2
        auto find_it = std::find(actual_ordered_neighbors.begin(),
                                 actual_ordered_neighbors.end(),
                                 expected_ordered_neighbors[0]);
        if (find_it == actual_ordered_neighbors.end()) {
            FAIL();
        } else {
            size_t offset = find_it - actual_ordered_neighbors.begin();
            std::rotate(actual_ordered_neighbors.begin(),
                        actual_ordered_neighbors.begin() + offset,
                        actual_ordered_neighbors.end());
        }
    }
    EXPECT_EQ(expected_ordered_neighbors, actual_ordered_neighbors);
}

void assert_vector_eq(const std::vector<int>& actual,
                      const std::vector<int>& expect,
                      bool allow_rotation = false) {
    std::vector<int> actual_copied = actual;
    if (allow_rotation) {
        // E.g. Actual 0, 1, 2, 3, 4
        //      Expect 2, 3, 4, 0, 1
        // Then left-rotate actual by 2
        auto find_it = std::find(actual_copied.begin(), actual_copied.end(),
                                 expect[0]);
        if (find_it == actual_copied.end()) {
            FAIL();
        } else {
            size_t offset = find_it - actual_copied.begin();
            std::rotate(actual_copied.begin(), actual_copied.begin() + offset,
                        actual_copied.end());
        }
    }
    EXPECT_EQ(actual_copied, expect);
}

void assert_ordreded_edges(
        const std::shared_ptr<geometry::HalfEdgeTriangleMesh>& het_mesh,
        const std::vector<int>& half_edge_indices,
        const std::vector<std::vector<int>>& expected_half_edge_vertices) {
    if (half_edge_indices.size() != expected_half_edge_vertices.size()) {
        FAIL();
    }
    std::vector<std::vector<int>> actual_half_edge_vertices;
    for (int half_edge_index : half_edge_indices) {
        const auto& he = het_mesh->half_edges_[half_edge_index];
        actual_half_edge_vertices.push_back(std::vector<int>(
                {he.vertex_indices_[0], he.vertex_indices_[1]}));
    }
    for (size_t i = 0; i < actual_half_edge_vertices.size(); ++i) {
        EXPECT_EQ(actual_half_edge_vertices[i], expected_half_edge_vertices[i]);
    }
}

void assert_same_vertices_and_triangles(
        const geometry::TriangleMesh& mesh,
        const geometry::HalfEdgeTriangleMesh& het_mesh) {
    tests::ExpectEQ(mesh.vertices_, het_mesh.vertices_);
    tests::ExpectEQ(mesh.triangles_, het_mesh.triangles_);
}

TEST(HalfEdgeTriangleMesh, Constructor_TwoTriangles) {
    geometry::TriangleMesh mesh = get_mesh_two_triangles();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    assert_same_vertices_and_triangles(mesh, *het_mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, Constructor_Hexagon) {
    geometry::TriangleMesh mesh = get_mesh_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    assert_same_vertices_and_triangles(mesh, *het_mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, Constructor_PartialHexagon) {
    geometry::TriangleMesh mesh = get_mesh_partial_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    assert_same_vertices_and_triangles(mesh, *het_mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, Constructor_Sphere) {
    auto mesh = geometry::TriangleMesh::CreateSphere(0.05);
    mesh->ComputeVertexNormals();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(*mesh);
    assert_same_vertices_and_triangles(*mesh, *het_mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, OrderedHalfEdgesFromVertex_TwoTriangles) {
    auto mesh = get_mesh_two_triangles();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    assert_ordreded_neighbor(het_mesh, 0, {2});
    assert_ordreded_neighbor(het_mesh, 1, {0, 2});
    assert_ordreded_neighbor(het_mesh, 2, {3, 1});
    assert_ordreded_neighbor(het_mesh, 3, {1});
}

TEST(HalfEdgeTriangleMesh, OrderedHalfEdgesFromVertex_Hexagon) {
    auto mesh = get_mesh_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    assert_ordreded_neighbor(het_mesh, 0, {2, 3});
    assert_ordreded_neighbor(het_mesh, 1, {0, 3});
    assert_ordreded_neighbor(het_mesh, 2, {5, 3});
    assert_ordreded_neighbor(het_mesh, 3, {0, 2, 5, 6, 4, 1}, true);
    assert_ordreded_neighbor(het_mesh, 3, {2, 5, 6, 4, 1, 0}, true);  // Rotate
    assert_ordreded_neighbor(het_mesh, 4, {1, 3});
    assert_ordreded_neighbor(het_mesh, 5, {6, 3});
    assert_ordreded_neighbor(het_mesh, 6, {4, 3});
}

TEST(HalfEdgeTriangleMesh, OrderedHalfEdgesFromVertex_PartialHexagon) {
    auto mesh = get_mesh_partial_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    assert_ordreded_neighbor(het_mesh, 0, {2, 3});
    assert_ordreded_neighbor(het_mesh, 1, {0, 3});
    assert_ordreded_neighbor(het_mesh, 2, {5, 3});
    assert_ordreded_neighbor(het_mesh, 3, {4, 1, 0, 2, 5});  // Rotate not ok
    assert_ordreded_neighbor(het_mesh, 4, {1});
    assert_ordreded_neighbor(het_mesh, 5, {6, 3});
    assert_ordreded_neighbor(het_mesh, 6, {3});
}

TEST(HalfEdgeTriangleMesh, BoundaryHalfEdgesFromVertex_TwoTriangles) {
    auto mesh = get_mesh_two_triangles();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(0),
                          {{0, 2}, {2, 3}, {3, 1}, {1, 0}});
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(1),
                          {{1, 0}, {0, 2}, {2, 3}, {3, 1}});
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(2),
                          {{2, 3}, {3, 1}, {1, 0}, {0, 2}});
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(3),
                          {{3, 1}, {1, 0}, {0, 2}, {2, 3}});
}

TEST(HalfEdgeTriangleMesh, BoundaryHalfEdgesFromVertex_Hexagon) {
    auto mesh = get_mesh_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(0),
                          {{0, 2}, {2, 5}, {5, 6}, {6, 4}, {4, 1}, {1, 0}});
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(1),
                          {{1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 4}, {4, 1}});
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(2),
                          {{2, 5}, {5, 6}, {6, 4}, {4, 1}, {1, 0}, {0, 2}});
    EXPECT_THROW(het_mesh->BoundaryHalfEdgesFromVertex(3),
                 std::runtime_error);  // Vertex 3 is not a boundary, thus empty
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(4),
                          {{4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 4}});
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(5),
                          {{5, 6}, {6, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}});
    assert_ordreded_edges(het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(6),
                          {{6, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}});
}

TEST(HalfEdgeTriangleMesh, BoundaryHalfEdgesFromVertex_PartialHexagon) {
    auto mesh = get_mesh_partial_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    assert_ordreded_edges(
            het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(0),
            {{0, 2}, {2, 5}, {5, 6}, {6, 3}, {3, 4}, {4, 1}, {1, 0}});
    assert_ordreded_edges(
            het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(1),
            {{1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 3}, {3, 4}, {4, 1}});
    assert_ordreded_edges(
            het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(2),
            {{2, 5}, {5, 6}, {6, 3}, {3, 4}, {4, 1}, {1, 0}, {0, 2}});
    assert_ordreded_edges(
            het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(3),
            {{3, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 3}});
    assert_ordreded_edges(
            het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(4),
            {{4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 3}, {3, 4}});
    assert_ordreded_edges(
            het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(5),
            {{5, 6}, {6, 3}, {3, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}});
    assert_ordreded_edges(
            het_mesh, het_mesh->BoundaryHalfEdgesFromVertex(6),
            {{6, 3}, {3, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}});
}

TEST(HalfEdgeTriangleMesh, BoundaryVerticesFromVertex_TwoTriangles) {
    auto mesh = get_mesh_two_triangles();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(0), {0, 2, 3, 1});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(1), {1, 0, 2, 3});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(2), {2, 3, 1, 0});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(3), {3, 1, 0, 2});
}

TEST(HalfEdgeTriangleMesh, BoundaryVerticesFromVertex_Hexagon) {
    auto mesh = get_mesh_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(0), {0, 2, 5, 6, 4, 1});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(1), {1, 0, 2, 5, 6, 4});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(2), {2, 5, 6, 4, 1, 0});
    EXPECT_THROW(het_mesh->BoundaryVerticesFromVertex(3),
                 std::runtime_error);  // Vertex 3 is not a boundary, thus empty
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(4), {4, 1, 0, 2, 5, 6});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(5), {5, 6, 4, 1, 0, 2});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(6), {6, 4, 1, 0, 2, 5});
}

TEST(HalfEdgeTriangleMesh, BoundaryVerticesFromVertex_PartialHexagon) {
    auto mesh = get_mesh_partial_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(0), {0, 2, 5, 6, 3, 4, 1});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(1), {1, 0, 2, 5, 6, 3, 4});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(2), {2, 5, 6, 3, 4, 1, 0});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(3), {3, 4, 1, 0, 2, 5, 6});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(4), {4, 1, 0, 2, 5, 6, 3});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(5), {5, 6, 3, 4, 1, 0, 2});
    ExpectEQ(het_mesh->BoundaryVerticesFromVertex(6), {6, 3, 4, 1, 0, 2, 5});
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_TwoTriangles) {
    auto mesh = get_mesh_two_triangles();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    EXPECT_EQ(het_mesh->GetBoundaries().size(), 1u);
    assert_vector_eq(het_mesh->GetBoundaries()[0], {0, 2, 3, 1}, true);
    assert_vector_eq(het_mesh->GetBoundaries()[0], {2, 3, 1, 0},
                     true);  // rotate
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_Hexagon) {
    auto mesh = get_mesh_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    EXPECT_EQ(het_mesh->GetBoundaries().size(), 1u);
    assert_vector_eq(het_mesh->GetBoundaries()[0], {0, 2, 5, 6, 4, 1}, true);
    assert_vector_eq(het_mesh->GetBoundaries()[0], {2, 5, 6, 4, 1, 0}, true);
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_PartialHexagon) {
    auto mesh = get_mesh_partial_hexagon();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    EXPECT_EQ(het_mesh->GetBoundaries().size(), 1u);
    assert_vector_eq(het_mesh->GetBoundaries()[0], {0, 2, 5, 6, 3, 4, 1}, true);
    assert_vector_eq(het_mesh->GetBoundaries()[0], {2, 5, 6, 3, 4, 1, 0}, true);
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_FourTrianglesDisconnect) {
    auto mesh = get_mesh_four_triangles_disconnect();
    auto het_mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(mesh);
    EXPECT_FALSE(het_mesh->IsEmpty());
    assert_same_vertices_and_triangles(mesh, *het_mesh);

    EXPECT_EQ(het_mesh->GetBoundaries().size(), 2u);
    assert_vector_eq(het_mesh->GetBoundaries()[0], {0, 2, 3, 1}, true);
    assert_vector_eq(het_mesh->GetBoundaries()[1], {4, 6, 7, 5}, true);
}

}  // namespace tests
}  // namespace open3d
