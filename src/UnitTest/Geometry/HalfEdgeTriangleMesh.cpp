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

#include <iostream>
#include <string>

#include "Open3D/Geometry/HalfEdgeTriangleMesh.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Utility/Helper.h"
#include "TestUtility/UnitTest.h"

using namespace open3d;
using namespace unit_test;

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
        const std::shared_ptr<geometry::HalfEdgeTriangleMesh>& mesh,
        int vertex_index,
        const std::vector<int>& expected_ordered_neighbors,
        bool allow_rotation = false) {
    std::vector<int> actual_ordered_neighbors;
    for (int half_edge_index :
         mesh->ordered_half_edge_from_vertex_[vertex_index]) {
        actual_ordered_neighbors.push_back(
                mesh->half_edges_[half_edge_index].vertex_indices_[1]);
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
        const std::shared_ptr<geometry::HalfEdgeTriangleMesh>& mesh,
        const std::vector<int>& half_edge_indices,
        const std::vector<std::vector<int>>& expected_half_edge_vertices) {
    if (half_edge_indices.size() != expected_half_edge_vertices.size()) {
        FAIL();
    }
    std::vector<std::vector<int>> actual_half_edge_vertices;
    for (int half_edge_index : half_edge_indices) {
        const auto& he = mesh->half_edges_[half_edge_index];
        actual_half_edge_vertices.push_back(std::vector<int>(
                {he.vertex_indices_[0], he.vertex_indices_[1]}));
    }
    for (size_t i = 0; i < actual_half_edge_vertices.size(); ++i) {
        EXPECT_EQ(actual_half_edge_vertices[i], expected_half_edge_vertices[i]);
    }
}

TEST(HalfEdgeTriangleMesh, Constructor_TwoTriangles) {
    geometry::TriangleMesh mesh = get_mesh_two_triangles();
    auto he_mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(mesh);
    EXPECT_FALSE(he_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, Constructor_TwoTrianglesFlipped) {
    geometry::TriangleMesh mesh = get_mesh_two_triangles_flipped();
    ASSERT_THROW(geometry::HalfEdgeTriangleMesh::CreateFromMesh(mesh),
                 std::runtime_error);  // Non-manifold
}

TEST(HalfEdgeTriangleMesh, Constructor_TwoTrianglesInvalidVertex) {
    geometry::TriangleMesh mesh = get_mesh_two_triangles_invalid_vertex();
    ASSERT_THROW(geometry::HalfEdgeTriangleMesh::CreateFromMesh(mesh),
                 std::runtime_error);  // Non-manifold
}

TEST(HalfEdgeTriangleMesh, Constructor_Hexagon) {
    geometry::TriangleMesh mesh = get_mesh_hexagon();
    auto he_mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(mesh);
    EXPECT_FALSE(he_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, Constructor_PartialHexagon) {
    geometry::TriangleMesh mesh = get_mesh_partial_hexagon();
    auto he_mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(mesh);
    EXPECT_FALSE(he_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, Constructor_Sphere) {
    geometry::TriangleMesh mesh;
    io::ReadTriangleMesh(std::string(TEST_DATA_DIR) + "/sphere.ply", mesh);
    auto he_mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(mesh);
    EXPECT_FALSE(he_mesh->IsEmpty());
}

TEST(HalfEdgeTriangleMesh, OrderedHalfEdgesFromVertex_TwoTriangles) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_two_triangles());
    EXPECT_FALSE(mesh->IsEmpty());
    assert_ordreded_neighbor(mesh, 0, {2});
    assert_ordreded_neighbor(mesh, 1, {0, 2});
    assert_ordreded_neighbor(mesh, 2, {3, 1});
    assert_ordreded_neighbor(mesh, 3, {1});
}

TEST(HalfEdgeTriangleMesh, OrderedHalfEdgesFromVertex_Hexagon) {
    auto mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromMesh(get_mesh_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());
    assert_ordreded_neighbor(mesh, 0, {2, 3});
    assert_ordreded_neighbor(mesh, 1, {0, 3});
    assert_ordreded_neighbor(mesh, 2, {5, 3});
    assert_ordreded_neighbor(mesh, 3, {0, 2, 5, 6, 4, 1}, true);
    assert_ordreded_neighbor(mesh, 3, {2, 5, 6, 4, 1, 0}, true);  // Rotate
    assert_ordreded_neighbor(mesh, 4, {1, 3});
    assert_ordreded_neighbor(mesh, 5, {6, 3});
    assert_ordreded_neighbor(mesh, 6, {4, 3});
}

TEST(HalfEdgeTriangleMesh, OrderedHalfEdgesFromVertex_PartialHexagon) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_partial_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());
    assert_ordreded_neighbor(mesh, 0, {2, 3});
    assert_ordreded_neighbor(mesh, 1, {0, 3});
    assert_ordreded_neighbor(mesh, 2, {5, 3});
    assert_ordreded_neighbor(mesh, 3, {4, 1, 0, 2, 5});  // Rotate not ok
    assert_ordreded_neighbor(mesh, 4, {1});
    assert_ordreded_neighbor(mesh, 5, {6, 3});
    assert_ordreded_neighbor(mesh, 6, {3});
}

TEST(HalfEdgeTriangleMesh, BoundaryHalfEdgesFromVertex_TwoTriangles) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_two_triangles());
    EXPECT_FALSE(mesh->IsEmpty());

    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(0),
                          {{0, 2}, {2, 3}, {3, 1}, {1, 0}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(1),
                          {{1, 0}, {0, 2}, {2, 3}, {3, 1}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(2),
                          {{2, 3}, {3, 1}, {1, 0}, {0, 2}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(3),
                          {{3, 1}, {1, 0}, {0, 2}, {2, 3}});
}

TEST(HalfEdgeTriangleMesh, BoundaryHalfEdgesFromVertex_Hexagon) {
    auto mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromMesh(get_mesh_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());

    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(0),
                          {{0, 2}, {2, 5}, {5, 6}, {6, 4}, {4, 1}, {1, 0}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(1),
                          {{1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 4}, {4, 1}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(2),
                          {{2, 5}, {5, 6}, {6, 4}, {4, 1}, {1, 0}, {0, 2}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(3),
                          {});  // Vertex 3 is not a boundary, thus empty
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(4),
                          {{4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 4}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(5),
                          {{5, 6}, {6, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}});
    assert_ordreded_edges(mesh, mesh->BoundaryHalfEdgesFromVertex(6),
                          {{6, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}});
}

TEST(HalfEdgeTriangleMesh, BoundaryHalfEdgesFromVertex_PartialHexagon) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_partial_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());

    assert_ordreded_edges(
            mesh, mesh->BoundaryHalfEdgesFromVertex(0),
            {{0, 2}, {2, 5}, {5, 6}, {6, 3}, {3, 4}, {4, 1}, {1, 0}});
    assert_ordreded_edges(
            mesh, mesh->BoundaryHalfEdgesFromVertex(1),
            {{1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 3}, {3, 4}, {4, 1}});
    assert_ordreded_edges(
            mesh, mesh->BoundaryHalfEdgesFromVertex(2),
            {{2, 5}, {5, 6}, {6, 3}, {3, 4}, {4, 1}, {1, 0}, {0, 2}});
    assert_ordreded_edges(
            mesh, mesh->BoundaryHalfEdgesFromVertex(3),
            {{3, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 3}});
    assert_ordreded_edges(
            mesh, mesh->BoundaryHalfEdgesFromVertex(4),
            {{4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}, {6, 3}, {3, 4}});
    assert_ordreded_edges(
            mesh, mesh->BoundaryHalfEdgesFromVertex(5),
            {{5, 6}, {6, 3}, {3, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}});
    assert_ordreded_edges(
            mesh, mesh->BoundaryHalfEdgesFromVertex(6),
            {{6, 3}, {3, 4}, {4, 1}, {1, 0}, {0, 2}, {2, 5}, {5, 6}});
}

TEST(HalfEdgeTriangleMesh, BoundaryVerticesFromVertex_TwoTriangles) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_two_triangles());
    EXPECT_FALSE(mesh->IsEmpty());
    ExpectEQ(mesh->BoundaryVerticesFromVertex(0), {0, 2, 3, 1});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(1), {1, 0, 2, 3});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(2), {2, 3, 1, 0});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(3), {3, 1, 0, 2});
}

TEST(HalfEdgeTriangleMesh, BoundarVerticesFromVertex_Hexagon) {
    auto mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromMesh(get_mesh_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());
    ExpectEQ(mesh->BoundaryVerticesFromVertex(0), {0, 2, 5, 6, 4, 1});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(1), {1, 0, 2, 5, 6, 4});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(2), {2, 5, 6, 4, 1, 0});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(3),
             {});  // Vertex 3 is not a boundary, thus empty
    ExpectEQ(mesh->BoundaryVerticesFromVertex(4), {4, 1, 0, 2, 5, 6});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(5), {5, 6, 4, 1, 0, 2});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(6), {6, 4, 1, 0, 2, 5});
}

TEST(HalfEdgeTriangleMesh, BoundaryVerticesFromVertex_PartialHexagon) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_partial_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());
    ExpectEQ(mesh->BoundaryVerticesFromVertex(0), {0, 2, 5, 6, 3, 4, 1});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(1), {1, 0, 2, 5, 6, 3, 4});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(2), {2, 5, 6, 3, 4, 1, 0});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(3), {3, 4, 1, 0, 2, 5, 6});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(4), {4, 1, 0, 2, 5, 6, 3});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(5), {5, 6, 3, 4, 1, 0, 2});
    ExpectEQ(mesh->BoundaryVerticesFromVertex(6), {6, 3, 4, 1, 0, 2, 5});
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_TwoTriangles) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_two_triangles());
    EXPECT_FALSE(mesh->IsEmpty());
    EXPECT_EQ(mesh->GetBoundaries().size(), 1u);
    assert_vector_eq(mesh->GetBoundaries()[0], {0, 2, 3, 1}, true);
    assert_vector_eq(mesh->GetBoundaries()[0], {2, 3, 1, 0}, true);  // rotate
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_Hexagon) {
    auto mesh =
            geometry::HalfEdgeTriangleMesh::CreateFromMesh(get_mesh_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());
    EXPECT_EQ(mesh->GetBoundaries().size(), 1u);
    assert_vector_eq(mesh->GetBoundaries()[0], {0, 2, 5, 6, 4, 1}, true);
    assert_vector_eq(mesh->GetBoundaries()[0], {2, 5, 6, 4, 1, 0}, true);
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_PartialHexagon) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_partial_hexagon());
    EXPECT_FALSE(mesh->IsEmpty());
    EXPECT_EQ(mesh->GetBoundaries().size(), 1u);
    assert_vector_eq(mesh->GetBoundaries()[0], {0, 2, 5, 6, 3, 4, 1}, true);
    assert_vector_eq(mesh->GetBoundaries()[0], {2, 5, 6, 3, 4, 1, 0}, true);
}

TEST(HalfEdgeTriangleMesh, GetBoundaries_FourTrianglesDisconnect) {
    auto mesh = geometry::HalfEdgeTriangleMesh::CreateFromMesh(
            get_mesh_four_triangles_disconnect());
    EXPECT_FALSE(mesh->IsEmpty());
    EXPECT_EQ(mesh->GetBoundaries().size(), 2u);
    assert_vector_eq(mesh->GetBoundaries()[0], {0, 2, 3, 1}, true);
    assert_vector_eq(mesh->GetBoundaries()[1], {4, 6, 7, 5}, true);
}
