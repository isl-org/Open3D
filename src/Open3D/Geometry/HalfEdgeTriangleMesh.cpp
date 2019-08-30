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

#include "Open3D/Geometry/HalfEdgeTriangleMesh.h"
#include "Open3D/Geometry/BoundingVolume.h"

#include <numeric>

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {
namespace geometry {

HalfEdgeTriangleMesh::HalfEdge::HalfEdge(const Eigen::Vector2i &vertex_indices,
                                         int triangle_index,
                                         int next,
                                         int twin)
    : next_(next),
      twin_(twin),
      vertex_indices_(vertex_indices),
      triangle_index_(triangle_index) {}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::Clear() {
    // TriangleMesh::Clear();
    half_edges_.clear();
    ordered_half_edge_from_vertex_.clear();
    return *this;
}

bool HalfEdgeTriangleMesh::ComputeHalfEdges() {
    // Clean up half-edge related data structures
    half_edges_.clear();
    ordered_half_edge_from_vertex_.clear();

    // Collect half edges
    // Check: for valid manifolds, there mustn't be duplicated half-edges
    std::unordered_map<Eigen::Vector2i, size_t,
                       utility::hash_eigen::hash<Eigen::Vector2i>>
            vertex_indices_to_half_edge_index;

    for (size_t triangle_index = 0; triangle_index < triangles_.size();
         triangle_index++) {
        const Eigen::Vector3i &triangle = triangles_[triangle_index];
        size_t num_half_edges = half_edges_.size();

        size_t he_0_index = num_half_edges;
        size_t he_1_index = num_half_edges + 1;
        size_t he_2_index = num_half_edges + 2;
        HalfEdge he_0(Eigen::Vector2i(triangle(0), triangle(1)),
                      int(triangle_index), int(he_1_index), -1);
        HalfEdge he_1(Eigen::Vector2i(triangle(1), triangle(2)),
                      int(triangle_index), int(he_2_index), -1);
        HalfEdge he_2(Eigen::Vector2i(triangle(2), triangle(0)),
                      int(triangle_index), int(he_0_index), -1);

        if (vertex_indices_to_half_edge_index.find(he_0.vertex_indices_) !=
                    vertex_indices_to_half_edge_index.end() ||
            vertex_indices_to_half_edge_index.find(he_1.vertex_indices_) !=
                    vertex_indices_to_half_edge_index.end() ||
            vertex_indices_to_half_edge_index.find(he_2.vertex_indices_) !=
                    vertex_indices_to_half_edge_index.end()) {
            utility::LogWarning(
                    "ComputeHalfEdges failed. Duplicated half-edges.\n");
            return false;
        }

        half_edges_.push_back(he_0);
        half_edges_.push_back(he_1);
        half_edges_.push_back(he_2);
        vertex_indices_to_half_edge_index[he_0.vertex_indices_] = he_0_index;
        vertex_indices_to_half_edge_index[he_1.vertex_indices_] = he_1_index;
        vertex_indices_to_half_edge_index[he_2.vertex_indices_] = he_2_index;
    }

    // Fill twin half-edge. In the previous step, it is already guaranteed that
    // each half-edge can have at most one twin half-edge.
    for (size_t this_he_index = 0; this_he_index < half_edges_.size();
         this_he_index++) {
        HalfEdge &this_he = half_edges_[this_he_index];
        Eigen::Vector2i twin_end_points(this_he.vertex_indices_(1),
                                        this_he.vertex_indices_(0));
        if (this_he.twin_ == -1 &&
            vertex_indices_to_half_edge_index.find(twin_end_points) !=
                    vertex_indices_to_half_edge_index.end()) {
            size_t twin_he_index =
                    vertex_indices_to_half_edge_index[twin_end_points];
            HalfEdge &twin_he = half_edges_[twin_he_index];
            this_he.twin_ = int(twin_he_index);
            twin_he.twin_ = int(this_he_index);
        }
    }

    // Get out-going half-edges from each vertex. This can be done during
    // half-edge construction. Done here for readability.
    std::vector<std::vector<int>> half_edges_from_vertex(vertices_.size());
    for (size_t half_edge_index = 0; half_edge_index < half_edges_.size();
         half_edge_index++) {
        int src_vertex_index = half_edges_[half_edge_index].vertex_indices_(0);
        half_edges_from_vertex[src_vertex_index].push_back(
                int(half_edge_index));
    }

    // Find ordered half-edges from each vertex by traversal. To be a valid
    // manifold, there can be at most 1 boundary half-edge from each vertex.
    ordered_half_edge_from_vertex_.resize(vertices_.size());
    for (size_t vertex_index = 0; vertex_index < vertices_.size();
         vertex_index++) {
        size_t num_boundaries = 0;
        int init_half_edge_index = 0;
        for (const int &half_edge_index :
             half_edges_from_vertex[vertex_index]) {
            if (half_edges_[half_edge_index].IsBoundary()) {
                num_boundaries++;
                init_half_edge_index = half_edge_index;
            }
        }
        if (num_boundaries > 1) {
            utility::LogWarning("ComputeHalfEdges failed. Invalid vertex.\n");
            return false;
        }
        // If there is a boundary edge, start from that; otherwise start
        // with any half-edge (default 0) started from this vertex.
        if (num_boundaries == 0) {
            init_half_edge_index = half_edges_from_vertex[vertex_index][0];
        }

        // Push edges to ordered_half_edge_from_vertex_.
        int curr_he_index = init_half_edge_index;
        ordered_half_edge_from_vertex_[vertex_index].push_back(curr_he_index);
        int next_next_twin_he_index = NextHalfEdgeFromVertex(curr_he_index);
        curr_he_index = next_next_twin_he_index;
        while (curr_he_index != -1 && curr_he_index != init_half_edge_index) {
            ordered_half_edge_from_vertex_[vertex_index].push_back(
                    curr_he_index);
            next_next_twin_he_index = NextHalfEdgeFromVertex(curr_he_index);
            curr_he_index = next_next_twin_he_index;
        }
    }
    return true;
}

bool HalfEdgeTriangleMesh::HasHalfEdges() const {
    return half_edges_.size() > 0 &&
           vertices_.size() == ordered_half_edge_from_vertex_.size();
}

int HalfEdgeTriangleMesh::NextHalfEdgeFromVertex(int half_edge_index) const {
    const HalfEdge &curr_he = half_edges_[half_edge_index];
    int next_he_index = curr_he.next_;
    const HalfEdge &next_he = half_edges_[next_he_index];
    int next_next_he_index = next_he.next_;
    const HalfEdge &next_next_he = half_edges_[next_next_he_index];
    int next_next_twin_he_index = next_next_he.twin_;
    return next_next_twin_he_index;
}

std::vector<int> HalfEdgeTriangleMesh::BoundaryHalfEdgesFromVertex(
        int vertex_index) const {
    int init_he_index = ordered_half_edge_from_vertex_[vertex_index][0];
    const HalfEdge &init_he = half_edges_[init_he_index];

    if (!init_he.IsBoundary()) {
        utility::LogWarning("The vertex {:d} is not on boundary.\n",
                            vertex_index);
        return {};
    }

    std::vector<int> boundary_half_edge_indices;
    int curr_he_index = init_he_index;
    boundary_half_edge_indices.push_back(curr_he_index);
    curr_he_index = NextHalfEdgeOnBoundary(curr_he_index);
    while (curr_he_index != init_he_index && curr_he_index != -1) {
        boundary_half_edge_indices.push_back(curr_he_index);
        curr_he_index = NextHalfEdgeOnBoundary(curr_he_index);
    }
    return boundary_half_edge_indices;
}

std::vector<int> HalfEdgeTriangleMesh::BoundaryVerticesFromVertex(
        int vertex_index) const {
    std::vector<int> boundary_half_edges =
            BoundaryHalfEdgesFromVertex(vertex_index);
    std::vector<int> boundary_vertices;
    for (const int &half_edge_idx : boundary_half_edges) {
        boundary_vertices.push_back(
                half_edges_[half_edge_idx].vertex_indices_(0));
    }
    return boundary_vertices;
}

std::vector<std::vector<int>> HalfEdgeTriangleMesh::GetBoundaries() const {
    std::vector<std::vector<int>> boundaries;
    std::unordered_set<int> visited;

    for (int vertex_ind = 0; vertex_ind < int(vertices_.size()); ++vertex_ind) {
        if (visited.find(vertex_ind) != visited.end()) {
            continue;
        }
        // It is guaranteed that if a vertex in on boundary, the starting
        // edge must be on boundary. After purging, it's also guaranteed that
        // a vertex always have out-going half-edges after purging.
        int first_half_edge_ind = ordered_half_edge_from_vertex_[vertex_ind][0];
        if (half_edges_[first_half_edge_ind].IsBoundary()) {
            std::vector<int> boundary = BoundaryVerticesFromVertex(vertex_ind);
            boundaries.push_back(boundary);
            for (int boundary_vertex : boundary) {
                visited.insert(boundary_vertex);
            }
        }
        visited.insert(vertex_ind);
    }
    return boundaries;
}

bool HalfEdgeTriangleMesh::IsEmpty() const { return !HasVertices(); }

Eigen::Vector3d HalfEdgeTriangleMesh::GetMinBound() const {
    return ComputeMinBound(vertices_);
}

Eigen::Vector3d HalfEdgeTriangleMesh::GetMaxBound() const {
    return ComputeMaxBound(vertices_);
}

Eigen::Vector3d HalfEdgeTriangleMesh::GetCenter() const {
    return ComputeCenter(vertices_);
}

AxisAlignedBoundingBox HalfEdgeTriangleMesh::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(vertices_);
}

OrientedBoundingBox HalfEdgeTriangleMesh::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromPoints(vertices_);
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::Transform(
        const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, vertices_);
    TransformNormals(transformation, vertex_normals_);
    TransformNormals(transformation, triangle_normals_);
    return *this;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::Translate(
        const Eigen::Vector3d &translation, bool relative) {
    TranslatePoints(translation, vertices_, relative);
    return *this;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::Scale(const double scale,
                                                  bool center) {
    ScalePoints(scale, vertices_, center);
    return *this;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::Rotate(
        const Eigen::Vector3d &rotation, bool center, RotationType type) {
    RotatePoints(rotation, vertices_, center, type);
    RotateNormals(rotation, vertex_normals_, center, type);
    RotateNormals(rotation, triangle_normals_, center, type);
    return *this;
}

int HalfEdgeTriangleMesh::NextHalfEdgeOnBoundary(
        int curr_half_edge_index) const {
    if (!HasHalfEdges() || curr_half_edge_index >= int(half_edges_.size()) ||
        curr_half_edge_index == -1) {
        utility::LogWarning(
                "edge index {:d} out of range or half-edges not available.\n",
                curr_half_edge_index);
        return -1;
    }
    if (!half_edges_[curr_half_edge_index].IsBoundary()) {
        utility::LogWarning(
                "The currented half-edge index {:d} is on boundary.\n",
                curr_half_edge_index);
        return -1;
    }

    // curr_half_edge's end point and next_half_edge's start point is the same
    // vertex. It is guaranteed that next_half_edge is the first edge
    // in ordered_half_edge_from_vertex_ and next_half_edge is a boundary edge.
    int vertex_index = half_edges_[curr_half_edge_index].vertex_indices_(1);
    int next_half_edge_index = ordered_half_edge_from_vertex_[vertex_index][0];
    if (!half_edges_[next_half_edge_index].IsBoundary()) {
        utility::LogError(
                "Internal algorithm error. The next half-edge along the "
                "boundary is not a boundary edge.\n");
        return -1;
    }
    return next_half_edge_index;
}

std::shared_ptr<HalfEdgeTriangleMesh> HalfEdgeTriangleMesh::CreateFromMesh(
        const TriangleMesh &mesh) {
    auto half_edge_mesh = std::make_shared<HalfEdgeTriangleMesh>();

    // Copy
    half_edge_mesh->vertices_ = mesh.vertices_;
    half_edge_mesh->vertex_normals_ = mesh.vertex_normals_;
    half_edge_mesh->vertex_colors_ = mesh.vertex_colors_;
    half_edge_mesh->triangles_ = mesh.triangles_;
    half_edge_mesh->triangle_normals_ = mesh.triangle_normals_;
    half_edge_mesh->adjacency_list_ = mesh.adjacency_list_;

    // Purge to remove duplications
    half_edge_mesh->RemoveDuplicatedVertices();
    half_edge_mesh->RemoveDuplicatedTriangles();
    half_edge_mesh->RemoveUnreferencedVertices();
    half_edge_mesh->RemoveDegenerateTriangles();

    // If the original mesh is not a manifold, we set HalfEdgeTriangleMesh to
    // be empty. Caller to this constructor is responsible to checking
    // HalfEdgeTriangleMesh::IsEmpty().
    if (!half_edge_mesh->ComputeHalfEdges()) {
        throw std::runtime_error(
                "Converting mesh to half-edge mesh filed, not manifold");
    }
    return half_edge_mesh;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::RemoveDuplicatedVertices() {
    size_t before_num = vertices_.size();
    // TriangleMesh::RemoveDuplicatedVertices();
    if (HasHalfEdges() && vertices_.size() != before_num) {
        ComputeHalfEdges();
    }
    return *this;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::RemoveDuplicatedTriangles() {
    size_t before_num = triangles_.size();
    // TriangleMesh::RemoveDuplicatedTriangles();
    if (HasHalfEdges() && triangles_.size() != before_num) {
        ComputeHalfEdges();
    }
    return *this;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::RemoveUnreferencedVertices() {
    size_t before_num = vertices_.size();
    // TriangleMesh::RemoveUnreferencedVertices();
    if (HasHalfEdges() && vertices_.size() != before_num) {
        ComputeHalfEdges();
    }
    return *this;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::RemoveDegenerateTriangles() {
    size_t before_num = triangles_.size();
    // TriangleMesh::RemoveDegenerateTriangles();
    if (HasHalfEdges() && triangles_.size() != before_num) {
        ComputeHalfEdges();
    }
    return *this;
}

HalfEdgeTriangleMesh &HalfEdgeTriangleMesh::operator+=(
        const HalfEdgeTriangleMesh &mesh) {
    // TriangleMesh::operator+=(mesh);
    if (HasHalfEdges()) {
        ComputeHalfEdges();
    }
    return *this;
}

HalfEdgeTriangleMesh HalfEdgeTriangleMesh::operator+(
        const HalfEdgeTriangleMesh &mesh) const {
    return (HalfEdgeTriangleMesh(*this) += mesh);
}

}  // namespace geometry
}  // namespace open3d
