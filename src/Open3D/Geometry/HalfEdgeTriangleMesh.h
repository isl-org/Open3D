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

#pragma once

#include <Eigen/Core>
#include <unordered_map>

#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d {
namespace geometry {

// TODO likely broken
class HalfEdgeTriangleMesh : public Geometry3D {
public:
    class HalfEdge {
    public:
        HalfEdge()
            : next_(-1),
              twin_(-1),
              vertex_indices_(-1, -1),
              triangle_index_(-1) {}
        HalfEdge(const Eigen::Vector2i &vertex_indices,
                 int triangle_index,
                 int next,
                 int twin);
        bool IsBoundary() const { return twin_ == -1; }

    public:
        // Index of the next HalfEdge
        int next_;
        // Index of the twin HalfEdge
        int twin_;
        // Index of the ordered vertices forming this half edge
        Eigen::Vector2i vertex_indices_;
        // Index of the triangle containing this half edge
        int triangle_index_;
    };

public:
    HalfEdgeTriangleMesh()
        : Geometry3D(Geometry::GeometryType::HalfEdgeTriangleMesh) {}

    /// Clear all data in HalfEdgeTriangleMesh
    HalfEdgeTriangleMesh &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const override;
    HalfEdgeTriangleMesh &Transform(
            const Eigen::Matrix4d &transformation) override;
    HalfEdgeTriangleMesh &Translate(const Eigen::Vector3d &translation,
                                    bool relative = true) override;
    HalfEdgeTriangleMesh &Scale(const double scale,
                                bool center = true) override;
    HalfEdgeTriangleMesh &Rotate(
            const Eigen::Vector3d &rotation,
            bool center = true,
            RotationType type = RotationType::XYZ) override;

    /// Compute and update half edges, half edge can only be computed if the
    /// mesh is a manifold. Returns true if half edges are computed.
    bool ComputeHalfEdges();

    bool HasVertices() const { return vertices_.size() > 0; }

    /// True if half-edges have already been computed
    bool HasHalfEdges() const;

    /// Query manifold boundary half edges from a starting vertex
    /// If query vertex is not on boundary, empty vector will be returned
    std::vector<int> BoundaryHalfEdgesFromVertex(int vertex_index) const;

    /// Query manifold boundary vertices from a starting vertex
    /// If query vertex is not on boundary, empty vector will be returned
    std::vector<int> BoundaryVerticesFromVertex(int vertex_index) const;

    /// Returns a vector of boundaries. A boundary is a vector of vertices.
    std::vector<std::vector<int>> GetBoundaries() const;

    HalfEdgeTriangleMesh &RemoveDuplicatedVertices();
    HalfEdgeTriangleMesh &RemoveDuplicatedTriangles();
    HalfEdgeTriangleMesh &RemoveUnreferencedVertices();
    HalfEdgeTriangleMesh &RemoveDegenerateTriangles();

    HalfEdgeTriangleMesh &operator+=(const HalfEdgeTriangleMesh &mesh);

    HalfEdgeTriangleMesh operator+(const HalfEdgeTriangleMesh &mesh) const;

    static std::shared_ptr<HalfEdgeTriangleMesh> CreateFromMesh(
            const TriangleMesh &mesh);

protected:
    HalfEdgeTriangleMesh(Geometry::GeometryType type) : Geometry3D(type) {}

    /// Returns the next half edge from starting vertex of the input half edge,
    /// in a counterclock wise manner. Returns -1 if when hitting a boundary.
    /// This is done by traversing to the next, next and twin half edge.
    int NextHalfEdgeFromVertex(int init_half_edge_index) const;
    int NextHalfEdgeOnBoundary(int curr_half_edge_index) const;

public:
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<Eigen::Vector3d> vertex_normals_;
    std::vector<Eigen::Vector3d> vertex_colors_;
    std::vector<Eigen::Vector3i> triangles_;
    std::vector<Eigen::Vector3d> triangle_normals_;
    std::vector<std::unordered_set<int>> adjacency_list_;

    std::vector<HalfEdge> half_edges_;

    /// Counter-clockwise ordered half-edges started from each vertex
    /// If the vertex is on boundary, the starting edge must be on boundary too
    std::vector<std::vector<int>> ordered_half_edge_from_vertex_;
};

}  // namespace geometry
}  // namespace open3d
