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
#include "Open3D/Geometry/MeshBase.h"

namespace open3d {
namespace geometry {

/// \class HalfEdgeTriangleMesh
///
/// \brief HalfEdgeTriangleMesh inherits TriangleMesh class with the addition of
/// HalfEdge data structure for each half edge in the mesh as well as related
/// functions.
class HalfEdgeTriangleMesh : public MeshBase {
public:
    /// \class HalfEdge
    ///
    /// \brief HalfEdge class contains vertex, triangle info about a half edge,
    /// as well as relations of next and twin half edge.
    class HalfEdge {
    public:
        /// \brief Default Constructor.
        ///
        /// Initializes all members of the instance with invalid values.
        HalfEdge()
            : next_(-1),
              twin_(-1),
              vertex_indices_(-1, -1),
              triangle_index_(-1) {}
        HalfEdge(const Eigen::Vector2i &vertex_indices,
                 int triangle_index,
                 int next,
                 int twin);
        /// Returns `true` iff the half edge is the boundary (has not twin, i.e.
        /// twin index == -1).
        bool IsBoundary() const { return twin_ == -1; }

    public:
        /// Index of the next HalfEdge in the same triangle.
        int next_;
        /// Index of the twin HalfEdge.
        int twin_;
        /// Index of the ordered vertices forming this half edge.
        Eigen::Vector2i vertex_indices_;
        /// Index of the triangle containing this half edge.
        int triangle_index_;
    };

public:
    /// \brief Default Constructor.
    ///
    /// Creates an empty instance with GeometryType of HalfEdgeTriangleMesh.
    HalfEdgeTriangleMesh()
        : MeshBase(Geometry::GeometryType::HalfEdgeTriangleMesh) {}

    virtual HalfEdgeTriangleMesh &Clear() override;

    /// Returns `true` if half-edges have already been computed.
    bool HasHalfEdges() const;

    /// Query manifold boundary half edges from a starting vertex
    /// If query vertex is not on boundary, empty vector will be returned.
    std::vector<int> BoundaryHalfEdgesFromVertex(int vertex_index) const;

    /// Query manifold boundary vertices from a starting vertex
    /// If query vertex is not on boundary, empty vector will be returned.
    std::vector<int> BoundaryVerticesFromVertex(int vertex_index) const;

    /// Returns a vector of boundaries. A boundary is a vector of vertices.
    std::vector<std::vector<int>> GetBoundaries() const;

    HalfEdgeTriangleMesh &operator+=(const HalfEdgeTriangleMesh &mesh);

    HalfEdgeTriangleMesh operator+(const HalfEdgeTriangleMesh &mesh) const;

    /// Convert HalfEdgeTriangleMesh from TriangleMesh. Throws exception if the
    /// input mesh is not manifold.
    static std::shared_ptr<HalfEdgeTriangleMesh> CreateFromTriangleMesh(
            const TriangleMesh &mesh);

protected:
    /// \brief Parameterized Constructor.
    ///
    /// Creates an empty instance with GeometryType of specified type.
    ///
    /// \param type Specifies GeometryType for the HalfEdgeTriangleMesh.
    HalfEdgeTriangleMesh(Geometry::GeometryType type) : MeshBase(type) {}

    /// Returns the next half edge from starting vertex of the input half edge,
    /// in a counterclock wise manner. Returns -1 if when hitting a boundary.
    /// This is done by traversing to the next, next and twin half edge.
    int NextHalfEdgeFromVertex(int init_half_edge_index) const;
    int NextHalfEdgeOnBoundary(int curr_half_edge_index) const;

public:
    /// List of triangles in the mesh.
    std::vector<Eigen::Vector3i> triangles_;
    /// List of triangle normals in the mesh.
    std::vector<Eigen::Vector3d> triangle_normals_;
    /// List of HalfEdge in the mesh.
    std::vector<HalfEdge> half_edges_;

    /// Counter-clockwise ordered half-edges started from each vertex.
    /// If the vertex is on boundary, the starting edge must be on boundary too.
    std::vector<std::vector<int>> ordered_half_edge_from_vertex_;
};

}  // namespace geometry
}  // namespace open3d
