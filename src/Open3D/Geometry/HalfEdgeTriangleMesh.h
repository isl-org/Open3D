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

#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Geometry/Geometry3D.h"

namespace open3d {

class HalfEdge {
public:
    HalfEdge(const Eigen::Vector2i& vertex_indices,
             int triangle_index,
             int next,
             int twin);
    bool IsBoundary() const { return twin_ == -1; }

public:
    // Index of the next HalfEdge
    int next_ = -1;
    // Index of the twin HalfEdge
    int twin_ = -1;
    // Index of the ordered vertices forming this half edge
    Eigen::Vector2i vertex_indices_ = Eigen::Vector2i(-1, -1);
    // Index of the triangle containing this half edge
    int triangle_index_ = -1;
};

class HalfEdgeTriangleMesh : public TriangleMesh {
public:
    HalfEdgeTriangleMesh()
        : TriangleMesh(Geometry::GeometryType::HalfEdgeTriangleMesh){};

    /// Copy constructocr to convert TriangleMesh to HalfEdgeTriangleMesh
    HalfEdgeTriangleMesh(const TriangleMesh& triangle_mesh);

    /// Compute and update half edges, half edge can only be computed if the
    /// mesh is a manifold. Returns true if half edges are computed.
    bool ComputeHalfEdges();

    /// True if half-edges have already been computed
    bool HasHalfEdges() const;

    /// Clear all data in HalfEdgeTriangleMesh
    void Clear() override;

    /// Query manifold boundary
    /// If query vertex is not on boundary, empty vector will be returned
    std::vector<int> BoundaryHalfEdgesFromVertex(int vertex_index) const;

public:
    std::vector<HalfEdge> half_edges_;
    // Counter-clockwise ordered half-edges started from each vertex
    // If the vertex is on boundary, the starting edge must be on boundary too
    std::vector<std::vector<int>> ordered_half_edge_from_vertex_;

protected:
    int NextNextTwinHalfEdgeIndex(int init_half_edge_index) const;
    int NextHalfEdgeOnBoundary(int curr_half_edge_index) const;
};

}  // namespace open3d
