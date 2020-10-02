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

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#include "open3d/geometry/HalfEdgeTriangleMesh.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace geometry {

class HoleFilling {
public:
    HoleFilling(const TriangleMesh &mesh) {
        mesh_ = std::make_shared<TriangleMesh>();
        mesh_->vertices_ = mesh.vertices_;
        mesh_->vertex_normals_ = mesh.vertex_normals_;
        mesh_->vertex_colors_ = mesh.vertex_colors_;
        mesh_->triangles_ = mesh.triangles_;

        mesh_->RemoveDuplicatedVertices();
        mesh_->RemoveDuplicatedTriangles();
        mesh_->RemoveUnreferencedVertices();
        mesh_->RemoveDegenerateTriangles();

        mesh_->ComputeAdjacencyList();
        mesh_->ComputeVertexNormals(/*normalized=*/false);
    }

    std::pair<int, int> GetAdjacentFrontIndices(int index,
                                                std::vector<int> hole) {
        const int hidx1 = index - 1 < 0 ? hole.size() - 1 : index - 1;
        const int hidx2 = index + 1 >= hole.size() ? 0 : index + 1;
        return std::pair<int, int>(hidx1, hidx2);
    }

    void AddTriangle(size_t vidx0, size_t vidx1, size_t vidx2) {
        mesh_->triangles_.emplace_back(Eigen::Vector3i(vidx0, vidx1, vidx2));

        auto &triangle = mesh_->triangles_.back();
        Eigen::Vector3d v1 =
                mesh_->vertices_[triangle(1)] - mesh_->vertices_[triangle(0)];
        Eigen::Vector3d v2 =
                mesh_->vertices_[triangle(2)] - mesh_->vertices_[triangle(0)];
        mesh_->triangle_normals_.push_back(v1.cross(v2));

        mesh_->vertex_normals_[triangle(0)] += mesh_->triangle_normals_.back();
        mesh_->vertex_normals_[triangle(1)] += mesh_->triangle_normals_.back();
        mesh_->vertex_normals_[triangle(2)] += mesh_->triangle_normals_.back();
    }

    void IdentifyHoles() {
        // Assume that the mesh is oriented, manifold, and connected.
        auto het_mesh = HalfEdgeTriangleMesh::CreateFromTriangleMesh(*mesh_);

        if (het_mesh->IsEmpty()) {
            utility::LogWarning("The mesh is non-manifold\n");
            return;
        }

        boundaries_ = het_mesh->GetBoundaries();
        utility::LogDebug("Number of holes: {}\n",
                          std::to_string(boundaries_.size()));
    }

    // Triangulation by the modified advancing front method.
    void TriangulateHoles() {
        for (std::vector<int> front : boundaries_) {
            int size = int(front.size());
            std::vector<double> angles(size, 0);

            // Calculate the angle between two adjacent boundary edges at each
            // vertex on the front.
            for (int i = 0; i < size; i++) {
                auto adjacent_front_indices = GetAdjacentFrontIndices(i, front);
                angles[i] = TriangleMesh::ComputeAngle(
                        mesh_->vertices_[front[adjacent_front_indices.first]],
                        mesh_->vertices_[front[i]],
                        mesh_->vertices_[front[adjacent_front_indices.second]]);
            }

            while (front.size() > 3) {
            }

            AddTriangle(front[2], front[1], front[0]);
        }
    }

    std::shared_ptr<TriangleMesh> Run() {
        IdentifyHoles();
        TriangulateHoles();

        mesh_->ComputeVertexNormals();
        return mesh_;
    }

private:
    std::shared_ptr<TriangleMesh> mesh_;
    std::vector<std::vector<int>> boundaries_;
};

std::shared_ptr<TriangleMesh> TriangleMesh::FillHoles() {
    HoleFilling hf(*this);
    return hf.Run();
};

}  // namespace geometry
}  // namespace open3d