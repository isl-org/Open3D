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

#include "open3d/geometry/HalfEdgeTriangleMesh.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"

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

        edges_to_triangles_ = mesh_->GetEdgeToTrianglesMap();
    }

    std::pair<int, int> GetAdjacentFrontIndices(int index,
                                                std::vector<int> hole) {
        const int hidx1 = index - 1 < 0 ? hole.size() - 1 : index - 1;
        const int hidx2 = index + 1 >= hole.size() ? 0 : index + 1;
        return std::pair<int, int>(hidx1, hidx2);
    }

    bool IsExistingTriangle(size_t vidx0, size_t vidx1, size_t vidx2) {
        auto triangles_indices =
                edges_to_triangles_[mesh_->GetOrderedEdge(vidx0, vidx1)];
        for (size_t tidx : triangles_indices) {
            const auto &triangle = mesh_->triangles_[tidx];
            if (triangle(0) == vidx2 || triangle(1) == vidx2 ||
                triangle(2) == vidx2) {
                return true;
            }
        }
        return false;
    }

    double ComputeAngle(size_t vidx0, size_t vidx1, size_t vidx2) {
        // Return 2 * M_PI if a triangle with the indices vidx0, vidx1 and vidx2
        // (in any order) already exists.
        if (IsExistingTriangle(vidx0, vidx1, vidx2)) {
            return 2 * M_PI;
        }

        return TriangleMesh::ComputeAngle(mesh_->vertices_[vidx0],
                                          mesh_->vertices_[vidx1],
                                          mesh_->vertices_[vidx2]);
    }

    void ComputeAnglesForNeighbors(std::vector<double> &angles,
                                   std::vector<int> &front,
                                   std::pair<int, int> adjacent_front_indices) {
        // Update the vertex angles for vidx_prev and vidx_next
        const size_t vidx_prev = front[adjacent_front_indices.first];
        const size_t vidx_next = front[adjacent_front_indices.second];
        auto adjacent_front_indices_to_neighbors =
                GetAdjacentFrontIndices(adjacent_front_indices.first, front);
        angles[adjacent_front_indices.first] = ComputeAngle(
                front[adjacent_front_indices_to_neighbors.first], vidx_prev,
                front[adjacent_front_indices_to_neighbors.second]);
        adjacent_front_indices_to_neighbors =
                GetAdjacentFrontIndices(adjacent_front_indices.second, front);
        angles[adjacent_front_indices.second] = ComputeAngle(
                front[adjacent_front_indices_to_neighbors.first], vidx_next,
                front[adjacent_front_indices_to_neighbors.second]);
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
            utility::LogError("The mesh is non-manifold\n");
            return;
        }

        boundaries_ = het_mesh->GetBoundaries();
        utility::LogDebug("Number of holes: {}\n",
                          std::to_string(boundaries_.size()));
    }

    void TriangulateHoles() {
        for (std::vector<int> front : boundaries_) {
            int size = int(front.size());
            std::vector<double> angles(size, 0);

            // Calculate the angle between two adjacent boundary edges at each
            // vertex on the front.
            for (int i = 0; i < size; i++) {
                auto adjacent_front_indices = GetAdjacentFrontIndices(i, front);
                angles[i] = ComputeAngle(front[adjacent_front_indices.first],
                                         front[i],
                                         front[adjacent_front_indices.second]);
            }

            while (front.size() > 3) {
                int min_angle_index =
                        std::min_element(angles.begin(), angles.end()) -
                        angles.begin();
                double min_angle = angles[min_angle_index];
                const size_t vidx = front[min_angle_index];

                auto adjacent_front_indices =
                        GetAdjacentFrontIndices(min_angle_index, front);
                const size_t &vidx_prev = front[adjacent_front_indices.first];
                const size_t &vidx_next = front[adjacent_front_indices.second];

                std::vector<Eigen::Vector3d> vertex_candidates;
                if (min_angle < lower_angle_threshold_) {
                    // Do nothing.
                } else if (lower_angle_threshold_ < min_angle &&
                           min_angle <= upper_angle_threshold_) {
                    // TODO: Compute one vertex candidate
                } else if (upper_angle_threshold_ < min_angle &&
                           min_angle < M_PI) {
                    // TODO: Compute two vertex candidates
                }

                switch (vertex_candidates.size()) {
                    case 0: {
                        // Add one triangle, but no new vertices.
                        AddTriangle(vidx_prev, vidx_next, vidx);

                        mesh_->adjacency_list_[vidx_prev].insert(vidx_next);
                        mesh_->adjacency_list_[vidx_next].insert(vidx_prev);

                        front.erase(front.begin() + min_angle_index);
                        angles.erase(angles.begin() + min_angle_index);
                        adjacent_front_indices.second =
                                adjacent_front_indices.second - 1 < 0
                                        ? front.size() - 1
                                        : adjacent_front_indices.second - 1;
                        ComputeAnglesForNeighbors(angles, front,
                                                  adjacent_front_indices);
                    }
                }
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
    std::unordered_map<Eigen::Vector2i,
                       std::vector<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
            edges_to_triangles_;

    double lower_angle_threshold_ = M_PI;  // TODO: Should be 75 degrees
    double upper_angle_threshold_ = M_PI;  // TODO: Should be 135 degrees
};

std::shared_ptr<TriangleMesh> TriangleMesh::FillHoles() {
    HoleFilling hf(*this);
    return hf.Run();
};

}  // namespace geometry
}  // namespace open3d