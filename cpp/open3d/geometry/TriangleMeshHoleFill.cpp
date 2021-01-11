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
#include <map>

#include "open3d/geometry/HalfEdgeTriangleMesh.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Eigen.h"

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
        const int hidx1 = index - 1 < 0 ? int(hole.size()) - 1 : index - 1;
        const int hidx2 = index + 1 >= int(hole.size()) ? 0 : index + 1;
        return std::pair<int, int>(hidx1, hidx2);
    }

    bool IsExistingTriangle(int vidx0, int vidx1, int vidx2) {
        auto triangles_indices =
                edges_to_triangles_[mesh_->GetOrderedEdge(vidx0, vidx1)];
        for (int tidx : triangles_indices) {
            const auto &triangle = mesh_->triangles_[tidx];
            if (triangle(0) == vidx2 || triangle(1) == vidx2 ||
                triangle(2) == vidx2) {
                return true;
            }
        }
        return false;
    }

    double ComputeAngle(int vidx0, int vidx1, int vidx2) {
        // Return 2 * M_PI if a triangle with the indices vidx0, vidx1 and vidx2
        // (in any order) already exists.
        if (IsExistingTriangle(vidx0, vidx1, vidx2)) {
            return 2 * M_PI;
        }

        auto &v0 = mesh_->vertices_[vidx0];
        auto &v1 = mesh_->vertices_[vidx1];
        auto &v2 = mesh_->vertices_[vidx2];

        const auto e0 = v0 - v1;
        const auto e1 = v2 - v1;
        auto acute_angle = utility::ComputeAcuteAngle(e0, e1);

        bool is_convex = hole_normal_.dot(e0.cross(e1)) > 0.0;
        if (!is_convex) {
            acute_angle = 2 * M_PI - acute_angle;
        }
        return acute_angle;
    }

    void ComputeAnglesForNeighbors(std::vector<double> &angles,
                                   std::vector<int> &front,
                                   std::pair<int, int> adjacent_front_indices) {
        adjacent_front_indices.first =
                adjacent_front_indices.first == int(front.size())
                        ? int(front.size()) - 1
                        : adjacent_front_indices.first;
        adjacent_front_indices.second =
                adjacent_front_indices.second - 1 < 0
                        ? int(front.size()) - 1
                        : adjacent_front_indices.second - 1;

        // Update the vertex angles for vidx_prev and vidx_next
        const int vidx_prev = front[adjacent_front_indices.first];
        const int vidx_next = front[adjacent_front_indices.second];
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

    bool IsInsideCircumsphere(int vidx, const Eigen::Vector2i &edge) {
        Eigen::Vector3d midpoint =
                0.5 * (mesh_->vertices_[edge[0]] + mesh_->vertices_[edge[1]]);
        double radius = (midpoint - mesh_->vertices_[edge[0]]).norm();
        return (mesh_->vertices_[vidx] - midpoint).norm() < radius;
    }

    bool RelaxEdge(
            std::shared_ptr<open3d::geometry::TriangleMesh> &patching_mesh,
            const Eigen::Vector2i &edge,
            std::vector<int> &triangle_indices) {
        for (int tidx : triangle_indices) {
            Eigen::Vector3i &triangle = patching_mesh->triangles_[tidx];
            int non_mutual_vertex_idx;
            if (triangle[0] != edge[0] && triangle[0] != edge[1]) {
                non_mutual_vertex_idx = triangle[0];
            } else if (triangle[1] != edge[0] && triangle[1] != edge[1]) {
                non_mutual_vertex_idx = triangle[1];
            } else {
                non_mutual_vertex_idx = triangle[2];
            }

            if (IsInsideCircumsphere(non_mutual_vertex_idx, edge)) {
                // TODO: Swap edge
                return true;
            }
        }

        return false;
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
        for (auto front : boundaries_) {
            int size = int(front.size());
            std::vector<double> angles(size, 0);
            hole_normal_ = Eigen::Vector3d::Zero();
            std::shared_ptr<TriangleMesh> patching_mesh =
                    std::make_shared<TriangleMesh>();

            // Compute the averaged hole normal.
            for (int i = 0; i < size; i++) {
                auto adjacent_front_indices = GetAdjacentFrontIndices(i, front);

                auto &v0 =
                        mesh_->vertices_[front[adjacent_front_indices.first]];
                auto &v1 = mesh_->vertices_[front[i]];
                auto &v2 =
                        mesh_->vertices_[front[adjacent_front_indices.second]];
                hole_normal_ += (v0 - v1).cross(v2 - v1);
            }
            hole_normal_ *= (1.0 / std::sqrt(hole_normal_.dot(hole_normal_)));

            // Calculate the angle between two adjacent boundary edges at each
            // vertex on the front.
            for (int i = 0; i < size; i++) {
                auto adjacent_front_indices = GetAdjacentFrontIndices(i, front);
                angles[i] = ComputeAngle(front[adjacent_front_indices.first],
                                         front[i],
                                         front[adjacent_front_indices.second]);
            }

            while (front.size() > 3) {
                auto min_angle_index =
                        std::min_element(angles.begin(), angles.end()) -
                        angles.begin();
                const int vidx = front[min_angle_index];

                auto adjacent_front_indices =
                        GetAdjacentFrontIndices(int(min_angle_index), front);
                const int vidx_prev = front[adjacent_front_indices.first];
                const int vidx_next = front[adjacent_front_indices.second];

                patching_mesh->triangles_.push_back(
                        Eigen::Vector3i(vidx_prev, vidx_next, vidx));

                front.erase(front.begin() + min_angle_index);
                angles.erase(angles.begin() + min_angle_index);
                ComputeAnglesForNeighbors(angles, front,
                                          adjacent_front_indices);
            }

            patching_mesh->triangles_.push_back(
                    Eigen::Vector3i(front[2], front[1], front[0]));

            patching_meshes_.push_back(patching_mesh);
        }
    }

    void RefineHolePatching() {
        int front_idx = 0;
        for (auto &front : boundaries_) {
            std::map<int, double> scale_attributes;
            for (int vidx : front) {
                double scale_attribute = 0;
                for (auto adjacent_vidx : mesh_->adjacency_list_[vidx]) {
                    scale_attribute += (mesh_->vertices_[vidx] -
                                        mesh_->vertices_[adjacent_vidx])
                                               .norm();
                }
                scale_attributes.insert({vidx, scale_attribute});
            }

            auto patching_mesh = patching_meshes_[front_idx];
            std::shared_ptr<TriangleMesh> new_patching_mesh =
                    std::make_shared<TriangleMesh>();
            bool has_added_new_triangles = false;
            while (!has_added_new_triangles) {
                has_added_new_triangles = false;
                for (auto &triangle : patching_mesh->triangles_) {
                    Eigen::Vector3d centroid = (mesh_->vertices_[triangle[0]] +
                                                mesh_->vertices_[triangle[1]] +
                                                mesh_->vertices_[triangle[2]]) /
                                               3;
                    double centroid_scale_attribute =
                            (scale_attributes[triangle[0]] +
                             scale_attributes[triangle[1]] +
                             scale_attributes[triangle[2]]) /
                            3;

                    auto ComputeTriangleAngle = [&](int vidx0, int vidx1,
                                                    int vidx2) {
                        Eigen::Vector3d &vertex0 = mesh_->vertices_[vidx0];
                        Eigen::Vector3d &vertex1 = mesh_->vertices_[vidx1];
                        Eigen::Vector3d &vertex2 = mesh_->vertices_[vidx2];
                        return utility::ComputeAcuteAngle(vertex0 - vertex1,
                                                          vertex2 - vertex1);
                    };
                    double max_inner_triangle_angle = 0;
                    for (int i = 0; i < 3; i++) {
                        double angle = ComputeTriangleAngle(
                                triangle[i], triangle[(i + 1) % 3],
                                triangle[(i + 2) % 3]);
                        if (angle > max_inner_triangle_angle) {
                            max_inner_triangle_angle = angle;
                        }
                    }

                    bool replace_triangle = false;
                    for (int vidx : {triangle[0], triangle[1], triangle[2]}) {
                        auto density_control =
                                density_control_factor *
                                (centroid - mesh_->vertices_[vidx]).norm();
                        if (density_control > scale_attributes[vidx] &&
                            density_control > centroid_scale_attribute &&
                            max_inner_triangle_angle < 5 * M_PI / 6) {
                            replace_triangle = true;
                            has_added_new_triangles = true;
                        }
                    }
                    if (replace_triangle) {
                        mesh_->vertices_.push_back(centroid);
                        int vidx_centroid = int(mesh_->vertices_.size() - 1);
                        new_patching_mesh->triangles_.push_back(Eigen::Vector3i(
                                vidx_centroid, triangle[1], triangle[2]));
                        new_patching_mesh->triangles_.push_back(Eigen::Vector3i(
                                triangle[0], vidx_centroid, triangle[2]));
                        new_patching_mesh->triangles_.push_back(Eigen::Vector3i(
                                triangle[0], triangle[1], vidx_centroid));
                    } else {
                        new_patching_mesh->triangles_.push_back(triangle);
                    }
                }
                patching_meshes_[front_idx] = new_patching_mesh;

                if (!has_added_new_triangles) {
                    break;
                }

                bool has_edges_been_swapped = false;
                while (!has_edges_been_swapped) {
                    // Relax all interior edges of the patching mesh
                    has_edges_been_swapped = false;
                    auto edges = patching_mesh->GetEdgeToTrianglesMap();
                    for (auto &kv : edges) {
                        if (kv.second.size() == 2u) {
                            bool swapped_edge = RelaxEdge(patching_mesh,
                                                          kv.first, kv.second);
                            if (swapped_edge) {
                                has_edges_been_swapped = true;
                            }
                        }
                    }
                }
            }

            front_idx++;
        }
    }

    void AddNewTrianglesToMesh() {
        for (auto &patching_mesh : patching_meshes_) {
            mesh_->triangles_.insert(mesh_->triangles_.end(),
                                     patching_mesh->triangles_.begin(),
                                     patching_mesh->triangles_.end());
        }
    }

    std::shared_ptr<TriangleMesh> Run() {
        IdentifyHoles();
        TriangulateHoles();
        RefineHolePatching();

        AddNewTrianglesToMesh();
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
    std::vector<std::shared_ptr<TriangleMesh>> patching_meshes_;

    Eigen::Vector3d hole_normal_;
    double density_control_factor = M_SQRT2;
};

std::shared_ptr<TriangleMesh> TriangleMesh::FillHoles() {
    HoleFilling hf(*this);
    return hf.Run();
};

}  // namespace geometry
}  // namespace open3d