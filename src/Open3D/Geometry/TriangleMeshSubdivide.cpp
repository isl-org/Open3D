// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "Open3D/Geometry/TriangleMesh.h"

#include <Eigen/Dense>
#include <queue>
#include <tuple>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

std::shared_ptr<TriangleMesh> SubdivideMidpoint(const TriangleMesh& input,
                                                int number_of_iterations) {
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_ = input.vertices_;
    mesh->vertex_colors_ = input.vertex_colors_;
    mesh->vertex_normals_ = input.vertex_normals_;
    mesh->triangles_ = input.triangles_;

    bool has_vert_normal = input.HasVertexNormals();
    bool has_vert_color = input.HasVertexColors();
    // Compute and return midpoint.
    // Also adds edge - new vertex refrence to new_verts map.
    auto SubdivideEdge =
            [&](std::unordered_map<Edge, int, utility::hash_tuple::hash<Edge>>&
                        new_verts,
                int vidx0, int vidx1) {
                int min = std::min(vidx0, vidx1);
                int max = std::max(vidx0, vidx1);
                Edge edge(min, max);
                if (new_verts.count(edge) == 0) {
                    mesh->vertices_.push_back(0.5 * (mesh->vertices_[min] +
                                                     mesh->vertices_[max]));
                    if (has_vert_normal) {
                        mesh->vertex_normals_.push_back(
                                0.5 * (mesh->vertex_normals_[min] +
                                       mesh->vertex_normals_[max]));
                    }
                    if (has_vert_color) {
                        mesh->vertex_colors_.push_back(
                                0.5 * (mesh->vertex_colors_[min] +
                                       mesh->vertex_colors_[max]));
                    }
                    int v01idx = mesh->vertices_.size() - 1;
                    new_verts[edge] = v01idx;
                    return v01idx;
                } else {
                    return new_verts[edge];
                }
            };
    for (int iter = 0; iter < number_of_iterations; ++iter) {
        std::unordered_map<Edge, int, utility::hash_tuple::hash<Edge>>
                new_verts;
        std::vector<Eigen::Vector3i> new_triangles(4 * mesh->triangles_.size());
        for (size_t tidx = 0; tidx < mesh->triangles_.size(); ++tidx) {
            const auto& triangle = mesh->triangles_[tidx];
            int vidx0 = triangle(0);
            int vidx1 = triangle(1);
            int vidx2 = triangle(2);
            int vidx01 = SubdivideEdge(new_verts, vidx0, vidx1);
            int vidx12 = SubdivideEdge(new_verts, vidx1, vidx2);
            int vidx20 = SubdivideEdge(new_verts, vidx2, vidx0);
            new_triangles[tidx * 4 + 0] =
                    Eigen::Vector3i(vidx0, vidx01, vidx20);
            new_triangles[tidx * 4 + 1] =
                    Eigen::Vector3i(vidx01, vidx1, vidx12);
            new_triangles[tidx * 4 + 2] =
                    Eigen::Vector3i(vidx12, vidx2, vidx20);
            new_triangles[tidx * 4 + 3] =
                    Eigen::Vector3i(vidx01, vidx12, vidx20);
        }
        mesh->triangles_ = new_triangles;
    }

    return mesh;
}

}  // namespace geometry
}  // namespace open3d
