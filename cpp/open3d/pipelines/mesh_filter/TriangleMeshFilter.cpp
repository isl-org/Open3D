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

#include "open3d/pipelines/mesh_filter/TriangleMeshFilter.h"

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/utility/Console.h"

#include <queue>
#include <random>

namespace open3d {
namespace pipelines {
namespace mesh_filter {

std::shared_ptr<geometry::TriangleMesh> FilterSharpen(
        const geometry::TriangleMesh &mesh,
        int number_of_iterations,
        double strength,
        FilterScope scope) {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            mesh.HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            mesh.HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = mesh.vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = mesh.vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = mesh.vertex_colors_;

    std::shared_ptr<geometry::TriangleMesh> out_mesh =
            std::make_shared<geometry::TriangleMesh>();
    out_mesh->vertices_.resize(mesh.vertices_.size());
    out_mesh->vertex_normals_.resize(mesh.vertex_normals_.size());
    out_mesh->vertex_colors_.resize(mesh.vertex_colors_.size());
    out_mesh->triangles_ = mesh.triangles_;
    out_mesh->adjacency_list_ = mesh.adjacency_list_;
    if (!out_mesh->HasAdjacencyList()) {
        out_mesh->ComputeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        for (size_t vidx = 0; vidx < out_mesh->vertices_.size(); ++vidx) {
            Eigen::Vector3d vertex_sum(0, 0, 0);
            Eigen::Vector3d normal_sum(0, 0, 0);
            Eigen::Vector3d color_sum(0, 0, 0);
            for (int nbidx : out_mesh->adjacency_list_[vidx]) {
                if (filter_vertex) {
                    vertex_sum += prev_vertices[nbidx];
                }
                if (filter_normal) {
                    normal_sum += prev_vertex_normals[nbidx];
                }
                if (filter_color) {
                    color_sum += prev_vertex_colors[nbidx];
                }
            }

            size_t nb_size = out_mesh->adjacency_list_[vidx].size();
            if (filter_vertex) {
                out_mesh->vertices_[vidx] =
                        prev_vertices[vidx] +
                        strength * (prev_vertices[vidx] * nb_size - vertex_sum);
            }
            if (filter_normal) {
                out_mesh->vertex_normals_[vidx] =
                        prev_vertex_normals[vidx] +
                        strength * (prev_vertex_normals[vidx] * nb_size -
                                    normal_sum);
            }
            if (filter_color) {
                out_mesh->vertex_colors_[vidx] =
                        prev_vertex_colors[vidx] +
                        strength * (prev_vertex_colors[vidx] * nb_size -
                                    color_sum);
            }
        }
        if (iter < number_of_iterations - 1) {
            std::swap(out_mesh->vertices_, prev_vertices);
            std::swap(out_mesh->vertex_normals_, prev_vertex_normals);
            std::swap(out_mesh->vertex_colors_, prev_vertex_colors);
        }
    }

    return out_mesh;
}

std::shared_ptr<geometry::TriangleMesh> FilterSmoothSimple(
        const geometry::TriangleMesh &mesh,
        int number_of_iterations,
        FilterScope scope) {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            mesh.HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            mesh.HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = mesh.vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = mesh.vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = mesh.vertex_colors_;

    std::shared_ptr<geometry::TriangleMesh> out_mesh =
            std::make_shared<geometry::TriangleMesh>();
    out_mesh->vertices_.resize(mesh.vertices_.size());
    out_mesh->vertex_normals_.resize(mesh.vertex_normals_.size());
    out_mesh->vertex_colors_.resize(mesh.vertex_colors_.size());
    out_mesh->triangles_ = mesh.triangles_;
    out_mesh->adjacency_list_ = mesh.adjacency_list_;
    if (!out_mesh->HasAdjacencyList()) {
        out_mesh->ComputeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        for (size_t vidx = 0; vidx < out_mesh->vertices_.size(); ++vidx) {
            Eigen::Vector3d vertex_sum(0, 0, 0);
            Eigen::Vector3d normal_sum(0, 0, 0);
            Eigen::Vector3d color_sum(0, 0, 0);
            for (int nbidx : out_mesh->adjacency_list_[vidx]) {
                if (filter_vertex) {
                    vertex_sum += prev_vertices[nbidx];
                }
                if (filter_normal) {
                    normal_sum += prev_vertex_normals[nbidx];
                }
                if (filter_color) {
                    color_sum += prev_vertex_colors[nbidx];
                }
            }

            size_t nb_size = out_mesh->adjacency_list_[vidx].size();
            if (filter_vertex) {
                out_mesh->vertices_[vidx] =
                        (prev_vertices[vidx] + vertex_sum) / (1 + nb_size);
            }
            if (filter_normal) {
                out_mesh->vertex_normals_[vidx] =
                        (prev_vertex_normals[vidx] + normal_sum) /
                        (1 + nb_size);
            }
            if (filter_color) {
                out_mesh->vertex_colors_[vidx] =
                        (prev_vertex_colors[vidx] + color_sum) / (1 + nb_size);
            }
        }
        if (iter < number_of_iterations - 1) {
            std::swap(out_mesh->vertices_, prev_vertices);
            std::swap(out_mesh->vertex_normals_, prev_vertex_normals);
            std::swap(out_mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return out_mesh;
}

void FilterSmoothLaplacianHelper(
        std::shared_ptr<geometry::TriangleMesh> &mesh,
        const std::vector<Eigen::Vector3d> &prev_vertices,
        const std::vector<Eigen::Vector3d> &prev_vertex_normals,
        const std::vector<Eigen::Vector3d> &prev_vertex_colors,
        const std::vector<std::unordered_set<int>> &adjacency_list,
        double lambda,
        bool filter_vertex,
        bool filter_normal,
        bool filter_color) {
    for (size_t vidx = 0; vidx < mesh->vertices_.size(); ++vidx) {
        Eigen::Vector3d vertex_sum(0, 0, 0);
        Eigen::Vector3d normal_sum(0, 0, 0);
        Eigen::Vector3d color_sum(0, 0, 0);
        double total_weight = 0;
        for (int nbidx : mesh->adjacency_list_[vidx]) {
            auto diff = prev_vertices[vidx] - prev_vertices[nbidx];
            double dist = diff.norm();
            double weight = 1. / (dist + 1e-12);
            total_weight += weight;

            if (filter_vertex) {
                vertex_sum += weight * prev_vertices[nbidx];
            }
            if (filter_normal) {
                normal_sum += weight * prev_vertex_normals[nbidx];
            }
            if (filter_color) {
                color_sum += weight * prev_vertex_colors[nbidx];
            }
        }

        if (filter_vertex) {
            mesh->vertices_[vidx] =
                    prev_vertices[vidx] +
                    lambda * (vertex_sum / total_weight - prev_vertices[vidx]);
        }
        if (filter_normal) {
            mesh->vertex_normals_[vidx] = prev_vertex_normals[vidx] +
                                          lambda * (normal_sum / total_weight -
                                                    prev_vertex_normals[vidx]);
        }
        if (filter_color) {
            mesh->vertex_colors_[vidx] = prev_vertex_colors[vidx] +
                                         lambda * (color_sum / total_weight -
                                                   prev_vertex_colors[vidx]);
        }
    }
}

std::shared_ptr<geometry::TriangleMesh> FilterSmoothLaplacian(
        const geometry::TriangleMesh &mesh,
        int number_of_iterations,
        double lambda,
        FilterScope scope) {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            mesh.HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            mesh.HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = mesh.vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = mesh.vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = mesh.vertex_colors_;

    std::shared_ptr<geometry::TriangleMesh> out_mesh =
            std::make_shared<geometry::TriangleMesh>();
    out_mesh->vertices_.resize(mesh.vertices_.size());
    out_mesh->vertex_normals_.resize(mesh.vertex_normals_.size());
    out_mesh->vertex_colors_.resize(mesh.vertex_colors_.size());
    out_mesh->triangles_ = mesh.triangles_;
    out_mesh->adjacency_list_ = mesh.adjacency_list_;
    if (!out_mesh->HasAdjacencyList()) {
        out_mesh->ComputeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        FilterSmoothLaplacianHelper(out_mesh, prev_vertices,
                                    prev_vertex_normals, prev_vertex_colors,
                                    out_mesh->adjacency_list_, lambda,
                                    filter_vertex, filter_normal, filter_color);
        if (iter < number_of_iterations - 1) {
            std::swap(out_mesh->vertices_, prev_vertices);
            std::swap(out_mesh->vertex_normals_, prev_vertex_normals);
            std::swap(out_mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return out_mesh;
}

std::shared_ptr<geometry::TriangleMesh> FilterSmoothTaubin(
        const geometry::TriangleMesh &mesh,
        int number_of_iterations,
        double lambda,
        double mu,
        FilterScope scope) {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            mesh.HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            mesh.HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = mesh.vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = mesh.vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = mesh.vertex_colors_;

    std::shared_ptr<geometry::TriangleMesh> out_mesh =
            std::make_shared<geometry::TriangleMesh>();
    out_mesh->vertices_.resize(mesh.vertices_.size());
    out_mesh->vertex_normals_.resize(mesh.vertex_normals_.size());
    out_mesh->vertex_colors_.resize(mesh.vertex_colors_.size());
    out_mesh->triangles_ = mesh.triangles_;
    out_mesh->adjacency_list_ = mesh.adjacency_list_;
    if (!out_mesh->HasAdjacencyList()) {
        out_mesh->ComputeAdjacencyList();
    }
    for (int iter = 0; iter < number_of_iterations; ++iter) {
        FilterSmoothLaplacianHelper(out_mesh, prev_vertices,
                                    prev_vertex_normals, prev_vertex_colors,
                                    out_mesh->adjacency_list_, lambda,
                                    filter_vertex, filter_normal, filter_color);
        std::swap(out_mesh->vertices_, prev_vertices);
        std::swap(out_mesh->vertex_normals_, prev_vertex_normals);
        std::swap(out_mesh->vertex_colors_, prev_vertex_colors);
        FilterSmoothLaplacianHelper(out_mesh, prev_vertices,
                                    prev_vertex_normals, prev_vertex_colors,
                                    out_mesh->adjacency_list_, mu,
                                    filter_vertex, filter_normal, filter_color);
        if (iter < number_of_iterations - 1) {
            std::swap(out_mesh->vertices_, prev_vertices);
            std::swap(out_mesh->vertex_normals_, prev_vertex_normals);
            std::swap(out_mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return out_mesh;
}

}  // namespace mesh_filter
}  // namespace pipelines
}  // namespace open3d
