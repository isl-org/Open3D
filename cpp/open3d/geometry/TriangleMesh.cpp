// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/geometry/TriangleMesh.h"

#include <Eigen/Dense>
#include <numeric>
#include <queue>
#include <tuple>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/IntersectionTest.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace geometry {

TriangleMesh &TriangleMesh::Clear() {
    MeshBase::Clear();
    triangles_.clear();
    triangle_normals_.clear();
    adjacency_list_.clear();
    triangle_uvs_.clear();
    materials_.clear();
    triangle_material_ids_.clear();
    textures_.clear();

    return *this;
}

TriangleMesh &TriangleMesh::Transform(const Eigen::Matrix4d &transformation) {
    MeshBase::Transform(transformation);
    TransformNormals(transformation, triangle_normals_);
    return *this;
}

TriangleMesh &TriangleMesh::Rotate(const Eigen::Matrix3d &R,
                                   const Eigen::Vector3d &center) {
    MeshBase::Rotate(R, center);
    RotateNormals(R, triangle_normals_);
    return *this;
}

TriangleMesh &TriangleMesh::operator+=(const TriangleMesh &mesh) {
    if (mesh.IsEmpty()) return (*this);
    bool add_textures = HasTriangleUvs() && HasTextures() &&
                        HasTriangleMaterialIds() && mesh.HasTriangleUvs() &&
                        mesh.HasTextures() && mesh.HasTriangleMaterialIds();
    size_t old_vert_num = vertices_.size();
    MeshBase::operator+=(mesh);
    size_t old_tri_num = triangles_.size();
    size_t add_tri_num = mesh.triangles_.size();
    size_t new_tri_num = old_tri_num + add_tri_num;
    if ((!HasTriangles() || HasTriangleNormals()) &&
        mesh.HasTriangleNormals()) {
        triangle_normals_.resize(new_tri_num);
        for (size_t i = 0; i < add_tri_num; i++)
            triangle_normals_[old_tri_num + i] = mesh.triangle_normals_[i];
    } else {
        triangle_normals_.clear();
    }
    triangles_.resize(triangles_.size() + mesh.triangles_.size());
    Eigen::Vector3i index_shift((int)old_vert_num, (int)old_vert_num,
                                (int)old_vert_num);
    for (size_t i = 0; i < add_tri_num; i++) {
        triangles_[old_tri_num + i] = mesh.triangles_[i] + index_shift;
    }
    if (HasAdjacencyList()) {
        ComputeAdjacencyList();
    }
    if (add_textures) {
        size_t old_tri_uv_num = triangle_uvs_.size();
        triangle_uvs_.resize(old_tri_uv_num + mesh.triangle_uvs_.size());
        for (size_t i = 0; i < mesh.triangle_uvs_.size(); i++) {
            triangle_uvs_[old_tri_uv_num + i] = mesh.triangle_uvs_[i];
        }

        size_t old_tex_num = textures_.size();
        textures_.resize(old_tex_num + mesh.textures_.size());
        for (size_t i = 0; i < mesh.textures_.size(); i++) {
            textures_[old_tex_num + i] = mesh.textures_[i];
        }

        size_t old_mat_id_num = triangle_material_ids_.size();
        triangle_material_ids_.resize(old_mat_id_num +
                                      mesh.triangle_material_ids_.size());
        for (size_t i = 0; i < mesh.triangle_material_ids_.size(); i++) {
            triangle_material_ids_[old_mat_id_num + i] =
                    mesh.triangle_material_ids_[i] + (int)old_tex_num;
        }
    } else {
        triangle_uvs_.clear();
        textures_.clear();
        triangle_material_ids_.clear();
    }
    return (*this);
}

TriangleMesh TriangleMesh::operator+(const TriangleMesh &mesh) const {
    return (TriangleMesh(*this) += mesh);
}

TriangleMesh &TriangleMesh::ComputeTriangleNormals(
        bool normalized /* = true*/) {
    triangle_normals_.resize(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); i++) {
        auto &triangle = triangles_[i];
        Eigen::Vector3d v01 = vertices_[triangle(1)] - vertices_[triangle(0)];
        Eigen::Vector3d v02 = vertices_[triangle(2)] - vertices_[triangle(0)];
        triangle_normals_[i] = v01.cross(v02);
    }
    if (normalized) {
        NormalizeNormals();
    }
    return *this;
}

TriangleMesh &TriangleMesh::ComputeVertexNormals(bool normalized /* = true*/) {
    ComputeTriangleNormals(false);
    vertex_normals_.resize(vertices_.size(), Eigen::Vector3d::Zero());
    for (size_t i = 0; i < triangles_.size(); i++) {
        auto &triangle = triangles_[i];
        vertex_normals_[triangle(0)] += triangle_normals_[i];
        vertex_normals_[triangle(1)] += triangle_normals_[i];
        vertex_normals_[triangle(2)] += triangle_normals_[i];
    }
    if (normalized) {
        NormalizeNormals();
    }
    return *this;
}

TriangleMesh &TriangleMesh::ComputeAdjacencyList() {
    adjacency_list_.clear();
    adjacency_list_.resize(vertices_.size());
    for (const auto &triangle : triangles_) {
        adjacency_list_[triangle(0)].insert(triangle(1));
        adjacency_list_[triangle(0)].insert(triangle(2));
        adjacency_list_[triangle(1)].insert(triangle(0));
        adjacency_list_[triangle(1)].insert(triangle(2));
        adjacency_list_[triangle(2)].insert(triangle(0));
        adjacency_list_[triangle(2)].insert(triangle(1));
    }
    return *this;
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSharpen(
        int number_of_iterations, double strength, FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->HasAdjacencyList()) {
        mesh->ComputeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        for (size_t vidx = 0; vidx < mesh->vertices_.size(); ++vidx) {
            Eigen::Vector3d vertex_sum(0, 0, 0);
            Eigen::Vector3d normal_sum(0, 0, 0);
            Eigen::Vector3d color_sum(0, 0, 0);
            for (int nbidx : mesh->adjacency_list_[vidx]) {
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

            size_t nb_size = mesh->adjacency_list_[vidx].size();
            if (filter_vertex) {
                mesh->vertices_[vidx] =
                        prev_vertices[vidx] +
                        strength * (prev_vertices[vidx] * nb_size - vertex_sum);
            }
            if (filter_normal) {
                mesh->vertex_normals_[vidx] =
                        prev_vertex_normals[vidx] +
                        strength * (prev_vertex_normals[vidx] * nb_size -
                                    normal_sum);
            }
            if (filter_color) {
                mesh->vertex_colors_[vidx] =
                        prev_vertex_colors[vidx] +
                        strength * (prev_vertex_colors[vidx] * nb_size -
                                    color_sum);
            }
        }
        if (iter < number_of_iterations - 1) {
            std::swap(mesh->vertices_, prev_vertices);
            std::swap(mesh->vertex_normals_, prev_vertex_normals);
            std::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSmoothSimple(
        int number_of_iterations, FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->HasAdjacencyList()) {
        mesh->ComputeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        for (size_t vidx = 0; vidx < mesh->vertices_.size(); ++vidx) {
            Eigen::Vector3d vertex_sum(0, 0, 0);
            Eigen::Vector3d normal_sum(0, 0, 0);
            Eigen::Vector3d color_sum(0, 0, 0);
            for (int nbidx : mesh->adjacency_list_[vidx]) {
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

            size_t nb_size = mesh->adjacency_list_[vidx].size();
            if (filter_vertex) {
                mesh->vertices_[vidx] =
                        (prev_vertices[vidx] + vertex_sum) / (1 + nb_size);
            }
            if (filter_normal) {
                mesh->vertex_normals_[vidx] =
                        (prev_vertex_normals[vidx] + normal_sum) /
                        (1 + nb_size);
            }
            if (filter_color) {
                mesh->vertex_colors_[vidx] =
                        (prev_vertex_colors[vidx] + color_sum) / (1 + nb_size);
            }
        }
        if (iter < number_of_iterations - 1) {
            std::swap(mesh->vertices_, prev_vertices);
            std::swap(mesh->vertex_normals_, prev_vertex_normals);
            std::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return mesh;
}

void TriangleMesh::FilterSmoothLaplacianHelper(
        std::shared_ptr<TriangleMesh> &mesh,
        const std::vector<Eigen::Vector3d> &prev_vertices,
        const std::vector<Eigen::Vector3d> &prev_vertex_normals,
        const std::vector<Eigen::Vector3d> &prev_vertex_colors,
        const std::vector<std::unordered_set<int>> &adjacency_list,
        double lambda_filter,
        bool filter_vertex,
        bool filter_normal,
        bool filter_color) const {
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
            mesh->vertices_[vidx] = prev_vertices[vidx] +
                                    lambda_filter * (vertex_sum / total_weight -
                                                     prev_vertices[vidx]);
        }
        if (filter_normal) {
            mesh->vertex_normals_[vidx] =
                    prev_vertex_normals[vidx] +
                    lambda_filter * (normal_sum / total_weight -
                                     prev_vertex_normals[vidx]);
        }
        if (filter_color) {
            mesh->vertex_colors_[vidx] =
                    prev_vertex_colors[vidx] +
                    lambda_filter * (color_sum / total_weight -
                                     prev_vertex_colors[vidx]);
        }
    }
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSmoothLaplacian(
        int number_of_iterations,
        double lambda_filter,
        FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->HasAdjacencyList()) {
        mesh->ComputeAdjacencyList();
    }

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        FilterSmoothLaplacianHelper(mesh, prev_vertices, prev_vertex_normals,
                                    prev_vertex_colors, mesh->adjacency_list_,
                                    lambda_filter, filter_vertex, filter_normal,
                                    filter_color);
        if (iter < number_of_iterations - 1) {
            std::swap(mesh->vertices_, prev_vertices);
            std::swap(mesh->vertex_normals_, prev_vertex_normals);
            std::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSmoothTaubin(
        int number_of_iterations,
        double lambda_filter,
        double mu,
        FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    std::vector<Eigen::Vector3d> prev_vertices = vertices_;
    std::vector<Eigen::Vector3d> prev_vertex_normals = vertex_normals_;
    std::vector<Eigen::Vector3d> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->adjacency_list_ = adjacency_list_;
    if (!mesh->HasAdjacencyList()) {
        mesh->ComputeAdjacencyList();
    }
    for (int iter = 0; iter < number_of_iterations; ++iter) {
        FilterSmoothLaplacianHelper(mesh, prev_vertices, prev_vertex_normals,
                                    prev_vertex_colors, mesh->adjacency_list_,
                                    lambda_filter, filter_vertex, filter_normal,
                                    filter_color);
        std::swap(mesh->vertices_, prev_vertices);
        std::swap(mesh->vertex_normals_, prev_vertex_normals);
        std::swap(mesh->vertex_colors_, prev_vertex_colors);
        FilterSmoothLaplacianHelper(mesh, prev_vertices, prev_vertex_normals,
                                    prev_vertex_colors, mesh->adjacency_list_,
                                    mu, filter_vertex, filter_normal,
                                    filter_color);
        if (iter < number_of_iterations - 1) {
            std::swap(mesh->vertices_, prev_vertices);
            std::swap(mesh->vertex_normals_, prev_vertex_normals);
            std::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return mesh;
}

std::shared_ptr<PointCloud> TriangleMesh::SamplePointsUniformlyImpl(
        size_t number_of_points,
        std::vector<double> &triangle_areas,
        double surface_area,
        bool use_triangle_normal) {
    if (surface_area <= 0) {
        utility::LogError("Invalid surface area {}, it must be > 0.",
                          surface_area);
    }

    // triangle areas to cdf
    triangle_areas[0] /= surface_area;
    for (size_t tidx = 1; tidx < triangles_.size(); ++tidx) {
        triangle_areas[tidx] =
                triangle_areas[tidx] / surface_area + triangle_areas[tidx - 1];
    }

    // sample point cloud
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    utility::random::UniformDoubleGenerator uniform_generator(0.0, 1.0);
    auto pcd = std::make_shared<PointCloud>();
    pcd->points_.resize(number_of_points);
    if (has_vert_normal || use_triangle_normal) {
        pcd->normals_.resize(number_of_points);
    }
    if (use_triangle_normal && !HasTriangleNormals()) {
        ComputeTriangleNormals(true);
    }
    if (has_vert_color) {
        pcd->colors_.resize(number_of_points);
    }
    size_t point_idx = 0;
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        size_t n = size_t(std::round(triangle_areas[tidx] * number_of_points));
        while (point_idx < n) {
            double r1 = uniform_generator();
            double r2 = uniform_generator();
            double a = (1 - std::sqrt(r1));
            double b = std::sqrt(r1) * (1 - r2);
            double c = std::sqrt(r1) * r2;

            const Eigen::Vector3i &triangle = triangles_[tidx];
            pcd->points_[point_idx] = a * vertices_[triangle(0)] +
                                      b * vertices_[triangle(1)] +
                                      c * vertices_[triangle(2)];
            if (has_vert_normal && !use_triangle_normal) {
                pcd->normals_[point_idx] = a * vertex_normals_[triangle(0)] +
                                           b * vertex_normals_[triangle(1)] +
                                           c * vertex_normals_[triangle(2)];
            }
            if (use_triangle_normal) {
                pcd->normals_[point_idx] = triangle_normals_[tidx];
            }
            if (has_vert_color) {
                pcd->colors_[point_idx] = a * vertex_colors_[triangle(0)] +
                                          b * vertex_colors_[triangle(1)] +
                                          c * vertex_colors_[triangle(2)];
            }

            point_idx++;
        }
    }

    return pcd;
}

std::shared_ptr<PointCloud> TriangleMesh::SamplePointsUniformly(
        size_t number_of_points, bool use_triangle_normal /* = false */) {
    if (number_of_points <= 0) {
        utility::LogError("number_of_points <= 0");
    }
    if (triangles_.size() == 0) {
        utility::LogError("Input mesh has no triangles.");
    }

    // Compute area of each triangle and sum surface area
    std::vector<double> triangle_areas;
    double surface_area = GetSurfaceArea(triangle_areas);

    return SamplePointsUniformlyImpl(number_of_points, triangle_areas,
                                     surface_area, use_triangle_normal);
}

std::shared_ptr<PointCloud> TriangleMesh::SamplePointsPoissonDisk(
        size_t number_of_points,
        double init_factor /* = 5 */,
        const std::shared_ptr<PointCloud> pcl_init /* = nullptr */,
        bool use_triangle_normal /* = false */) {
    if (number_of_points <= 0) {
        utility::LogError("number_of_points <= 0");
    }
    if (triangles_.size() == 0) {
        utility::LogError("Input mesh has no triangles.");
    }
    if (pcl_init == nullptr && init_factor < 1) {
        utility::LogError(
                "Either pass pcl_init with #points > number_of_points or "
                "init_factor > 1");
    }
    if (pcl_init != nullptr && pcl_init->points_.size() < number_of_points) {
        utility::LogError(
                "Either pass pcl_init with #points > number_of_points, or "
                "init_factor > 1");
    }

    // Compute area of each triangle and sum surface area
    std::vector<double> triangle_areas;
    double surface_area = GetSurfaceArea(triangle_areas);

    // Compute init points using uniform sampling
    std::shared_ptr<PointCloud> pcl;
    if (pcl_init == nullptr) {
        pcl = SamplePointsUniformlyImpl(size_t(init_factor * number_of_points),
                                        triangle_areas, surface_area,
                                        use_triangle_normal);
    } else {
        pcl = std::make_shared<PointCloud>();
        pcl->points_ = pcl_init->points_;
        pcl->normals_ = pcl_init->normals_;
        pcl->colors_ = pcl_init->colors_;
    }

    // Set-up sample elimination
    double alpha = 8;    // constant defined in paper
    double beta = 0.5;   // constant defined in paper
    double gamma = 1.5;  // constant defined in paper
    double ratio = double(number_of_points) / double(pcl->points_.size());
    double r_max = 2 * std::sqrt((surface_area / number_of_points) /
                                 (2 * std::sqrt(3.)));
    double r_min = r_max * beta * (1 - std::pow(ratio, gamma));

    std::vector<double> weights(pcl->points_.size());
    std::vector<bool> deleted(pcl->points_.size(), false);
    KDTreeFlann kdtree(*pcl);

    auto WeightFcn = [&](double d2) {
        double d = std::sqrt(d2);
        if (d < r_min) {
            d = r_min;
        }
        return std::pow(1 - d / r_max, alpha);
    };

    auto ComputePointWeight = [&](int pidx0) {
        std::vector<int> nbs;
        std::vector<double> dists2;
        kdtree.SearchRadius(pcl->points_[pidx0], r_max, nbs, dists2);
        double weight = 0;
        for (size_t nbidx = 0; nbidx < nbs.size(); ++nbidx) {
            int pidx1 = nbs[nbidx];
            // only count weights if not the same point if not deleted
            if (pidx0 == pidx1 || deleted[pidx1]) {
                continue;
            }
            weight += WeightFcn(dists2[nbidx]);
        }

        weights[pidx0] = weight;
    };

    // init weights and priority queue
    typedef std::tuple<int, double> QueueEntry;
    auto WeightCmp = [](const QueueEntry &a, const QueueEntry &b) {
        return std::get<1>(a) < std::get<1>(b);
    };
    std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                        decltype(WeightCmp)>
            queue(WeightCmp);
    for (size_t pidx0 = 0; pidx0 < pcl->points_.size(); ++pidx0) {
        ComputePointWeight(int(pidx0));
        queue.push(QueueEntry(int(pidx0), weights[pidx0]));
    };

    // sample elimination
    size_t current_number_of_points = pcl->points_.size();
    while (current_number_of_points > number_of_points) {
        int pidx;
        double weight;
        std::tie(pidx, weight) = queue.top();
        queue.pop();

        // test if the entry is up to date (because of reinsert)
        if (deleted[pidx] || weight != weights[pidx]) {
            continue;
        }

        // delete current sample
        deleted[pidx] = true;
        current_number_of_points--;

        // update weights
        std::vector<int> nbs;
        std::vector<double> dists2;
        kdtree.SearchRadius(pcl->points_[pidx], r_max, nbs, dists2);
        for (int nb : nbs) {
            ComputePointWeight(nb);
            queue.push(QueueEntry(nb, weights[nb]));
        }
    }

    // update pcl
    bool has_vert_normal = pcl->HasNormals();
    bool has_vert_color = pcl->HasColors();
    int next_free = 0;
    for (size_t idx = 0; idx < pcl->points_.size(); ++idx) {
        if (!deleted[idx]) {
            pcl->points_[next_free] = pcl->points_[idx];
            if (has_vert_normal) {
                pcl->normals_[next_free] = pcl->normals_[idx];
            }
            if (has_vert_color) {
                pcl->colors_[next_free] = pcl->colors_[idx];
            }
            next_free++;
        }
    }
    pcl->points_.resize(next_free);
    if (has_vert_normal) {
        pcl->normals_.resize(next_free);
    }
    if (has_vert_color) {
        pcl->colors_.resize(next_free);
    }

    return pcl;
}

TriangleMesh &TriangleMesh::RemoveDuplicatedVertices() {
    typedef std::tuple<double, double, double> Coordinate3;
    std::unordered_map<Coordinate3, size_t, utility::hash_tuple<Coordinate3>>
            point_to_old_index;
    std::vector<int> index_old_to_new(vertices_.size());
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    size_t old_vertex_num = vertices_.size();
    size_t k = 0;                                  // new index
    for (size_t i = 0; i < old_vertex_num; i++) {  // old index
        Coordinate3 coord = std::make_tuple(vertices_[i](0), vertices_[i](1),
                                            vertices_[i](2));
        if (point_to_old_index.find(coord) == point_to_old_index.end()) {
            point_to_old_index[coord] = i;
            vertices_[k] = vertices_[i];
            if (has_vert_normal) vertex_normals_[k] = vertex_normals_[i];
            if (has_vert_color) vertex_colors_[k] = vertex_colors_[i];
            index_old_to_new[i] = (int)k;
            k++;
        } else {
            index_old_to_new[i] = index_old_to_new[point_to_old_index[coord]];
        }
    }
    vertices_.resize(k);
    if (has_vert_normal) vertex_normals_.resize(k);
    if (has_vert_color) vertex_colors_.resize(k);
    if (k < old_vertex_num) {
        for (auto &triangle : triangles_) {
            triangle(0) = index_old_to_new[triangle(0)];
            triangle(1) = index_old_to_new[triangle(1)];
            triangle(2) = index_old_to_new[triangle(2)];
        }
        if (HasAdjacencyList()) {
            ComputeAdjacencyList();
        }
    }
    utility::LogDebug(
            "[RemoveDuplicatedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

TriangleMesh &TriangleMesh::RemoveDuplicatedTriangles() {
    if (HasTriangleUvs()) {
        utility::LogWarning(
                "[RemoveDuplicatedTriangles] This mesh contains triangle uvs "
                "that are not handled in this function");
    }
    typedef std::tuple<int, int, int> Index3;
    std::unordered_map<Index3, size_t, utility::hash_tuple<Index3>>
            triangle_to_old_index;
    bool has_tri_normal = HasTriangleNormals();
    size_t old_triangle_num = triangles_.size();
    size_t k = 0;
    for (size_t i = 0; i < old_triangle_num; i++) {
        Index3 index;
        // We first need to find the minimum index. Because triangle (0-1-2)
        // and triangle (2-0-1) are the same.
        if (triangles_[i](0) <= triangles_[i](1)) {
            if (triangles_[i](0) <= triangles_[i](2)) {
                index = std::make_tuple(triangles_[i](0), triangles_[i](1),
                                        triangles_[i](2));
            } else {
                index = std::make_tuple(triangles_[i](2), triangles_[i](0),
                                        triangles_[i](1));
            }
        } else {
            if (triangles_[i](1) <= triangles_[i](2)) {
                index = std::make_tuple(triangles_[i](1), triangles_[i](2),
                                        triangles_[i](0));
            } else {
                index = std::make_tuple(triangles_[i](2), triangles_[i](0),
                                        triangles_[i](1));
            }
        }
        if (triangle_to_old_index.find(index) == triangle_to_old_index.end()) {
            triangle_to_old_index[index] = i;
            triangles_[k] = triangles_[i];
            if (has_tri_normal) triangle_normals_[k] = triangle_normals_[i];
            k++;
        }
    }
    triangles_.resize(k);
    if (has_tri_normal) triangle_normals_.resize(k);
    if (k < old_triangle_num && HasAdjacencyList()) {
        ComputeAdjacencyList();
    }
    utility::LogDebug(
            "[RemoveDuplicatedTriangles] {:d} triangles have been removed.",
            (int)(old_triangle_num - k));

    return *this;
}

TriangleMesh &TriangleMesh::RemoveUnreferencedVertices() {
    std::vector<bool> vertex_has_reference(vertices_.size(), false);
    for (const auto &triangle : triangles_) {
        vertex_has_reference[triangle(0)] = true;
        vertex_has_reference[triangle(1)] = true;
        vertex_has_reference[triangle(2)] = true;
    }
    std::vector<int> index_old_to_new(vertices_.size());
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    size_t old_vertex_num = vertices_.size();
    size_t k = 0;                                  // new index
    for (size_t i = 0; i < old_vertex_num; i++) {  // old index
        if (vertex_has_reference[i]) {
            vertices_[k] = vertices_[i];
            if (has_vert_normal) vertex_normals_[k] = vertex_normals_[i];
            if (has_vert_color) vertex_colors_[k] = vertex_colors_[i];
            index_old_to_new[i] = (int)k;
            k++;
        } else {
            index_old_to_new[i] = -1;
        }
    }
    vertices_.resize(k);
    if (has_vert_normal) vertex_normals_.resize(k);
    if (has_vert_color) vertex_colors_.resize(k);
    if (k < old_vertex_num) {
        for (auto &triangle : triangles_) {
            triangle(0) = index_old_to_new[triangle(0)];
            triangle(1) = index_old_to_new[triangle(1)];
            triangle(2) = index_old_to_new[triangle(2)];
        }
        if (HasAdjacencyList()) {
            ComputeAdjacencyList();
        }
    }
    utility::LogDebug(
            "[RemoveUnreferencedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

TriangleMesh &TriangleMesh::RemoveDegenerateTriangles() {
    if (HasTriangleUvs()) {
        utility::LogWarning(
                "[RemoveDegenerateTriangles] This mesh contains triangle uvs "
                "that are not handled in this function");
    }
    bool has_tri_normal = HasTriangleNormals();
    size_t old_triangle_num = triangles_.size();
    size_t k = 0;
    for (size_t i = 0; i < old_triangle_num; i++) {
        const auto &triangle = triangles_[i];
        if (triangle(0) != triangle(1) && triangle(1) != triangle(2) &&
            triangle(2) != triangle(0)) {
            triangles_[k] = triangles_[i];
            if (has_tri_normal) triangle_normals_[k] = triangle_normals_[i];
            k++;
        }
    }
    triangles_.resize(k);
    if (has_tri_normal) triangle_normals_.resize(k);
    if (k < old_triangle_num && HasAdjacencyList()) {
        ComputeAdjacencyList();
    }
    utility::LogDebug(
            "[RemoveDegenerateTriangles] {:d} triangles have been "
            "removed.",
            (int)(old_triangle_num - k));
    return *this;
}

TriangleMesh &TriangleMesh::RemoveNonManifoldEdges() {
    if (HasTriangleUvs()) {
        utility::LogWarning(
                "[RemoveNonManifoldEdges] This mesh contains triangle uvs that "
                "are not handled in this function");
    }
    std::vector<double> triangle_areas;
    GetSurfaceArea(triangle_areas);

    bool mesh_is_edge_manifold = false;
    while (!mesh_is_edge_manifold) {
        mesh_is_edge_manifold = true;
        auto edges_to_triangles = GetEdgeToTrianglesMap();

        for (auto &kv : edges_to_triangles) {
            size_t n_edge_triangle_refs = kv.second.size();
            // check if the given edge is manifold
            // (has exactly 1, or 2 adjacent triangles)
            if (n_edge_triangle_refs == 1u || n_edge_triangle_refs == 2u) {
                continue;
            }

            // There is at least one edge that is non-manifold
            mesh_is_edge_manifold = false;

            // if the edge is non-manifold, then check if a referenced
            // triangle has already been removed
            // (triangle area has been set to < 0), otherwise remove triangle
            // with smallest surface area until number of adjacent triangles
            // is <= 2.
            // 1) count triangles that are not marked deleted
            int n_triangles = 0;
            for (int tidx : kv.second) {
                if (triangle_areas[tidx] > 0) {
                    n_triangles++;
                }
            }
            // 2) mark smallest triangles as deleted by setting
            // surface area to -1
            int n_triangles_to_delete = n_triangles - 2;
            while (n_triangles_to_delete > 0) {
                // find triangle with smallest area
                int min_tidx = -1;
                double min_area = std::numeric_limits<double>::max();
                for (int tidx : kv.second) {
                    double area = triangle_areas[tidx];
                    if (area > 0 && area < min_area) {
                        min_tidx = tidx;
                        min_area = area;
                    }
                }

                // mark triangle as deleted by setting area to -1
                triangle_areas[min_tidx] = -1;
                n_triangles_to_delete--;
            }
        }

        // delete marked triangles
        bool has_tri_normal = HasTriangleNormals();
        int to_tidx = 0;
        for (size_t from_tidx = 0; from_tidx < triangles_.size(); ++from_tidx) {
            if (triangle_areas[from_tidx] > 0) {
                triangles_[to_tidx] = triangles_[from_tidx];
                triangle_areas[to_tidx] = triangle_areas[from_tidx];
                if (has_tri_normal) {
                    triangle_normals_[to_tidx] = triangle_normals_[from_tidx];
                }
                to_tidx++;
            }
        }
        triangles_.resize(to_tidx);
        triangle_areas.resize(to_tidx);
        if (has_tri_normal) {
            triangle_normals_.resize(to_tidx);
        }
    }
    return *this;
}

TriangleMesh &TriangleMesh::MergeCloseVertices(double eps) {
    KDTreeFlann kdtree(*this);
    // precompute all neighbours
    utility::LogDebug("Precompute Neighbours");
    std::vector<std::vector<int>> nbs(vertices_.size());
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int idx = 0; idx < int(vertices_.size()); ++idx) {
        std::vector<double> dists2;
        kdtree.SearchRadius(vertices_[idx], eps, nbs[idx], dists2);
    }
    utility::LogDebug("Done Precompute Neighbours");

    bool has_vertex_normals = HasVertexNormals();
    bool has_vertex_colors = HasVertexColors();
    std::vector<Eigen::Vector3d> new_vertices;
    std::vector<Eigen::Vector3d> new_vertex_normals;
    std::vector<Eigen::Vector3d> new_vertex_colors;
    std::unordered_map<int, int> new_vert_mapping;
    for (int vidx = 0; vidx < int(vertices_.size()); ++vidx) {
        if (new_vert_mapping.count(vidx) > 0) {
            continue;
        }

        int new_vidx = int(new_vertices.size());
        new_vert_mapping[vidx] = new_vidx;

        Eigen::Vector3d vertex = vertices_[vidx];
        Eigen::Vector3d normal{0, 0, 0};
        if (has_vertex_normals) {
            normal = vertex_normals_[vidx];
        }
        Eigen::Vector3d color{0, 0, 0};
        if (has_vertex_colors) {
            color = vertex_colors_[vidx];
        }
        int n = 1;
        for (int nb : nbs[vidx]) {
            if (vidx == nb || new_vert_mapping.count(nb) > 0) {
                continue;
            }
            vertex += vertices_[nb];
            if (has_vertex_normals) {
                normal += vertex_normals_[nb];
            }
            if (has_vertex_colors) {
                color += vertex_colors_[nb];
            }
            new_vert_mapping[nb] = new_vidx;
            n += 1;
        }
        new_vertices.push_back(vertex / n);
        if (has_vertex_normals) {
            new_vertex_normals.push_back(normal / n);
        }
        if (has_vertex_colors) {
            new_vertex_colors.push_back(color / n);
        }
    }
    utility::LogDebug("Merged {} vertices",
                      vertices_.size() - new_vertices.size());

    std::swap(vertices_, new_vertices);
    std::swap(vertex_normals_, new_vertex_normals);
    std::swap(vertex_colors_, new_vertex_colors);

    for (auto &triangle : triangles_) {
        triangle(0) = new_vert_mapping[triangle(0)];
        triangle(1) = new_vert_mapping[triangle(1)];
        triangle(2) = new_vert_mapping[triangle(2)];
    }

    if (HasTriangleNormals()) {
        ComputeTriangleNormals();
    }

    return *this;
}

template <typename F>
bool OrientTriangleHelper(const std::vector<Eigen::Vector3i> &triangles,
                          F &swap) {
    std::unordered_map<Eigen::Vector2i, Eigen::Vector2i,
                       utility::hash_eigen<Eigen::Vector2i>>
            edge_to_orientation;
    std::unordered_set<int> unvisited_triangles;
    std::unordered_map<Eigen::Vector2i, std::unordered_set<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
            adjacent_triangles;
    std::queue<int> triangle_queue;

    auto VerifyAndAdd = [&](int vidx0, int vidx1) {
        Eigen::Vector2i key = TriangleMesh::GetOrderedEdge(vidx0, vidx1);
        if (edge_to_orientation.count(key) > 0) {
            if (edge_to_orientation.at(key)(0) == vidx0) {
                return false;
            }
        } else {
            edge_to_orientation[key] = Eigen::Vector2i(vidx0, vidx1);
        }
        return true;
    };
    auto AddTriangleNbsToQueue = [&](const Eigen::Vector2i &edge) {
        for (int nb_tidx : adjacent_triangles[edge]) {
            triangle_queue.push(nb_tidx);
        }
    };

    for (size_t tidx = 0; tidx < triangles.size(); ++tidx) {
        unvisited_triangles.insert(int(tidx));
        const auto &triangle = triangles[tidx];
        int vidx0 = triangle(0);
        int vidx1 = triangle(1);
        int vidx2 = triangle(2);
        adjacent_triangles[TriangleMesh::GetOrderedEdge(vidx0, vidx1)].insert(
                int(tidx));
        adjacent_triangles[TriangleMesh::GetOrderedEdge(vidx1, vidx2)].insert(
                int(tidx));
        adjacent_triangles[TriangleMesh::GetOrderedEdge(vidx2, vidx0)].insert(
                int(tidx));
    }

    while (!unvisited_triangles.empty()) {
        int tidx;
        if (triangle_queue.empty()) {
            tidx = *unvisited_triangles.begin();
        } else {
            tidx = triangle_queue.front();
            triangle_queue.pop();
        }
        if (unvisited_triangles.count(tidx) > 0) {
            unvisited_triangles.erase(tidx);
        } else {
            continue;
        }

        const auto &triangle = triangles[tidx];
        int vidx0 = triangle(0);
        int vidx1 = triangle(1);
        int vidx2 = triangle(2);
        Eigen::Vector2i key01 = TriangleMesh::GetOrderedEdge(vidx0, vidx1);
        Eigen::Vector2i key12 = TriangleMesh::GetOrderedEdge(vidx1, vidx2);
        Eigen::Vector2i key20 = TriangleMesh::GetOrderedEdge(vidx2, vidx0);
        bool exist01 = edge_to_orientation.count(key01) > 0;
        bool exist12 = edge_to_orientation.count(key12) > 0;
        bool exist20 = edge_to_orientation.count(key20) > 0;

        if (!(exist01 || exist12 || exist20)) {
            edge_to_orientation[key01] = Eigen::Vector2i(vidx0, vidx1);
            edge_to_orientation[key12] = Eigen::Vector2i(vidx1, vidx2);
            edge_to_orientation[key20] = Eigen::Vector2i(vidx2, vidx0);
        } else {
            // one flip is allowed
            if (exist01 && edge_to_orientation.at(key01)(0) == vidx0) {
                std::swap(vidx0, vidx1);
                swap(tidx, 0, 1);
            } else if (exist12 && edge_to_orientation.at(key12)(0) == vidx1) {
                std::swap(vidx1, vidx2);
                swap(tidx, 1, 2);
            } else if (exist20 && edge_to_orientation.at(key20)(0) == vidx2) {
                std::swap(vidx2, vidx0);
                swap(tidx, 2, 0);
            }

            // check if each edge looks in different direction compared to
            // existing ones if not existed, add the edge to map
            if (!VerifyAndAdd(vidx0, vidx1)) {
                return false;
            }
            if (!VerifyAndAdd(vidx1, vidx2)) {
                return false;
            }
            if (!VerifyAndAdd(vidx2, vidx0)) {
                return false;
            }
        }

        AddTriangleNbsToQueue(key01);
        AddTriangleNbsToQueue(key12);
        AddTriangleNbsToQueue(key20);
    }
    return true;
}

bool TriangleMesh::IsOrientable() const {
    auto NoOp = [](int, int, int) {};
    return OrientTriangleHelper(triangles_, NoOp);
}

bool TriangleMesh::IsWatertight() const {
    return IsEdgeManifold(false) && IsVertexManifold() && !IsSelfIntersecting();
}

bool TriangleMesh::OrientTriangles() {
    auto SwapTriangleOrder = [&](int tidx, int idx0, int idx1) {
        std::swap(triangles_[tidx](idx0), triangles_[tidx](idx1));
    };
    return OrientTriangleHelper(triangles_, SwapTriangleOrder);
}

std::unordered_map<Eigen::Vector2i,
                   std::vector<int>,
                   utility::hash_eigen<Eigen::Vector2i>>
TriangleMesh::GetEdgeToTrianglesMap() const {
    std::unordered_map<Eigen::Vector2i, std::vector<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
            trias_per_edge;
    auto AddEdge = [&](int vidx0, int vidx1, int tidx) {
        trias_per_edge[GetOrderedEdge(vidx0, vidx1)].push_back(tidx);
    };
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        const auto &triangle = triangles_[tidx];
        AddEdge(triangle(0), triangle(1), int(tidx));
        AddEdge(triangle(1), triangle(2), int(tidx));
        AddEdge(triangle(2), triangle(0), int(tidx));
    }
    return trias_per_edge;
}

std::unordered_map<Eigen::Vector2i,
                   std::vector<int>,
                   utility::hash_eigen<Eigen::Vector2i>>
TriangleMesh::GetEdgeToVerticesMap() const {
    std::unordered_map<Eigen::Vector2i, std::vector<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
            trias_per_edge;
    auto AddEdge = [&](int vidx0, int vidx1, int vidx2) {
        trias_per_edge[GetOrderedEdge(vidx0, vidx1)].push_back(vidx2);
    };
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        const auto &triangle = triangles_[tidx];
        AddEdge(triangle(0), triangle(1), triangle(2));
        AddEdge(triangle(1), triangle(2), triangle(0));
        AddEdge(triangle(2), triangle(0), triangle(1));
    }
    return trias_per_edge;
}

double TriangleMesh::ComputeTriangleArea(const Eigen::Vector3d &p0,
                                         const Eigen::Vector3d &p1,
                                         const Eigen::Vector3d &p2) {
    const Eigen::Vector3d x = p0 - p1;
    const Eigen::Vector3d y = p0 - p2;
    double area = 0.5 * x.cross(y).norm();
    return area;
}

double TriangleMesh::GetTriangleArea(size_t triangle_idx) const {
    const Eigen::Vector3i &triangle = triangles_[triangle_idx];
    const Eigen::Vector3d &vertex0 = vertices_[triangle(0)];
    const Eigen::Vector3d &vertex1 = vertices_[triangle(1)];
    const Eigen::Vector3d &vertex2 = vertices_[triangle(2)];
    return ComputeTriangleArea(vertex0, vertex1, vertex2);
}

double TriangleMesh::GetSurfaceArea() const {
    double surface_area = 0;
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        double triangle_area = GetTriangleArea(tidx);
        surface_area += triangle_area;
    }
    return surface_area;
}

double TriangleMesh::GetSurfaceArea(std::vector<double> &triangle_areas) const {
    double surface_area = 0;
    triangle_areas.resize(triangles_.size());
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        double triangle_area = GetTriangleArea(tidx);
        triangle_areas[tidx] = triangle_area;
        surface_area += triangle_area;
    }
    return surface_area;
}

double TriangleMesh::GetVolume() const {
    // Computes the signed volume of the tetrahedron defined by
    // the three triangle vertices and the origin. The sign is determined by
    // checking if the origin is at the same side as the normal with respect to
    // the triangle.
    auto GetSignedVolumeOfTriangle = [&](size_t tidx) {
        const Eigen::Vector3i &triangle = triangles_[tidx];
        const Eigen::Vector3d &vertex0 = vertices_[triangle(0)];
        const Eigen::Vector3d &vertex1 = vertices_[triangle(1)];
        const Eigen::Vector3d &vertex2 = vertices_[triangle(2)];
        return vertex0.dot(vertex1.cross(vertex2)) / 6.0;
    };

    if (!IsWatertight()) {
        utility::LogError(
                "The mesh is not watertight, and the volume cannot be "
                "computed.");
    }
    if (!IsOrientable()) {
        utility::LogError(
                "The mesh is not orientable, and the volume cannot be "
                "computed.");
    }

    double volume = 0;
    int64_t num_triangles = triangles_.size();
#pragma omp parallel for reduction(+ : volume) num_threads(utility::EstimateMaxThreads())
    for (int64_t tidx = 0; tidx < num_triangles; ++tidx) {
        volume += GetSignedVolumeOfTriangle(tidx);
    }
    return std::abs(volume);
}

Eigen::Vector4d TriangleMesh::ComputeTrianglePlane(const Eigen::Vector3d &p0,
                                                   const Eigen::Vector3d &p1,
                                                   const Eigen::Vector3d &p2) {
    const Eigen::Vector3d e0 = p1 - p0;
    const Eigen::Vector3d e1 = p2 - p0;
    Eigen::Vector3d abc = e0.cross(e1);
    double norm = abc.norm();
    // if the three points are co-linear, return invalid plane
    if (norm == 0) {
        return Eigen::Vector4d(0, 0, 0, 0);
    }
    abc /= abc.norm();
    double d = -abc.dot(p0);
    return Eigen::Vector4d(abc(0), abc(1), abc(2), d);
}

Eigen::Vector4d TriangleMesh::GetTrianglePlane(size_t triangle_idx) const {
    const Eigen::Vector3i &triangle = triangles_[triangle_idx];
    const Eigen::Vector3d &vertex0 = vertices_[triangle(0)];
    const Eigen::Vector3d &vertex1 = vertices_[triangle(1)];
    const Eigen::Vector3d &vertex2 = vertices_[triangle(2)];
    return ComputeTrianglePlane(vertex0, vertex1, vertex2);
}

int TriangleMesh::EulerPoincareCharacteristic() const {
    std::unordered_set<Eigen::Vector2i, utility::hash_eigen<Eigen::Vector2i>>
            edges;
    for (auto triangle : triangles_) {
        edges.emplace(GetOrderedEdge(triangle(0), triangle(1)));
        edges.emplace(GetOrderedEdge(triangle(0), triangle(2)));
        edges.emplace(GetOrderedEdge(triangle(1), triangle(2)));
    }

    int E = int(edges.size());
    int V = int(vertices_.size());
    int F = int(triangles_.size());
    return V + F - E;
}

std::vector<Eigen::Vector2i> TriangleMesh::GetNonManifoldEdges(
        bool allow_boundary_edges /* = true */) const {
    auto edges = GetEdgeToTrianglesMap();
    std::vector<Eigen::Vector2i> non_manifold_edges;
    for (auto &kv : edges) {
        if ((allow_boundary_edges &&
             (kv.second.size() < 1 || kv.second.size() > 2)) ||
            (!allow_boundary_edges && kv.second.size() != 2)) {
            non_manifold_edges.push_back(kv.first);
        }
    }
    return non_manifold_edges;
}

bool TriangleMesh::IsEdgeManifold(
        bool allow_boundary_edges /* = true */) const {
    auto edges = GetEdgeToTrianglesMap();
    for (auto &kv : edges) {
        if ((allow_boundary_edges &&
             (kv.second.size() < 1 || kv.second.size() > 2)) ||
            (!allow_boundary_edges && kv.second.size() != 2)) {
            return false;
        }
    }
    return true;
}

std::vector<int> TriangleMesh::GetNonManifoldVertices() const {
    std::vector<std::unordered_set<int>> vert_to_triangles(vertices_.size());
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        const auto &tria = triangles_[tidx];
        vert_to_triangles[tria(0)].emplace(int(tidx));
        vert_to_triangles[tria(1)].emplace(int(tidx));
        vert_to_triangles[tria(2)].emplace(int(tidx));
    }

    std::vector<int> non_manifold_verts;
    for (int vidx = 0; vidx < int(vertices_.size()); ++vidx) {
        const auto &triangles = vert_to_triangles[vidx];
        if (triangles.size() == 0) {
            continue;
        }

        // collect edges and vertices
        std::unordered_map<int, std::unordered_set<int>> edges;
        for (int tidx : triangles) {
            const auto &triangle = triangles_[tidx];
            if (triangle(0) != vidx && triangle(1) != vidx) {
                edges[triangle(0)].emplace(triangle(1));
                edges[triangle(1)].emplace(triangle(0));
            } else if (triangle(0) != vidx && triangle(2) != vidx) {
                edges[triangle(0)].emplace(triangle(2));
                edges[triangle(2)].emplace(triangle(0));
            } else if (triangle(1) != vidx && triangle(2) != vidx) {
                edges[triangle(1)].emplace(triangle(2));
                edges[triangle(2)].emplace(triangle(1));
            }
        }

        // test if vertices are connected
        std::queue<int> next;
        std::unordered_set<int> visited;
        next.push(edges.begin()->first);
        visited.emplace(edges.begin()->first);
        while (!next.empty()) {
            int vert = next.front();
            next.pop();

            for (auto nb : edges[vert]) {
                if (visited.count(nb) == 0) {
                    visited.emplace(nb);
                    next.emplace(nb);
                }
            }
        }
        if (visited.size() != edges.size()) {
            non_manifold_verts.push_back(vidx);
        }
    }

    return non_manifold_verts;
}

bool TriangleMesh::IsVertexManifold() const {
    return GetNonManifoldVertices().empty();
}

std::vector<Eigen::Vector2i> TriangleMesh::GetSelfIntersectingTriangles()
        const {
    std::vector<Eigen::Vector2i> self_intersecting_triangles;
    for (size_t tidx0 = 0; tidx0 < triangles_.size() - 1; ++tidx0) {
        const Eigen::Vector3i &tria_p = triangles_[tidx0];
        const Eigen::Vector3d &p0 = vertices_[tria_p(0)];
        const Eigen::Vector3d &p1 = vertices_[tria_p(1)];
        const Eigen::Vector3d &p2 = vertices_[tria_p(2)];

        const Eigen::Vector3d bb_min1 =
                p0.array().min(p1.array().min(p2.array()));
        const Eigen::Vector3d bb_max1 =
                p0.array().max(p1.array().max(p2.array()));

        for (size_t tidx1 = tidx0 + 1; tidx1 < triangles_.size(); ++tidx1) {
            const Eigen::Vector3i &tria_q = triangles_[tidx1];
            // check if neighbour triangle
            if (tria_p(0) == tria_q(0) || tria_p(0) == tria_q(1) ||
                tria_p(0) == tria_q(2) || tria_p(1) == tria_q(0) ||
                tria_p(1) == tria_q(1) || tria_p(1) == tria_q(2) ||
                tria_p(2) == tria_q(0) || tria_p(2) == tria_q(1) ||
                tria_p(2) == tria_q(2)) {
                continue;
            }

            // check for intersection
            const Eigen::Vector3d &q0 = vertices_[tria_q(0)];
            const Eigen::Vector3d &q1 = vertices_[tria_q(1)];
            const Eigen::Vector3d &q2 = vertices_[tria_q(2)];

            const Eigen::Vector3d bb_min2 =
                    q0.array().min(q1.array().min(q2.array()));
            const Eigen::Vector3d bb_max2 =
                    q0.array().max(q1.array().max(q2.array()));
            if (IntersectionTest::AABBAABB(bb_min1, bb_max1, bb_min2,
                                           bb_max2) &&
                IntersectionTest::TriangleTriangle3d(p0, p1, p2, q0, q1, q2)) {
                self_intersecting_triangles.push_back(
                        Eigen::Vector2i(tidx0, tidx1));
            }
        }
    }
    return self_intersecting_triangles;
}

bool TriangleMesh::IsSelfIntersecting() const {
    return !GetSelfIntersectingTriangles().empty();
}

bool TriangleMesh::IsBoundingBoxIntersecting(const TriangleMesh &other) const {
    return IntersectionTest::AABBAABB(GetMinBound(), GetMaxBound(),
                                      other.GetMinBound(), other.GetMaxBound());
}

bool TriangleMesh::IsIntersecting(const TriangleMesh &other) const {
    if (!IsBoundingBoxIntersecting(other)) {
        return false;
    }
    for (size_t tidx0 = 0; tidx0 < triangles_.size(); ++tidx0) {
        const Eigen::Vector3i &tria_p = triangles_[tidx0];
        const Eigen::Vector3d &p0 = vertices_[tria_p(0)];
        const Eigen::Vector3d &p1 = vertices_[tria_p(1)];
        const Eigen::Vector3d &p2 = vertices_[tria_p(2)];
        for (size_t tidx1 = 0; tidx1 < other.triangles_.size(); ++tidx1) {
            const Eigen::Vector3i &tria_q = other.triangles_[tidx1];
            const Eigen::Vector3d &q0 = other.vertices_[tria_q(0)];
            const Eigen::Vector3d &q1 = other.vertices_[tria_q(1)];
            const Eigen::Vector3d &q2 = other.vertices_[tria_q(2)];
            if (IntersectionTest::TriangleTriangle3d(p0, p1, p2, q0, q1, q2)) {
                return true;
            }
        }
    }
    return false;
}

std::tuple<std::vector<int>, std::vector<size_t>, std::vector<double>>
TriangleMesh::ClusterConnectedTriangles() const {
    std::vector<int> triangle_clusters(triangles_.size(), -1);
    std::vector<size_t> num_triangles;
    std::vector<double> areas;

    utility::LogDebug("[ClusterConnectedTriangles] Compute triangle adjacency");
    auto edges_to_triangles = GetEdgeToTrianglesMap();
    std::vector<std::unordered_set<int>> adjacency_list(triangles_.size());
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int tidx = 0; tidx < int(triangles_.size()); ++tidx) {
        const auto &triangle = triangles_[tidx];
        for (auto tnb :
             edges_to_triangles[GetOrderedEdge(triangle(0), triangle(1))]) {
            adjacency_list[tidx].insert(tnb);
        }
        for (auto tnb :
             edges_to_triangles[GetOrderedEdge(triangle(0), triangle(2))]) {
            adjacency_list[tidx].insert(tnb);
        }
        for (auto tnb :
             edges_to_triangles[GetOrderedEdge(triangle(1), triangle(2))]) {
            adjacency_list[tidx].insert(tnb);
        }
    }
    utility::LogDebug(
            "[ClusterConnectedTriangles] Done computing triangle adjacency");

    int cluster_idx = 0;
    for (int tidx = 0; tidx < int(triangles_.size()); ++tidx) {
        if (triangle_clusters[tidx] != -1) {
            continue;
        }

        std::queue<int> triangle_queue;
        int cluster_n_triangles = 0;
        double cluster_area = 0;

        triangle_queue.push(tidx);
        triangle_clusters[tidx] = cluster_idx;
        while (!triangle_queue.empty()) {
            int cluster_tidx = triangle_queue.front();
            triangle_queue.pop();

            cluster_n_triangles++;
            cluster_area += GetTriangleArea(cluster_tidx);

            for (auto tnb : adjacency_list[cluster_tidx]) {
                if (triangle_clusters[tnb] == -1) {
                    triangle_queue.push(tnb);
                    triangle_clusters[tnb] = cluster_idx;
                }
            }
        }

        num_triangles.push_back(cluster_n_triangles);
        areas.push_back(cluster_area);
        cluster_idx++;
    }

    utility::LogDebug(
            "[ClusterConnectedTriangles] Done clustering, #clusters={}",
            cluster_idx);
    return std::make_tuple(triangle_clusters, num_triangles, areas);
}

void TriangleMesh::RemoveTrianglesByIndex(
        const std::vector<size_t> &triangle_indices) {
    std::vector<bool> triangle_mask(triangles_.size(), false);
    for (auto tidx : triangle_indices) {
        if (tidx < triangles_.size()) {
            triangle_mask[tidx] = true;
        } else {
            utility::LogWarning(
                    "[RemoveTriangles] contains triangle index {} that is not "
                    "within the bounds",
                    tidx);
        }
    }

    RemoveTrianglesByMask(triangle_mask);
}

void TriangleMesh::RemoveTrianglesByMask(
        const std::vector<bool> &triangle_mask) {
    if (triangle_mask.size() != triangles_.size()) {
        utility::LogError("triangle_mask has a different size than triangles_");
    }

    bool has_tri_normal = HasTriangleNormals();
    bool has_tri_uvs = HasTriangleUvs();
    int to_tidx = 0;
    for (size_t from_tidx = 0; from_tidx < triangles_.size(); ++from_tidx) {
        if (!triangle_mask[from_tidx]) {
            triangles_[to_tidx] = triangles_[from_tidx];
            if (has_tri_normal) {
                triangle_normals_[to_tidx] = triangle_normals_[from_tidx];
            }
            if (has_tri_uvs) {
                triangle_uvs_[to_tidx * 3] = triangle_uvs_[from_tidx * 3];
                triangle_uvs_[to_tidx * 3 + 1] =
                        triangle_uvs_[from_tidx * 3 + 1];
                triangle_uvs_[to_tidx * 3 + 2] =
                        triangle_uvs_[from_tidx * 3 + 2];
            }
            to_tidx++;
        }
    }
    triangles_.resize(to_tidx);
    if (has_tri_normal) {
        triangle_normals_.resize(to_tidx);
    }
    if (has_tri_uvs) {
        triangle_uvs_.resize(to_tidx * 3);
    }
}

void TriangleMesh::RemoveVerticesByIndex(
        const std::vector<size_t> &vertex_indices) {
    std::vector<bool> vertex_mask(vertices_.size(), false);
    for (auto vidx : vertex_indices) {
        if (vidx < vertices_.size()) {
            vertex_mask[vidx] = true;
        } else {
            utility::LogWarning(
                    "[RemoveVerticessByIndex] contains vertex index {} that is "
                    "not within the bounds",
                    vidx);
        }
    }

    RemoveVerticesByMask(vertex_mask);
}

void TriangleMesh::RemoveVerticesByMask(const std::vector<bool> &vertex_mask) {
    if (vertex_mask.size() != vertices_.size()) {
        utility::LogError("vertex_mask has a different size than vertices_");
    }

    bool has_normal = HasVertexNormals();
    bool has_color = HasVertexColors();
    int to_vidx = 0;
    std::unordered_map<int, int> vertex_map;
    for (size_t from_vidx = 0; from_vidx < vertices_.size(); ++from_vidx) {
        if (!vertex_mask[from_vidx]) {
            vertex_map[static_cast<int>(from_vidx)] = static_cast<int>(to_vidx);
            vertices_[to_vidx] = vertices_[from_vidx];
            if (has_normal) {
                vertex_normals_[to_vidx] = vertex_normals_[from_vidx];
            }
            if (has_color) {
                vertex_colors_[to_vidx] = vertex_colors_[from_vidx];
            }
            to_vidx++;
        }
    }
    vertices_.resize(to_vidx);
    if (has_normal) {
        vertex_normals_.resize(to_vidx);
    }
    if (has_color) {
        vertex_colors_.resize(to_vidx);
    }

    std::vector<bool> triangle_mask(triangles_.size());
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        auto &tria = triangles_[tidx];
        triangle_mask[tidx] = vertex_mask[tria(0)] || vertex_mask[tria(1)] ||
                              vertex_mask[tria(2)];
        if (!triangle_mask[tidx]) {
            tria(0) = vertex_map[tria(0)];
            tria(1) = vertex_map[tria(1)];
            tria(2) = vertex_map[tria(2)];
        }
    }
    RemoveTrianglesByMask(triangle_mask);
}

std::shared_ptr<TriangleMesh> TriangleMesh::SelectByIndex(
        const std::vector<size_t> &indices, bool cleanup) const {
    if (HasTriangleUvs()) {
        utility::LogWarning(
                "[SelectByIndex] This mesh contains triangle uvs that are "
                "not handled in this function");
    }
    auto output = std::make_shared<TriangleMesh>();
    bool has_triangle_normals = HasTriangleNormals();
    bool has_vertex_normals = HasVertexNormals();
    bool has_vertex_colors = HasVertexColors();

    std::vector<int> new_vert_ind(vertices_.size(), -1);
    for (const auto &sel_vidx : indices) {
        if (sel_vidx >= vertices_.size()) {
            utility::LogWarning(
                    "[SelectByIndex] indices contains index {} out of range. "
                    "It is ignored.",
                    sel_vidx);
            continue;
        }
        if (new_vert_ind[sel_vidx] >= 0) {
            continue;
        }
        new_vert_ind[sel_vidx] = int(output->vertices_.size());
        output->vertices_.push_back(vertices_[sel_vidx]);
        if (has_vertex_colors) {
            output->vertex_colors_.push_back(vertex_colors_[sel_vidx]);
        }
        if (has_vertex_normals) {
            output->vertex_normals_.push_back(vertex_normals_[sel_vidx]);
        }
    }
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        int nvidx0 = new_vert_ind[triangles_[tidx](0)];
        int nvidx1 = new_vert_ind[triangles_[tidx](1)];
        int nvidx2 = new_vert_ind[triangles_[tidx](2)];
        if (nvidx0 >= 0 && nvidx1 >= 0 && nvidx2 >= 0) {
            output->triangles_.push_back(
                    Eigen::Vector3i(nvidx0, nvidx1, nvidx2));
            if (has_triangle_normals) {
                output->triangle_normals_.push_back(triangle_normals_[tidx]);
            }
        }
    }

    if (cleanup) {
        output->RemoveDuplicatedVertices();
        output->RemoveDuplicatedTriangles();
        output->RemoveUnreferencedVertices();
        output->RemoveDegenerateTriangles();
    }
    utility::LogDebug(
            "Triangle mesh sampled from {:d} vertices and {:d} triangles to "
            "{:d} vertices and {:d} triangles.",
            (int)vertices_.size(), (int)triangles_.size(),
            (int)output->vertices_.size(), (int)output->triangles_.size());
    return output;
}  // namespace geometry

std::shared_ptr<TriangleMesh> TriangleMesh::Crop(
        const AxisAlignedBoundingBox &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "AxisAlignedBoundingBox either has zeros "
                "size, or has wrong bounds.");
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(vertices_));
}

std::shared_ptr<TriangleMesh> TriangleMesh::Crop(
        const OrientedBoundingBox &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "AxisAlignedBoundingBox either has zeros "
                "size, or has wrong bounds.");
        return std::make_shared<TriangleMesh>();
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(vertices_));
}

std::unordered_map<Eigen::Vector2i,
                   double,
                   utility::hash_eigen<Eigen::Vector2i>>
TriangleMesh::ComputeEdgeWeightsCot(
        const std::unordered_map<Eigen::Vector2i,
                                 std::vector<int>,
                                 utility::hash_eigen<Eigen::Vector2i>>
                &edges_to_vertices,
        double min_weight) const {
    std::unordered_map<Eigen::Vector2i, double,
                       utility::hash_eigen<Eigen::Vector2i>>
            weights;
    for (const auto &edge_v2s : edges_to_vertices) {
        Eigen::Vector2i edge = edge_v2s.first;
        double weight_sum = 0;
        int N = 0;
        for (int v2 : edge_v2s.second) {
            Eigen::Vector3d a = vertices_[edge(0)] - vertices_[v2];
            Eigen::Vector3d b = vertices_[edge(1)] - vertices_[v2];
            double weight = a.dot(b) / (a.cross(b)).norm();
            weight_sum += weight;
            N++;
        }
        double weight = N > 0 ? weight_sum / N : 0;
        if (weight < min_weight) {
            weights[edge] = min_weight;
        } else {
            weights[edge] = weight;
        }
    }
    return weights;
}

}  // namespace geometry
}  // namespace open3d
