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

#include "open3d/pipelines/mesh_sampling/TriangleMeshSampling.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/utility/Console.h"

#include <queue>
#include <random>

namespace open3d {
namespace pipelines {
namespace mesh_sampling {

std::shared_ptr<geometry::PointCloud> SamplePointsUniformlyImpl(
        const geometry::TriangleMesh &mesh,
        size_t number_of_points,
        std::vector<double> &triangle_areas,
        double surface_area,
        bool use_triangle_normal,
        int seed) {
    // triangle areas to cdf
    triangle_areas[0] /= surface_area;
    for (size_t tidx = 1; tidx < mesh.triangles_.size(); ++tidx) {
        triangle_areas[tidx] =
                triangle_areas[tidx] / surface_area + triangle_areas[tidx - 1];
    }

    // sample point cloud
    bool has_vert_normal = mesh.HasVertexNormals();
    bool has_vert_color = mesh.HasVertexColors();
    if (seed == -1) {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 mt(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto pcd = std::make_shared<geometry::PointCloud>();
    pcd->points_.resize(number_of_points);
    if (has_vert_normal || use_triangle_normal) {
        pcd->normals_.resize(number_of_points);
    }
    if (use_triangle_normal && !mesh.HasTriangleNormals()) {
        utility::LogError(
                "[SamplePoints] defined use_triangle_normal=true, but mesh has "
                "no triangle normals. You can compute them using "
                "ComputeTriangleNormals(true)");
    }
    if (has_vert_color) {
        pcd->colors_.resize(number_of_points);
    }
    size_t point_idx = 0;
    for (size_t tidx = 0; tidx < mesh.triangles_.size(); ++tidx) {
        size_t n = size_t(std::round(triangle_areas[tidx] * number_of_points));
        while (point_idx < n) {
            double r1 = dist(mt);
            double r2 = dist(mt);
            double a = (1 - std::sqrt(r1));
            double b = std::sqrt(r1) * (1 - r2);
            double c = std::sqrt(r1) * r2;

            const Eigen::Vector3i &triangle = mesh.triangles_[tidx];
            pcd->points_[point_idx] = a * mesh.vertices_[triangle(0)] +
                                      b * mesh.vertices_[triangle(1)] +
                                      c * mesh.vertices_[triangle(2)];
            if (has_vert_normal && !use_triangle_normal) {
                pcd->normals_[point_idx] =
                        a * mesh.vertex_normals_[triangle(0)] +
                        b * mesh.vertex_normals_[triangle(1)] +
                        c * mesh.vertex_normals_[triangle(2)];
            }
            if (use_triangle_normal) {
                pcd->normals_[point_idx] = mesh.triangle_normals_[tidx];
            }
            if (has_vert_color) {
                pcd->colors_[point_idx] = a * mesh.vertex_colors_[triangle(0)] +
                                          b * mesh.vertex_colors_[triangle(1)] +
                                          c * mesh.vertex_colors_[triangle(2)];
            }

            point_idx++;
        }
    }

    return pcd;
}

std::shared_ptr<geometry::PointCloud> SamplePointsUniformly(
        const geometry::TriangleMesh &mesh,
        size_t number_of_points,
        bool use_triangle_normal,
        int seed) {
    if (number_of_points <= 0) {
        utility::LogError("[SamplePointsUniformly] number_of_points <= 0");
    }
    if (mesh.triangles_.size() == 0) {
        utility::LogError(
                "[SamplePointsUniformly] input mesh has no triangles");
    }

    // Compute area of each triangle and sum surface area
    std::vector<double> triangle_areas;
    double surface_area = mesh.GetSurfaceArea(triangle_areas);

    return SamplePointsUniformlyImpl(mesh, number_of_points, triangle_areas,
                                     surface_area, use_triangle_normal, seed);
}

std::shared_ptr<geometry::PointCloud> SamplePointsPoissonDisk(
        const geometry::TriangleMesh &mesh,
        size_t number_of_points,
        double init_factor,
        const std::shared_ptr<geometry::PointCloud> pcl_init,
        bool use_triangle_normal,
        int seed) {
    if (number_of_points <= 0) {
        utility::LogError("[SamplePointsPoissonDisk] number_of_points <= 0");
    }
    if (mesh.triangles_.size() == 0) {
        utility::LogError(
                "[SamplePointsPoissonDisk] input mesh has no triangles");
    }
    if (pcl_init == nullptr && init_factor < 1) {
        utility::LogError(
                "[SamplePointsPoissonDisk] either pass pcl_init with #points "
                "> number_of_points or init_factor > 1");
    }
    if (pcl_init != nullptr && pcl_init->points_.size() < number_of_points) {
        utility::LogError(
                "[SamplePointsPoissonDisk] either pass pcl_init with #points "
                "> number_of_points, or init_factor > 1");
    }

    // Compute area of each triangle and sum surface area
    std::vector<double> triangle_areas;
    double surface_area = mesh.GetSurfaceArea(triangle_areas);

    // Compute init points using uniform sampling
    std::shared_ptr<geometry::PointCloud> pcl;
    if (pcl_init == nullptr) {
        pcl = SamplePointsUniformlyImpl(
                mesh, size_t(init_factor * number_of_points), triangle_areas,
                surface_area, use_triangle_normal, seed);
    } else {
        pcl = std::make_shared<geometry::PointCloud>();
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
    geometry::KDTreeFlann kdtree(*pcl);

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

}  // namespace mesh_sampling
}  // namespace pipelines
}  // namespace open3d
