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

#include <numeric>
#include <unordered_map>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {

namespace {

using namespace geometry;

class AccumulatedPoint {
public:
    AccumulatedPoint()
        : num_of_points_(0),
          point_(0.0, 0.0, 0.0),
          normal_(0.0, 0.0, 0.0),
          color_(0.0, 0.0, 0.0) {}

public:
    void AddPoint(const PointCloud &cloud, int index) {
        point_ += cloud.points_[index];
        if (cloud.HasNormals()) {
            if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
                normal_ += cloud.normals_[index];
            }
        }
        if (cloud.HasColors()) {
            color_ += cloud.colors_[index];
        }
        num_of_points_++;
    }

    Eigen::Vector3d GetAveragePoint() const {
        return point_ / double(num_of_points_);
    }

    Eigen::Vector3d GetAverageNormal() const { return normal_.normalized(); }

    Eigen::Vector3d GetAverageColor() const {
        return color_ / double(num_of_points_);
    }

public:
    int num_of_points_;
    Eigen::Vector3d point_;
    Eigen::Vector3d normal_;
    Eigen::Vector3d color_;
};

class point_cubic_id {
public:
    size_t point_id;
    int cubic_id;
};

class AccumulatedPointForTrace : public AccumulatedPoint {
public:
    void AddPoint(const PointCloud &cloud,
                  size_t index,
                  int cubic_index,
                  bool approximate_class) {
        point_ += cloud.points_[index];
        if (cloud.HasNormals()) {
            if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
                normal_ += cloud.normals_[index];
            }
        }
        if (cloud.HasColors()) {
            if (approximate_class) {
                auto got = classes.find(int(cloud.colors_[index][0]));
                if (got == classes.end())
                    classes[int(cloud.colors_[index][0])] = 1;
                else
                    classes[int(cloud.colors_[index][0])] += 1;
            } else {
                color_ += cloud.colors_[index];
            }
        }
        point_cubic_id new_id;
        new_id.point_id = index;
        new_id.cubic_id = cubic_index;
        original_id.push_back(new_id);
        num_of_points_++;
    }

    Eigen::Vector3d GetMaxClass() {
        int max_class = -1;
        int max_count = -1;
        for (auto it = classes.begin(); it != classes.end(); it++) {
            if (it->second > max_count) {
                max_count = it->second;
                max_class = it->first;
            }
        }
        return Eigen::Vector3d(max_class, max_class, max_class);
    }

    std::vector<point_cubic_id> GetOriginalID() { return original_id; }

private:
    // original point cloud id in higher resolution + its cubic id
    std::vector<point_cubic_id> original_id;
    std::unordered_map<int, int> classes;
};

}  // unnamed namespace

namespace geometry {
std::shared_ptr<PointCloud> PointCloud::SelectDownSample(
        const std::vector<size_t> &indices, bool invert /* = false */) const {
    auto output = std::make_shared<PointCloud>();
    bool has_normals = HasNormals();
    bool has_colors = HasColors();

    std::vector<bool> mask = std::vector<bool>(points_.size(), invert);
    for (size_t i : indices) {
        mask[i] = !invert;
    }

    for (size_t i = 0; i < points_.size(); i++) {
        if (mask[i]) {
            output->points_.push_back(points_[i]);
            if (has_normals) output->normals_.push_back(normals_[i]);
            if (has_colors) output->colors_.push_back(colors_[i]);
        }
    }
    utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.\n",
            (int)points_.size(), (int)output->points_.size());
    return output;
}

std::shared_ptr<TriangleMesh> TriangleMesh::SelectDownSample(
        const std::vector<size_t> &indices) const {
    auto output = std::make_shared<TriangleMesh>();
    bool has_triangle_normals = HasTriangleNormals();
    bool has_vertex_normals = HasVertexNormals();
    bool has_vertex_colors = HasVertexColors();
    // For each vertex, list face indices.
    std::vector<std::vector<int>> vertex_to_triangle_temp(vertices_.size());
    int triangle_id = 0;
    for (auto trangle : triangles_) {
        for (int i = 0; i < 3; i++)
            vertex_to_triangle_temp[trangle(i)].push_back(triangle_id);
        triangle_id++;
    }
    // Remove face indices of vertex_to_triangle_temp
    // if it does not correspond to selected vertices
    std::vector<std::vector<int>> vertex_to_triangle(vertices_.size());
    for (auto vertex_id : indices) {
        vertex_to_triangle[vertex_id] = vertex_to_triangle_temp[vertex_id];
    }
    // Make a triangle_to_vertex using vertex_to_triangle
    std::vector<std::vector<int>> triangle_to_vertex(triangles_.size());
    int vertex_id = 0;
    for (auto face_ids : vertex_to_triangle) {
        for (auto face_id : face_ids)
            triangle_to_vertex[face_id].push_back(vertex_id);
        vertex_id++;
    }
    // Only a face with three selected points contributes to mark
    // mask_observed_vertex.
    std::vector<bool> mask_observed_vertex(vertices_.size());
    for (auto vertex_ids : triangle_to_vertex) {
        if ((int)vertex_ids.size() == 3)
            for (int i = 0; i < 3; i++)
                mask_observed_vertex[vertex_ids[i]] = true;
    }
    // Rename vertex id based on selected points
    std::vector<int> new_vertex_id(vertices_.size());
    for (size_t i = 0, cnt = 0; i < mask_observed_vertex.size(); i++) {
        if (mask_observed_vertex[i]) {
            new_vertex_id[i] = int(cnt);
            cnt++;
        }
    }
    // Push a triangle that has 3 selected vertices.
    triangle_id = 0;
    for (auto vertex_ids : triangle_to_vertex) {
        if ((int)vertex_ids.size() == 3) {
            Eigen::Vector3i new_face;
            for (int i = 0; i < 3; i++)
                new_face(i) = new_vertex_id[triangles_[triangle_id][i]];
            output->triangles_.push_back(new_face);
            if (has_triangle_normals)
                output->triangle_normals_.push_back(
                        triangle_normals_[triangle_id]);
        }
        triangle_id++;
    }
    // Push marked vertex.
    for (size_t i = 0; i < mask_observed_vertex.size(); i++) {
        if (mask_observed_vertex[i]) {
            output->vertices_.push_back(vertices_[i]);
            if (has_vertex_normals)
                output->vertex_normals_.push_back(vertex_normals_[i]);
            if (has_vertex_colors)
                output->vertex_colors_.push_back(vertex_colors_[i]);
        }
    }
    output->RemoveDuplicatedVertices();
    output->RemoveDuplicatedTriangles();
    output->RemoveUnreferencedVertices();
    output->RemoveDegenerateTriangles();
    utility::LogDebug(
            "Triangle mesh sampled from {:d} vertices and {:d} triangles to "
            "{:d} "
            "vertices and {:d} triangles.\n",
            (int)vertices_.size(), (int)triangles_.size(),
            (int)output->vertices_.size(), (int)output->triangles_.size());
    return output;
}

std::shared_ptr<PointCloud> PointCloud::VoxelDownSample(
        double voxel_size) const {
    auto output = std::make_shared<PointCloud>();
    if (voxel_size <= 0.0) {
        utility::LogWarning("[VoxelDownSample] voxel_size <= 0.\n");
        return output;
    }
    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d voxel_min_bound = GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d voxel_max_bound = GetMaxBound() + voxel_size3 * 0.5;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::LogWarning("[VoxelDownSample] voxel_size is too small.\n");
        return output;
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPoint,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            voxelindex_to_accpoint;

    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    for (int i = 0; i < (int)points_.size(); i++) {
        ref_coord = (points_[i] - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(*this, i);
    }
    bool has_normals = HasNormals();
    bool has_colors = HasColors();
    for (auto accpoint : voxelindex_to_accpoint) {
        output->points_.push_back(accpoint.second.GetAveragePoint());
        if (has_normals) {
            output->normals_.push_back(accpoint.second.GetAverageNormal());
        }
        if (has_colors) {
            output->colors_.push_back(accpoint.second.GetAverageColor());
        }
    }
    utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.\n",
            (int)points_.size(), (int)output->points_.size());
    return output;
}

std::tuple<std::shared_ptr<PointCloud>, Eigen::MatrixXi>
PointCloud::VoxelDownSampleAndTrace(double voxel_size,
                                    const Eigen::Vector3d &min_bound,
                                    const Eigen::Vector3d &max_bound,
                                    bool approximate_class) const {
    auto output = std::make_shared<PointCloud>();
    Eigen::MatrixXi cubic_id;
    if (voxel_size <= 0.0) {
        utility::LogWarning("[VoxelDownSample] voxel_size <= 0.\n");
        return std::make_tuple(output, cubic_id);
    }
    // Note: this is different from VoxelDownSample.
    // It is for fixing coordinate for multiscale voxel space
    auto voxel_min_bound = min_bound;
    auto voxel_max_bound = max_bound;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::LogWarning("[VoxelDownSample] voxel_size is too small.\n");
        return std::make_tuple(output, cubic_id);
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPointForTrace,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    int cid_temp[3] = {1, 2, 4};
    for (size_t i = 0; i < points_.size(); i++) {
        auto ref_coord = (points_[i] - voxel_min_bound) / voxel_size;
        auto voxel_index = Eigen::Vector3i(int(floor(ref_coord(0))),
                                           int(floor(ref_coord(1))),
                                           int(floor(ref_coord(2))));
        int cid = 0;
        for (int c = 0; c < 3; c++) {
            if ((ref_coord(c) - voxel_index(c)) >= 0.5) {
                cid += cid_temp[c];
            }
        }
        voxelindex_to_accpoint[voxel_index].AddPoint(*this, i, cid,
                                                     approximate_class);
    }
    bool has_normals = HasNormals();
    bool has_colors = HasColors();
    int cnt = 0;
    cubic_id.resize(voxelindex_to_accpoint.size(), 8);
    cubic_id.setConstant(-1);
    for (auto accpoint : voxelindex_to_accpoint) {
        output->points_.push_back(accpoint.second.GetAveragePoint());
        if (has_normals) {
            output->normals_.push_back(accpoint.second.GetAverageNormal());
        }
        if (has_colors) {
            if (approximate_class) {
                output->colors_.push_back(accpoint.second.GetMaxClass());
            } else {
                output->colors_.push_back(accpoint.second.GetAverageColor());
            }
        }
        auto original_id = accpoint.second.GetOriginalID();
        for (int i = 0; i < (int)original_id.size(); i++) {
            size_t pid = original_id[i].point_id;
            int cid = original_id[i].cubic_id;
            cubic_id(cnt, cid) = int(pid);
        }
        cnt++;
    }
    utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.\n",
            (int)points_.size(), (int)output->points_.size());
    return std::make_tuple(output, cubic_id);
}

std::shared_ptr<PointCloud> PointCloud::UniformDownSample(
        size_t every_k_points) const {
    if (every_k_points == 0) {
        utility::LogWarning("[UniformDownSample] Illegal sample rate.\n");
        return std::make_shared<PointCloud>();
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < points_.size(); i += every_k_points) {
        indices.push_back(i);
    }
    return SelectDownSample(indices);
}

std::shared_ptr<PointCloud> PointCloud::Crop(
        const Eigen::Vector3d &min_bound,
        const Eigen::Vector3d &max_bound) const {
    if (min_bound(0) > max_bound(0) || min_bound(1) > max_bound(1) ||
        min_bound(2) > max_bound(2)) {
        utility::LogWarning(
                "[CropPointCloud] Illegal boundary clipped all points.\n");
        return std::make_shared<PointCloud>();
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < points_.size(); i++) {
        const auto &point = points_[i];
        if (point(0) >= min_bound(0) && point(0) <= max_bound(0) &&
            point(1) >= min_bound(1) && point(1) <= max_bound(1) &&
            point(2) >= min_bound(2) && point(2) <= max_bound(2)) {
            indices.push_back(i);
        }
    }
    return SelectDownSample(indices);
}

std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
PointCloud::RemoveRadiusOutliers(size_t nb_points, double search_radius) const {
    if (nb_points < 1 || search_radius <= 0) {
        utility::LogWarning(
                "[RemoveRadiusOutliers] Illegal input parameters,"
                "number of points and radius must be positive\n");
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
    std::vector<bool> mask = std::vector<bool>(points_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < int(points_.size()); i++) {
        std::vector<int> tmp_indices;
        std::vector<double> dist;
        size_t nb_neighbors = kdtree.SearchRadius(points_[i], search_radius,
                                                  tmp_indices, dist);
        mask[i] = (nb_neighbors > nb_points);
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i]) {
            indices.push_back(i);
        }
    }
    return std::make_tuple(SelectDownSample(indices), indices);
}

std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
PointCloud::RemoveStatisticalOutliers(size_t nb_neighbors,
                                      double std_ratio) const {
    if (nb_neighbors < 1 || std_ratio <= 0) {
        utility::LogWarning(
                "[RemoveStatisticalOutliers] Illegal input parameters, number "
                "of neighbors"
                "and standard deviation ratio must be positive\n");
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    if (points_.size() == 0) {
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
    std::vector<double> avg_distances = std::vector<double>(points_.size());
    std::vector<size_t> indices;
    size_t valid_distances = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < int(points_.size()); i++) {
        std::vector<int> tmp_indices;
        std::vector<double> dist;
        kdtree.SearchKNN(points_[i], int(nb_neighbors), tmp_indices, dist);
        double mean = -1.0;
        if (dist.size() > 0u) {
            valid_distances++;
            std::for_each(dist.begin(), dist.end(),
                          [](double &d) { d = std::sqrt(d); });
            mean = std::accumulate(dist.begin(), dist.end(), 0.0) / dist.size();
        }
        avg_distances[i] = mean;
    }
    if (valid_distances == 0) {
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    double cloud_mean = std::accumulate(
            avg_distances.begin(), avg_distances.end(), 0.0,
            [](double const &x, double const &y) { return y > 0 ? x + y : x; });
    cloud_mean /= valid_distances;
    double sq_sum = std::inner_product(
            avg_distances.begin(), avg_distances.end(), avg_distances.begin(),
            0.0, [](double const &x, double const &y) { return x + y; },
            [cloud_mean](double const &x, double const &y) {
                return x > 0 ? (x - cloud_mean) * (y - cloud_mean) : 0;
            });
    // Bessel's correction
    double std_dev = std::sqrt(sq_sum / (valid_distances - 1));
    double distance_threshold = cloud_mean + std_ratio * std_dev;
    for (size_t i = 0; i < avg_distances.size(); i++) {
        if (avg_distances[i] > 0 && avg_distances[i] < distance_threshold) {
            indices.push_back(i);
        }
    }
    return std::make_tuple(SelectDownSample(indices), indices);
}

std::shared_ptr<TriangleMesh> TriangleMesh::Crop(
        const Eigen::Vector3d &min_bound,
        const Eigen::Vector3d &max_bound) const {
    if (min_bound(0) > max_bound(0) || min_bound(1) > max_bound(1) ||
        min_bound(2) > max_bound(2)) {
        utility::LogWarning(
                "[CropTriangleMesh] Illegal boundary clipped all points.\n");
        return std::make_shared<TriangleMesh>();
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < vertices_.size(); i++) {
        const auto &point = vertices_[i];
        if (point(0) >= min_bound(0) && point(0) <= max_bound(0) &&
            point(1) >= min_bound(1) && point(1) <= max_bound(1) &&
            point(2) >= min_bound(2) && point(2) <= max_bound(2)) {
            indices.push_back(i);
        }
    }
    return SelectDownSample(indices);
}
}  // namespace geometry
}  // namespace open3d
