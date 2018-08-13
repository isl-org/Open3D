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

#include "PointCloud.h"
#include "TriangleMesh.h"

#include <unordered_map>

#include <Core/Utility/Helper.h>
#include <Core/Utility/Console.h>

namespace open3d{

namespace {

class AccumulatedPoint
{
public:
    AccumulatedPoint() :
            num_of_points_(0),
            point_(0.0, 0.0, 0.0),
            normal_(0.0, 0.0, 0.0),
            color_(0.0, 0.0, 0.0)
    {
    }

public:
    void AddPoint(const PointCloud &cloud, int index)
    {
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

    Eigen::Vector3d GetAveragePoint() const
    {
        return point_ / double(num_of_points_);
    }

    Eigen::Vector3d GetAverageNormal() const
    {
        return normal_.normalized();
    }

    Eigen::Vector3d GetAverageColor() const
    {
        return color_ / double(num_of_points_);
    }

private:
    int num_of_points_;
    Eigen::Vector3d point_;
    Eigen::Vector3d normal_;
    Eigen::Vector3d color_;
};

}    // unnamed namespace

std::shared_ptr<PointCloud> SelectDownSample(const PointCloud &input,
        const std::vector<size_t> &indices)
{
    auto output = std::make_shared<PointCloud>();
    bool has_normals = input.HasNormals();
    bool has_colors = input.HasColors();
    for (size_t i : indices) {
        output->points_.push_back(input.points_[i]);
        if (has_normals) output->normals_.push_back(input.normals_[i]);
        if (has_colors) output->colors_.push_back(input.colors_[i]);
    }
    PrintDebug("Pointcloud down sampled from %d points to %d points.\n",
            (int)input.points_.size(), (int)output->points_.size());
    return output;
}

std::shared_ptr<TriangleMesh> SelectDownSample(const TriangleMesh &input,
        const std::vector<size_t> &indices)
{
    auto output = std::make_shared<TriangleMesh>();
    bool has_triangle_normals = input.HasTriangleNormals();
    bool has_vertex_normals = input.HasVertexNormals();
    bool has_vertex_colors = input.HasVertexColors();
    // For each vertex, list face indices.
    std::vector<std::vector<int>> vertex_to_triangle_temp(input.vertices_.size());
    int triangle_id = 0;
    for (auto trangle : input.triangles_) {
        for (int i=0; i<3; i++)
            vertex_to_triangle_temp[trangle(i)].push_back(triangle_id);
        triangle_id++;
    }
    // Remove face indices of vertex_to_triangle_temp
    // if it does not correspond to selected vertices
    std::vector<std::vector<int>> vertex_to_triangle(input.vertices_.size());
    for (auto vertex_id : indices) {
        vertex_to_triangle[vertex_id] = vertex_to_triangle_temp[vertex_id];
    }
    // Make a triangle_to_vertex using vertex_to_triangle
    std::vector<std::vector<int>> triangle_to_vertex(input.triangles_.size());
    int vertex_id = 0;
    for (auto face_ids : vertex_to_triangle) {
        for (auto face_id : face_ids)
            triangle_to_vertex[face_id].push_back(vertex_id);
        vertex_id++;
    }
    // Only a face with three selected points contributes to mark mask_observed_vertex.
    std::vector<bool> mask_observed_vertex(input.vertices_.size());
    for (auto vertex_ids : triangle_to_vertex) {
        if ((int)vertex_ids.size() == 3)
            for (int i=0; i<3; i++)
                mask_observed_vertex[vertex_ids[i]] = true;
    }
    // Rename vertex id based on selected points
    std::vector<int> new_vertex_id(input.vertices_.size());
    for (auto i=0, cnt=0; i<mask_observed_vertex.size(); i++) {
        if (mask_observed_vertex[i])
            new_vertex_id[i] = cnt++;
    }
    // Push a triangle that has 3 selected vertices.
    triangle_id = 0;
    for (auto vertex_ids : triangle_to_vertex) {
        if ((int)vertex_ids.size() == 3) {
            Eigen::Vector3i new_face;
            for (int i=0; i<3; i++)
                new_face(i) = new_vertex_id[input.triangles_[triangle_id][i]];
            output->triangles_.push_back(new_face);
            if (has_triangle_normals) output->triangle_normals_.push_back(
                    input.triangle_normals_[triangle_id]);
        }
        triangle_id++;
    }
    // Push marked vertex.
    for (auto i=0; i<mask_observed_vertex.size(); i++) {
        if (mask_observed_vertex[i]) {
            output->vertices_.push_back(input.vertices_[i]);
            if (has_vertex_normals) output->vertex_normals_.push_back(
                    input.vertex_normals_[i]);
            if (has_vertex_colors) output->vertex_colors_.push_back(
                    input.vertex_colors_[i]);
        }
    }
    output->Purge();
    PrintDebug("Triangle mesh sampled from %d vertices and %d triangles to %d vertices and %d triangles.\n",
            (int)input.vertices_.size(), (int)input.triangles_.size(),
            (int)output->vertices_.size(), (int)output->triangles_.size());
    return output;
}

std::shared_ptr<PointCloud> VoxelDownSample(const PointCloud &input,
        double voxel_size)
{
    auto output = std::make_shared<PointCloud>();
    if (voxel_size <= 0.0) {
        PrintDebug("[VoxelDownSample] voxel_size <= 0.\n");
        return output;
    }
    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d voxel_min_bound = input.GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d voxel_max_bound = input.GetMaxBound() + voxel_size3 * 0.5;
    if (voxel_size * std::numeric_limits<int>::max() <
            (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        PrintDebug("[VoxelDownSample] voxel_size is too small.\n");
        return output;
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPoint,
            hash_eigen::hash<Eigen::Vector3i>> voxelindex_to_accpoint;
    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    for (int i = 0; i < (int)input.points_.size(); i++) {
        ref_coord = (input.points_[i] - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))),
                int(floor(ref_coord(1))), int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(input, i);
    }
    bool has_normals = input.HasNormals();
    bool has_colors = input.HasColors();
    for (auto accpoint : voxelindex_to_accpoint) {
        output->points_.push_back(accpoint.second.GetAveragePoint());
        if (has_normals) {
            output->normals_.push_back(accpoint.second.GetAverageNormal());
        }
        if (has_colors) {
            output->colors_.push_back(accpoint.second.GetAverageColor());
        }
    }
    PrintDebug("Pointcloud down sampled from %d points to %d points.\n",
            (int)input.points_.size(), (int)output->points_.size());
    return output;
}

std::shared_ptr<PointCloud> UniformDownSample(const PointCloud &input,
        size_t every_k_points)
{
    if (every_k_points == 0) {
        PrintDebug("[UniformDownSample] Illegal sample rate.\n");
        return std::make_shared<PointCloud>();
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < input.points_.size(); i += every_k_points) {
        indices.push_back(i);
    }
    return SelectDownSample(input, indices);
}

std::shared_ptr<PointCloud> CropPointCloud(const PointCloud &input,
        const Eigen::Vector3d &min_bound, const Eigen::Vector3d &max_bound)
{
    if (min_bound(0) > max_bound(0) || min_bound(1) > max_bound(1) ||
            min_bound(2) > max_bound(2)) {
        PrintDebug("[CropPointCloud] Illegal boundary clipped all points.\n");
        return std::make_shared<PointCloud>();
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < input.points_.size(); i++) {
        const auto &point = input.points_[i];
        if (point(0) >= min_bound(0) && point(0) <= max_bound(0) &&
                point(1) >= min_bound(1) && point(1) <= max_bound(1) &&
                point(2) >= min_bound(2) && point(2) <= max_bound(2)) {
            indices.push_back(i);
        }
    }
    return SelectDownSample(input, indices);
}

std::shared_ptr<TriangleMesh> CropTriangleMesh(const TriangleMesh &input,
        const Eigen::Vector3d &min_bound, const Eigen::Vector3d &max_bound)
{
    if (min_bound(0) > max_bound(0) || min_bound(1) > max_bound(1) ||
            min_bound(2) > max_bound(2)) {
        PrintDebug("[CropTriangleMesh] Illegal boundary clipped all points.\n");
        return std::make_shared<TriangleMesh>();
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < input.vertices_.size(); i++) {
        const auto &point = input.vertices_[i];
        if (point(0) >= min_bound(0) && point(0) <= max_bound(0) &&
                point(1) >= min_bound(1) && point(1) <= max_bound(1) &&
                point(2) >= min_bound(2) && point(2) <= max_bound(2)) {
            indices.push_back(i);
        }
    }
    return SelectDownSample(input, indices);
}

}    // namespace open3d
