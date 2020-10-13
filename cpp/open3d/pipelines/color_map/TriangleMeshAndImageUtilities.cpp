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

#include "open3d/pipelines/color_map/TriangleMeshAndImageUtilities.h"

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/pipelines/color_map/ImageWarpingField.h"

namespace open3d {
namespace pipelines {
namespace color_map {

inline std::tuple<float, float, float> Project3DPointAndGetUVDepth(
        const Eigen::Vector3d X,
        const camera::PinholeCameraTrajectory& camera,
        int camid) {
    std::pair<double, double> f =
            camera.parameters_[camid].intrinsic_.GetFocalLength();
    std::pair<double, double> p =
            camera.parameters_[camid].intrinsic_.GetPrincipalPoint();
    Eigen::Vector4d Vt = camera.parameters_[camid].extrinsic_ *
                         Eigen::Vector4d(X(0), X(1), X(2), 1);
    float u = float((Vt(0) * f.first) / Vt(2) + p.first);
    float v = float((Vt(1) * f.second) / Vt(2) + p.second);
    float z = float(Vt(2));
    return std::make_tuple(u, v, z);
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
CreateVertexAndImageVisibility(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_depth,
        const std::vector<std::shared_ptr<geometry::Image>>& images_mask,
        const camera::PinholeCameraTrajectory& camera,
        double maximum_allowable_depth,
        double depth_threshold_for_visibility_check) {
    size_t n_camera = camera.parameters_.size();
    size_t n_vertex = mesh.vertices_.size();
    // visibility_image_to_vertex[c]: vertices visible by camera c.
    std::vector<std::vector<int>> visibility_image_to_vertex;
    visibility_image_to_vertex.resize(n_camera);
    // visibility_vertex_to_image[v]: cameras that can see vertex v.
    std::vector<std::vector<int>> visibility_vertex_to_image;
    visibility_vertex_to_image.resize(n_vertex);

#pragma omp parallel for schedule(static)
    for (int camera_id = 0; camera_id < int(n_camera); camera_id++) {
        for (int vertex_id = 0; vertex_id < int(n_vertex); vertex_id++) {
            Eigen::Vector3d X = mesh.vertices_[vertex_id];
            float u, v, d;
            std::tie(u, v, d) =
                    Project3DPointAndGetUVDepth(X, camera, camera_id);
            int u_d = int(round(u)), v_d = int(round(v));
            // Skip if vertex in image boundary.
            if (d < 0.0 ||
                !images_depth[camera_id]->TestImageBoundary(u_d, v_d)) {
                continue;
            }
            // Skip if vertex's depth is too large (e.g. background).
            float d_sensor =
                    *images_depth[camera_id]->PointerAt<float>(u_d, v_d);
            if (d_sensor > maximum_allowable_depth) {
                continue;
            }
            // Check depth boundary mask. If a vertex is located at the boundary
            // of an object, its color will be highly diverse from different
            // viewing angles.
            if (*images_mask[camera_id]->PointerAt<unsigned char>(u_d, v_d) ==
                255) {
                continue;
            }
            // Check depth errors.
            if (std::fabs(d - d_sensor) >=
                depth_threshold_for_visibility_check) {
                continue;
            }
            visibility_image_to_vertex[camera_id].push_back(vertex_id);
#pragma omp critical
            { visibility_vertex_to_image[vertex_id].push_back(camera_id); }
        }
    }

    for (int camera_id = 0; camera_id < int(n_camera); camera_id++) {
        size_t n_visible_vertex = visibility_image_to_vertex[camera_id].size();
        utility::LogDebug(
                "[cam {:d}]: {:d}/{:d} ({:.5f}%) vertices are visible",
                camera_id, n_visible_vertex, n_vertex,
                double(n_visible_vertex) / n_vertex * 100);
    }

    return std::make_tuple(visibility_vertex_to_image,
                           visibility_image_to_vertex);
}

template <typename T>
std::tuple<bool, T> QueryImageIntensity(
        const geometry::Image& img,
        const Eigen::Vector3d& V,
        const camera::PinholeCameraTrajectory& camera,
        int camid,
        int ch /*= -1*/,
        int image_boundary_margin /*= 10*/) {
    float u, v, depth;
    std::tie(u, v, depth) = Project3DPointAndGetUVDepth(V, camera, camid);
    if (img.TestImageBoundary(u, v, image_boundary_margin)) {
        int u_round = int(round(u));
        int v_round = int(round(v));
        if (ch == -1) {
            return std::make_tuple(true, *img.PointerAt<T>(u_round, v_round));
        } else {
            return std::make_tuple(true,
                                   *img.PointerAt<T>(u_round, v_round, ch));
        }
    } else {
        return std::make_tuple(false, 0);
    }
}

template <typename T>
std::tuple<bool, T> QueryImageIntensity(
        const geometry::Image& img,
        const ImageWarpingField& field,
        const Eigen::Vector3d& V,
        const camera::PinholeCameraTrajectory& camera,
        int camid,
        int ch /*= -1*/,
        int image_boundary_margin /*= 10*/) {
    float u, v, depth;
    std::tie(u, v, depth) = Project3DPointAndGetUVDepth(V, camera, camid);
    if (img.TestImageBoundary(u, v, image_boundary_margin)) {
        Eigen::Vector2d uv_shift = field.GetImageWarpingField(u, v);
        if (img.TestImageBoundary(uv_shift(0), uv_shift(1),
                                  image_boundary_margin)) {
            int u_shift = int(round(uv_shift(0)));
            int v_shift = int(round(uv_shift(1)));
            if (ch == -1) {
                return std::make_tuple(true,
                                       *img.PointerAt<T>(u_shift, v_shift));
            } else {
                return std::make_tuple(true,
                                       *img.PointerAt<T>(u_shift, v_shift, ch));
            }
        }
    }
    return std::make_tuple(false, 0);
}

void SetProxyIntensityForVertex(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_gray,
        const std::vector<ImageWarpingField>& warping_field,
        const camera::PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        std::vector<double>& proxy_intensity,
        int image_boundary_margin) {
    auto n_vertex = mesh.vertices_.size();
    proxy_intensity.resize(n_vertex);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < int(n_vertex); i++) {
        proxy_intensity[i] = 0.0;
        float sum = 0.0;
        for (size_t iter = 0; iter < visibility_vertex_to_image[i].size();
             iter++) {
            int j = visibility_vertex_to_image[i][iter];
            float gray;
            bool valid = false;
            std::tie(valid, gray) = QueryImageIntensity<float>(
                    *images_gray[j], warping_field[j], mesh.vertices_[i],
                    camera, j, -1, image_boundary_margin);
            if (valid) {
                sum += 1.0;
                proxy_intensity[i] += gray;
            }
        }
        if (sum > 0) {
            proxy_intensity[i] /= sum;
        }
    }
}

void SetProxyIntensityForVertex(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_gray,
        const camera::PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        std::vector<double>& proxy_intensity,
        int image_boundary_margin) {
    auto n_vertex = mesh.vertices_.size();
    proxy_intensity.resize(n_vertex);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < int(n_vertex); i++) {
        proxy_intensity[i] = 0.0;
        float sum = 0.0;
        for (size_t iter = 0; iter < visibility_vertex_to_image[i].size();
             iter++) {
            int j = visibility_vertex_to_image[i][iter];
            float gray;
            bool valid = false;
            std::tie(valid, gray) = QueryImageIntensity<float>(
                    *images_gray[j], mesh.vertices_[i], camera, j, -1,
                    image_boundary_margin);
            if (valid) {
                sum += 1.0;
                proxy_intensity[i] += gray;
            }
        }
        if (sum > 0) {
            proxy_intensity[i] /= sum;
        }
    }
}

void SetGeometryColorAverage(
        geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_color,
        const camera::PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        int image_boundary_margin /*= 10*/,
        int invisible_vertex_color_knn /*= 3*/) {
    size_t n_vertex = mesh.vertices_.size();
    mesh.vertex_colors_.clear();
    mesh.vertex_colors_.resize(n_vertex);
    std::vector<size_t> valid_vertices;
    std::vector<size_t> invalid_vertices;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)n_vertex; i++) {
        mesh.vertex_colors_[i] = Eigen::Vector3d::Zero();
        double sum = 0.0;
        for (size_t iter = 0; iter < visibility_vertex_to_image[i].size();
             iter++) {
            int j = visibility_vertex_to_image[i][iter];
            unsigned char r_temp, g_temp, b_temp;
            bool valid = false;
            std::tie(valid, r_temp) = QueryImageIntensity<unsigned char>(
                    *images_color[j], mesh.vertices_[i], camera, j, 0,
                    image_boundary_margin);
            std::tie(valid, g_temp) = QueryImageIntensity<unsigned char>(
                    *images_color[j], mesh.vertices_[i], camera, j, 1,
                    image_boundary_margin);
            std::tie(valid, b_temp) = QueryImageIntensity<unsigned char>(
                    *images_color[j], mesh.vertices_[i], camera, j, 2,
                    image_boundary_margin);
            float r = (float)r_temp / 255.0f;
            float g = (float)g_temp / 255.0f;
            float b = (float)b_temp / 255.0f;
            if (valid) {
                mesh.vertex_colors_[i] += Eigen::Vector3d(r, g, b);
                sum += 1.0;
            }
        }
#pragma omp critical
        {
            if (sum > 0.0) {
                mesh.vertex_colors_[i] /= sum;
                valid_vertices.push_back(i);
            } else {
                invalid_vertices.push_back(i);
            }
        }
    }
    if (invisible_vertex_color_knn > 0) {
        std::shared_ptr<geometry::TriangleMesh> valid_mesh =
                mesh.SelectByIndex(valid_vertices);
        geometry::KDTreeFlann kd_tree(*valid_mesh);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)invalid_vertices.size(); ++i) {
            size_t invalid_vertex = invalid_vertices[i];
            std::vector<int> indices;  // indices to valid_mesh
            std::vector<double> dists;
            kd_tree.SearchKNN(mesh.vertices_[invalid_vertex],
                              invisible_vertex_color_knn, indices, dists);
            Eigen::Vector3d new_color(0, 0, 0);
            for (const int& index : indices) {
                new_color += valid_mesh->vertex_colors_[index];
            }
            if (indices.size() > 0) {
                new_color /= static_cast<double>(indices.size());
            }
            mesh.vertex_colors_[invalid_vertex] = new_color;
        }
    }
}

void SetGeometryColorAverage(
        geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_color,
        const std::vector<ImageWarpingField>& warping_fields,
        const camera::PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        int image_boundary_margin /*= 10*/,
        int invisible_vertex_color_knn /*= 3*/) {
    size_t n_vertex = mesh.vertices_.size();
    mesh.vertex_colors_.clear();
    mesh.vertex_colors_.resize(n_vertex);
    std::vector<size_t> valid_vertices;
    std::vector<size_t> invalid_vertices;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)n_vertex; i++) {
        mesh.vertex_colors_[i] = Eigen::Vector3d::Zero();
        double sum = 0.0;
        for (size_t iter = 0; iter < visibility_vertex_to_image[i].size();
             iter++) {
            int j = visibility_vertex_to_image[i][iter];
            unsigned char r_temp, g_temp, b_temp;
            bool valid = false;
            std::tie(valid, r_temp) = QueryImageIntensity<unsigned char>(
                    *images_color[j], warping_fields[j], mesh.vertices_[i],
                    camera, j, 0, image_boundary_margin);
            std::tie(valid, g_temp) = QueryImageIntensity<unsigned char>(
                    *images_color[j], warping_fields[j], mesh.vertices_[i],
                    camera, j, 1, image_boundary_margin);
            std::tie(valid, b_temp) = QueryImageIntensity<unsigned char>(
                    *images_color[j], warping_fields[j], mesh.vertices_[i],
                    camera, j, 2, image_boundary_margin);
            float r = (float)r_temp / 255.0f;
            float g = (float)g_temp / 255.0f;
            float b = (float)b_temp / 255.0f;
            if (valid) {
                mesh.vertex_colors_[i] += Eigen::Vector3d(r, g, b);
                sum += 1.0;
            }
        }
#pragma omp critical
        {
            if (sum > 0.0) {
                mesh.vertex_colors_[i] /= sum;
                valid_vertices.push_back(i);
            } else {
                invalid_vertices.push_back(i);
            }
        }
    }
    if (invisible_vertex_color_knn > 0) {
        std::shared_ptr<geometry::TriangleMesh> valid_mesh =
                mesh.SelectByIndex(valid_vertices);
        geometry::KDTreeFlann kd_tree(*valid_mesh);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)invalid_vertices.size(); ++i) {
            size_t invalid_vertex = invalid_vertices[i];
            std::vector<int> indices;  // indices to valid_mesh
            std::vector<double> dists;
            kd_tree.SearchKNN(mesh.vertices_[invalid_vertex],
                              invisible_vertex_color_knn, indices, dists);
            Eigen::Vector3d new_color(0, 0, 0);
            for (const int& index : indices) {
                new_color += valid_mesh->vertex_colors_[index];
            }
            if (indices.size() > 0) {
                new_color /= static_cast<double>(indices.size());
            }
            mesh.vertex_colors_[invalid_vertex] = new_color;
        }
    }
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
