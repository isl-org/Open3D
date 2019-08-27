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

#include "Open3D/ColorMap/TriangleMeshAndImageUtilities.h"

#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "Open3D/ColorMap/ImageWarpingField.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d {
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
        double depth_threshold_for_visiblity_check) {
    auto n_camera = camera.parameters_.size();
    auto n_vertex = mesh.vertices_.size();
    std::vector<std::vector<int>> visiblity_vertex_to_image;
    std::vector<std::vector<int>> visiblity_image_to_vertex;
    visiblity_vertex_to_image.resize(n_vertex);
    visiblity_image_to_vertex.resize(n_camera);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < int(n_camera); c++) {
        int viscnt = 0;
        for (size_t vertex_id = 0; vertex_id < n_vertex; vertex_id++) {
            Eigen::Vector3d X = mesh.vertices_[vertex_id];
            float u, v, d;
            std::tie(u, v, d) = Project3DPointAndGetUVDepth(X, camera, c);
            int u_d = int(round(u)), v_d = int(round(v));
            if (d < 0.0 || !images_depth[c]->TestImageBoundary(u_d, v_d))
                continue;
            float d_sensor = *images_depth[c]->PointerAt<float>(u_d, v_d);
            if (d_sensor > maximum_allowable_depth) continue;
            if (*images_mask[c]->PointerAt<unsigned char>(u_d, v_d) == 255)
                continue;
            if (std::fabs(d - d_sensor) < depth_threshold_for_visiblity_check) {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    visiblity_vertex_to_image[vertex_id].push_back(c);
                    visiblity_image_to_vertex[c].push_back(int(vertex_id));
                    viscnt++;
                }
            }
        }
        utility::LogDebug("[cam {:d}] {:.5f} percents are visible\n", c,
                          double(viscnt) / n_vertex * 100);
        fflush(stdout);
    }
    return std::make_tuple(visiblity_vertex_to_image,
                           visiblity_image_to_vertex);
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
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity,
        int image_boundary_margin) {
    auto n_vertex = mesh.vertices_.size();
    proxy_intensity.resize(n_vertex);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < int(n_vertex); i++) {
        proxy_intensity[i] = 0.0;
        float sum = 0.0;
        for (size_t iter = 0; iter < visiblity_vertex_to_image[i].size();
             iter++) {
            int j = visiblity_vertex_to_image[i][iter];
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
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity,
        int image_boundary_margin) {
    auto n_vertex = mesh.vertices_.size();
    proxy_intensity.resize(n_vertex);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < int(n_vertex); i++) {
        proxy_intensity[i] = 0.0;
        float sum = 0.0;
        for (size_t iter = 0; iter < visiblity_vertex_to_image[i].size();
             iter++) {
            int j = visiblity_vertex_to_image[i][iter];
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
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        int image_boundary_margin /*= 10*/,
        int invisible_vertex_color_knn /*= 3*/) {
    size_t n_vertex = mesh.vertices_.size();
    mesh.vertex_colors_.clear();
    mesh.vertex_colors_.resize(n_vertex);
    std::vector<size_t> valid_vertices;
    std::vector<size_t> invalid_vertices;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)n_vertex; i++) {
        mesh.vertex_colors_[i] = Eigen::Vector3d::Zero();
        double sum = 0.0;
        for (size_t iter = 0; iter < visiblity_vertex_to_image[i].size();
             iter++) {
            int j = visiblity_vertex_to_image[i][iter];
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
#ifdef _OPENMP
#pragma omp critical
#endif
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
                mesh.SelectDownSample(valid_vertices);
        geometry::KDTreeFlann kd_tree(*valid_mesh);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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
                new_color /= indices.size();
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
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        int image_boundary_margin /*= 10*/,
        int invisible_vertex_color_knn /*= 3*/) {
    size_t n_vertex = mesh.vertices_.size();
    mesh.vertex_colors_.clear();
    mesh.vertex_colors_.resize(n_vertex);
    std::vector<size_t> valid_vertices;
    std::vector<size_t> invalid_vertices;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)n_vertex; i++) {
        mesh.vertex_colors_[i] = Eigen::Vector3d::Zero();
        double sum = 0.0;
        for (size_t iter = 0; iter < visiblity_vertex_to_image[i].size();
             iter++) {
            int j = visiblity_vertex_to_image[i][iter];
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
#ifdef _OPENMP
#pragma omp critical
#endif
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
                mesh.SelectDownSample(valid_vertices);
        geometry::KDTreeFlann kd_tree(*valid_mesh);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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
                new_color /= indices.size();
            }
            mesh.vertex_colors_[invalid_vertex] = new_color;
        }
    }
}
}  // namespace color_map
}  // namespace open3d
