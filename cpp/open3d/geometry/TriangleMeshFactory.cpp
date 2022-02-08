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
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

std::shared_ptr<TriangleMesh> TriangleMesh::CreateTetrahedron(
        double radius /* = 1.0*/, bool create_uv_map /* = false*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateTetrahedron] radius <= 0");
    }

    // Vertices.
    mesh->vertices_.push_back(radius *
                              Eigen::Vector3d(std::sqrt(8. / 9.), 0, -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-std::sqrt(2. / 9.),
                                                       std::sqrt(2. / 3.),
                                                       -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-std::sqrt(2. / 9.),
                                                       -std::sqrt(2. / 3.),
                                                       -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0., 0., 1.));

    // Triangles.
    mesh->triangles_ = {
            {0, 2, 1},
            {0, 3, 2},
            {0, 1, 3},
            {1, 2, 3},
    };

    // UV Map.
    if (create_uv_map) {
        mesh->triangle_uvs_ = {{0.866, 0.5},  {0.433, 0.75}, {0.433, 0.25},
                               {0.866, 0.5},  {0.866, 1.0},  {0.433, 0.75},
                               {0.866, 0.5},  {0.433, 0.25}, {0.866, 0.0},
                               {0.433, 0.25}, {0.433, 0.75}, {0.0, 0.5}};
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateOctahedron(
        double radius /* = 1.0*/, bool create_uv_map /* = false*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateOctahedron] radius <= 0");
    }

    // Vertices.
    mesh->vertices_.push_back(radius * Eigen::Vector3d(1, 0, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, 1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, 0, 1));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-1, 0, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, -1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, 0, -1));

    // Triangles.
    mesh->triangles_ = {{0, 1, 2}, {1, 3, 2}, {3, 4, 2}, {4, 0, 2},
                        {0, 5, 1}, {1, 5, 3}, {3, 5, 4}, {4, 5, 0}};

    // UV Map.
    if (create_uv_map) {
        mesh->triangle_uvs_ = {
                {0.0, 0.75},    {0.1444, 0.5},  {0.2887, 0.75}, {0.1444, 0.5},
                {0.433, 0.5},   {0.2887, 0.75}, {0.433, 0.5},   {0.5773, 0.75},
                {0.2887, 0.75}, {0.5773, 0.75}, {0.433, 1.0},   {0.2887, 0.75},
                {0.0, 0.25},    {0.2887, 0.25}, {0.1444, 0.5},  {0.1444, 0.5},
                {0.2887, 0.25}, {0.433, 0.5},   {0.433, 0.5},   {0.2887, 0.25},
                {0.5773, 0.25}, {0.5773, 0.25}, {0.2887, 0.25}, {0.433, 0.0}};
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateIcosahedron(
        double radius /* = 1.0*/, bool create_uv_map /* = false*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateIcosahedron] radius <= 0");
    }
    const double p = (1. + std::sqrt(5.)) / 2.;

    // Vertices.
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-1, 0, p));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(1, 0, p));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(1, 0, -p));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-1, 0, -p));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, -p, 1));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, p, 1));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, p, -1));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, -p, -1));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-p, -1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(p, -1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(p, 1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-p, 1, 0));

    // Triangles.
    mesh->triangles_ = {{0, 4, 1},  {0, 1, 5},  {1, 4, 9},  {1, 9, 10},
                        {1, 10, 5}, {0, 8, 4},  {0, 11, 8}, {0, 5, 11},
                        {5, 6, 11}, {5, 10, 6}, {4, 8, 7},  {4, 7, 9},
                        {3, 6, 2},  {3, 2, 7},  {2, 6, 10}, {2, 10, 9},
                        {2, 9, 7},  {3, 11, 6}, {3, 8, 11}, {3, 7, 8}};

    // UV Map.
    if (create_uv_map) {
        mesh->triangle_uvs_ = {
                {0.0001, 0.1819}, {0.1575, 0.091},  {0.1575, 0.2728},
                {0.0001, 0.3637}, {0.1575, 0.2728}, {0.1575, 0.4546},
                {0.1575, 0.2728}, {0.1575, 0.091},  {0.3149, 0.1819},
                {0.1575, 0.2728}, {0.3149, 0.1819}, {0.3149, 0.3637},
                {0.1575, 0.2728}, {0.3149, 0.3637}, {0.1575, 0.4546},
                {0.0001, 0.909},  {0.1575, 0.8181}, {0.1575, 0.9999},
                {0.0001, 0.7272}, {0.1575, 0.6363}, {0.1575, 0.8181},
                {0.0001, 0.5454}, {0.1575, 0.4546}, {0.1575, 0.6363},
                {0.1575, 0.4546}, {0.3149, 0.5454}, {0.1575, 0.6363},
                {0.1575, 0.4546}, {0.3149, 0.3637}, {0.3149, 0.5454},
                {0.1575, 0.9999}, {0.1575, 0.8181}, {0.3149, 0.909},
                {0.1575, 0.091},  {0.3149, 0.0001}, {0.3149, 0.1819},
                {0.3149, 0.7272}, {0.3149, 0.5454}, {0.4724, 0.6363},
                {0.3149, 0.7272}, {0.4724, 0.8181}, {0.3149, 0.909},
                {0.4724, 0.4546}, {0.3149, 0.5454}, {0.3149, 0.3637},
                {0.4724, 0.2728}, {0.3149, 0.3637}, {0.3149, 0.1819},
                {0.4724, 0.091},  {0.3149, 0.1819}, {0.3149, 0.0001},
                {0.3149, 0.7272}, {0.1575, 0.6363}, {0.3149, 0.5454},
                {0.3149, 0.7272}, {0.1575, 0.8181}, {0.1575, 0.6363},
                {0.3149, 0.7272}, {0.3149, 0.909},  {0.1575, 0.8181}};
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateBox(
        double width /* = 1.0*/,
        double height /* = 1.0*/,
        double depth /* = 1.0*/,
        bool create_uv_map /* = false*/,
        bool map_texture_to_each_face /*= false*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (width <= 0) {
        utility::LogError("[CreateBox] width <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateBox] height <= 0");
    }
    if (depth <= 0) {
        utility::LogError("[CreateBox] depth <= 0");
    }

    // Vertices.
    mesh->vertices_.resize(8);
    mesh->vertices_[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
    mesh->vertices_[1] = Eigen::Vector3d(width, 0.0, 0.0);
    mesh->vertices_[2] = Eigen::Vector3d(0.0, 0.0, depth);
    mesh->vertices_[3] = Eigen::Vector3d(width, 0.0, depth);
    mesh->vertices_[4] = Eigen::Vector3d(0.0, height, 0.0);
    mesh->vertices_[5] = Eigen::Vector3d(width, height, 0.0);
    mesh->vertices_[6] = Eigen::Vector3d(0.0, height, depth);
    mesh->vertices_[7] = Eigen::Vector3d(width, height, depth);

    // Triangles.
    mesh->triangles_ = {{4, 7, 5}, {4, 6, 7}, {0, 2, 4}, {2, 6, 4},
                        {0, 1, 2}, {1, 3, 2}, {1, 5, 7}, {1, 7, 3},
                        {2, 3, 7}, {2, 7, 6}, {0, 4, 1}, {1, 4, 5}};

    // UV Map.
    if (create_uv_map) {
        if (map_texture_to_each_face) {
            mesh->triangle_uvs_ = {
                    {0.0, 0.0}, {1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0},
                    {1.0, 1.0}, {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0},
                    {1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
                    {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0},
                    {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0},
                    {1.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
                    {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
                    {1.0, 1.0}};
        } else {
            mesh->triangle_uvs_ = {
                    {0.5, 0.5},   {0.75, 0.75}, {0.5, 0.75},  {0.5, 0.5},
                    {0.75, 0.5},  {0.75, 0.75}, {0.25, 0.5},  {0.25, 0.25},
                    {0.5, 0.5},   {0.25, 0.25}, {0.5, 0.25},  {0.5, 0.5},
                    {0.25, 0.5},  {0.25, 0.75}, {0.0, 0.5},   {0.25, 0.75},
                    {0.0, 0.75},  {0.0, 0.5},   {0.25, 0.75}, {0.5, 0.75},
                    {0.5, 1.0},   {0.25, 0.75}, {0.5, 1.0},   {0.25, 1.0},
                    {0.25, 0.25}, {0.25, 0.0},  {0.5, 0.0},   {0.25, 0.25},
                    {0.5, 0.0},   {0.5, 0.25},  {0.25, 0.5},  {0.5, 0.5},
                    {0.25, 0.75}, {0.25, 0.75}, {0.5, 0.5},   {0.5, 0.75}};
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateSphere(
        double radius /* = 1.0*/,
        int resolution /* = 20*/,
        bool create_uv_map /* = false*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateSphere] radius <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateSphere] resolution <= 0");
    }
    mesh->vertices_.resize(2 * resolution * (resolution - 1) + 2);

    std::unordered_map<int64_t, std::pair<double, double>> map_vertices_to_uv;
    std::unordered_map<int64_t, std::pair<double, double>>
            map_cut_vertices_to_uv;

    mesh->vertices_[0] = Eigen::Vector3d(0.0, 0.0, radius);
    mesh->vertices_[1] = Eigen::Vector3d(0.0, 0.0, -radius);
    double step = M_PI / (double)resolution;
    for (int i = 1; i < resolution; i++) {
        double alpha = step * i;
        double uv_row = (1.0 / (resolution)) * i;
        int base = 2 + 2 * resolution * (i - 1);
        for (int j = 0; j < 2 * resolution; j++) {
            double theta = step * j;
            double uv_col = (1.0 / (2.0 * resolution)) * j;
            mesh->vertices_[base + j] =
                    Eigen::Vector3d(sin(alpha) * cos(theta),
                                    sin(alpha) * sin(theta), cos(alpha)) *
                    radius;
            if (create_uv_map) {
                map_vertices_to_uv[base + j] = std::make_pair(uv_row, uv_col);
            }
        }
        if (create_uv_map) {
            map_cut_vertices_to_uv[base] = std::make_pair(uv_row, 1.0);
        }
    }

    // Triangles for poles.
    for (int j = 0; j < 2 * resolution; j++) {
        int j1 = (j + 1) % (2 * resolution);
        int base = 2;
        mesh->triangles_.push_back(Eigen::Vector3i(0, base + j, base + j1));
        base = 2 + 2 * resolution * (resolution - 2);
        mesh->triangles_.push_back(Eigen::Vector3i(1, base + j1, base + j));
    }

    // UV coordinates mapped to triangles for poles.
    if (create_uv_map) {
        for (int j = 0; j < 2 * resolution - 1; j++) {
            int j1 = (j + 1) % (2 * resolution);
            int base = 2;
            double width = 1.0 / (2.0 * resolution);
            double base_offset = width / 2.0;
            double uv_col = base_offset + width * j;
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.0, uv_col));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                    map_vertices_to_uv[base + j].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j1].first,
                                    map_vertices_to_uv[base + j1].second));

            base = 2 + 2 * resolution * (resolution - 2);
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(1.0, uv_col));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j1].first,
                                    map_vertices_to_uv[base + j1].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                    map_vertices_to_uv[base + j].second));
        }

        // UV coordinates mapped to triangles for poles, with cut-vertices.
        int j = 2 * resolution - 1;
        int base = 2;
        double width = 1.0 / (2.0 * resolution);
        double base_offset = width / 2.0;
        double uv_col = base_offset + width * j;
        mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.0, uv_col));
        mesh->triangle_uvs_.push_back(
                Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                map_vertices_to_uv[base + j].second));
        mesh->triangle_uvs_.push_back(
                Eigen::Vector2d(map_cut_vertices_to_uv[base].first,
                                map_cut_vertices_to_uv[base].second));

        base = 2 + 2 * resolution * (resolution - 2);
        mesh->triangle_uvs_.push_back(Eigen::Vector2d(1.0, uv_col));
        mesh->triangle_uvs_.push_back(
                Eigen::Vector2d(map_cut_vertices_to_uv[base].first,
                                map_cut_vertices_to_uv[base].second));
        mesh->triangle_uvs_.push_back(
                Eigen::Vector2d(map_vertices_to_uv[base + j].first,
                                map_vertices_to_uv[base + j].second));
    }

    // Triangles for non-polar region.
    for (int i = 1; i < resolution - 1; i++) {
        int base1 = 2 + 2 * resolution * (i - 1);
        int base2 = base1 + 2 * resolution;
        for (int j = 0; j < 2 * resolution; j++) {
            int j1 = (j + 1) % (2 * resolution);
            mesh->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base1 + j1, base1 + j));
            mesh->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1));
        }
    }

    // UV coordinates mapped to triangles for non-polar region.
    if (create_uv_map) {
        for (int i = 1; i < resolution - 1; i++) {
            int base1 = 2 + 2 * resolution * (i - 1);
            int base2 = base1 + 2 * resolution;
            for (int j = 0; j < 2 * resolution - 1; j++) {
                int j1 = (j + 1) % (2 * resolution);
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j].first,
                                        map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j].first,
                                        map_vertices_to_uv[base1 + j].second));

                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j].first,
                                        map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j1].first,
                                        map_vertices_to_uv[base2 + j1].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
            }

            // UV coordinates mapped to triangles for non-polar region with
            // cut-vertices.
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + 2 * resolution - 1].first,
                    map_vertices_to_uv[base2 + 2 * resolution - 1].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    map_vertices_to_uv[base1 + 2 * resolution - 1].first,
                    map_vertices_to_uv[base1 + 2 * resolution - 1].second));

            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + 2 * resolution - 1].first,
                    map_vertices_to_uv[base2 + 2 * resolution - 1].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base2].first,
                                    map_cut_vertices_to_uv[base2].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCylinder(
        double radius /* = 1.0*/,
        double height /* = 2.0*/,
        int resolution /* = 20*/,
        int split /* = 4*/,
        bool create_uv_map /* = false*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateCylinder] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateCylinder] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateCylinder] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateCylinder] split <= 0");
    }
    mesh->vertices_.resize(resolution * (split + 1) + 2);
    mesh->vertices_[0] = Eigen::Vector3d(0.0, 0.0, height * 0.5);
    mesh->vertices_[1] = Eigen::Vector3d(0.0, 0.0, -height * 0.5);
    double step = M_PI * 2.0 / (double)resolution;
    double h_step = height / (double)split;
    for (int i = 0; i <= split; i++) {
        for (int j = 0; j < resolution; j++) {
            double theta = step * j;
            mesh->vertices_[2 + resolution * i + j] =
                    Eigen::Vector3d(cos(theta) * radius, sin(theta) * radius,
                                    height * 0.5 - h_step * i);
        }
    }

    std::unordered_map<int64_t, std::pair<double, double>> map_vertices_to_uv;
    std::unordered_map<int64_t, std::pair<double, double>>
            map_cut_vertices_to_uv;

    // Mapping vertices to UV coordinates.
    if (create_uv_map) {
        for (int i = 0; i <= split; i++) {
            double uv_row = (1.0 / (double)split) * i;
            for (int j = 0; j < resolution; j++) {
                // double theta = step * j;
                double uv_col = (1.0 / (double)resolution) * j;
                map_vertices_to_uv[2 + resolution * i + j] =
                        std::make_pair(uv_row, uv_col);
            }
            map_cut_vertices_to_uv[2 + resolution * i] =
                    std::make_pair(uv_row, 1.0);
        }
    }

    // Triangles for top and bottom face.
    for (int j = 0; j < resolution; j++) {
        int j1 = (j + 1) % resolution;
        int base = 2;
        mesh->triangles_.push_back(Eigen::Vector3i(0, base + j, base + j1));
        base = 2 + resolution * split;
        mesh->triangles_.push_back(Eigen::Vector3i(1, base + j1, base + j));
    }

    // UV coordinates mapped to triangles for top and bottom face.
    if (create_uv_map) {
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            double theta = step * j;
            double theta1 = step * j1;
            double uv_radius = 0.25;

            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.75, 0.25));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta),
                                    0.25 + uv_radius * sin(theta)));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta1),
                                    0.25 + uv_radius * sin(theta1)));

            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.75, 0.75));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta1),
                                    0.75 + uv_radius * sin(theta1)));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(0.75 + uv_radius * cos(theta),
                                    0.75 + uv_radius * sin(theta)));
        }
    }

    // Triangles for cylindrical surface.
    for (int i = 0; i < split; i++) {
        int base1 = 2 + resolution * i;
        int base2 = base1 + resolution;
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            mesh->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base1 + j1, base1 + j));
            mesh->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1));
        }
    }

    // UV coordinates mapped to triangles for cylindrical surface.
    if (create_uv_map) {
        for (int i = 0; i < split; i++) {
            int base1 = 2 + resolution * i;
            int base2 = base1 + resolution;
            for (int j = 0; j < resolution - 1; j++) {
                int j1 = (j + 1) % resolution;

                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j].first,
                                        map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j].first,
                                        map_vertices_to_uv[base1 + j].second));

                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j].first,
                                        map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base2 + j1].first,
                                        map_vertices_to_uv[base2 + j1].second));
                mesh->triangle_uvs_.push_back(
                        Eigen::Vector2d(map_vertices_to_uv[base1 + j1].first,
                                        map_vertices_to_uv[base1 + j1].second));
            }

            // UV coordinates mapped to triangles for cylindrical surface with
            // cut-vertices.
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + resolution - 1].first,
                    map_vertices_to_uv[base2 + resolution - 1].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    map_vertices_to_uv[base1 + resolution - 1].first,
                    map_vertices_to_uv[base1 + resolution - 1].second));

            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    map_vertices_to_uv[base2 + resolution - 1].first,
                    map_vertices_to_uv[base2 + resolution - 1].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base2].first,
                                    map_cut_vertices_to_uv[base2].second));
            mesh->triangle_uvs_.push_back(
                    Eigen::Vector2d(map_cut_vertices_to_uv[base1].first,
                                    map_cut_vertices_to_uv[base1].second));
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCone(
        double radius /* = 1.0*/,
        double height /* = 2.0*/,
        int resolution /* = 20*/,
        int split /* = 4*/,
        bool create_uv_map /* = false*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateCone] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateCone] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateCone] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateCone] split <= 0");
    }
    mesh->vertices_.resize(resolution * split + 2);
    mesh->vertices_[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
    mesh->vertices_[1] = Eigen::Vector3d(0.0, 0.0, height);
    double step = M_PI * 2.0 / (double)resolution;
    double h_step = height / (double)split;
    double r_step = radius / (double)split;
    std::unordered_map<int64_t, std::pair<double, double>> map_vertices_to_uv;
    for (int i = 0; i < split; i++) {
        int base = 2 + resolution * i;
        double r = r_step * (split - i);
        for (int j = 0; j < resolution; j++) {
            double theta = step * j;
            mesh->vertices_[base + j] =
                    Eigen::Vector3d(cos(theta) * r, sin(theta) * r, h_step * i);

            // Mapping vertices to UV coordinates.
            if (create_uv_map) {
                double factor = 0.25 * r / radius;
                map_vertices_to_uv[base + j] = std::make_pair(
                        factor * cos(theta), factor * sin(theta));
            }
        }
    }

    for (int j = 0; j < resolution; j++) {
        int j1 = (j + 1) % resolution;
        // Triangles for bottom surface.
        int base = 2;
        mesh->triangles_.push_back(Eigen::Vector3i(0, base + j1, base + j));

        // Triangles for top segment of conical surface.
        base = 2 + resolution * (split - 1);
        mesh->triangles_.push_back(Eigen::Vector3i(1, base + j, base + j1));
    }

    if (create_uv_map) {
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            // UV coordinates mapped to triangles for bottom surface.
            int base = 2;
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.5, 0.25));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j1].first,
                    0.25 + map_vertices_to_uv[base + j1].second));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j].first,
                    0.25 + map_vertices_to_uv[base + j].second));

            // UV coordinates mapped to triangles for top segment of conical
            // surface.
            base = 2 + resolution * (split - 1);
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(0.5, 0.75));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j].first,
                    0.75 + map_vertices_to_uv[base + j].second));
            mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                    0.5 + map_vertices_to_uv[base + j1].first,
                    0.75 + map_vertices_to_uv[base + j1].second));
        }
    }

    // Triangles for conical surface other than top-segment.
    for (int i = 0; i < split - 1; i++) {
        int base1 = 2 + resolution * i;
        int base2 = base1 + resolution;
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            mesh->triangles_.push_back(
                    Eigen::Vector3i(base2 + j1, base1 + j, base1 + j1));
            mesh->triangles_.push_back(
                    Eigen::Vector3i(base2 + j1, base2 + j, base1 + j));
        }
    }

    // UV coordinates mapped to triangles for conical surface other than
    // top-segment.
    if (create_uv_map) {
        for (int i = 0; i < split - 1; i++) {
            int base1 = 2 + resolution * i;
            int base2 = base1 + resolution;
            for (int j = 0; j < resolution; j++) {
                int j1 = (j + 1) % resolution;
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base2 + j1].first,
                        0.75 + map_vertices_to_uv[base2 + j1].second));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base1 + j].first,
                        0.75 + map_vertices_to_uv[base1 + j].second));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base1 + j1].first,
                        0.75 + map_vertices_to_uv[base1 + j1].second));

                mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base2 + j1].first,
                        0.75 + map_vertices_to_uv[base2 + j1].second));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base2 + j].first,
                        0.75 + map_vertices_to_uv[base2 + j].second));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(
                        0.5 + map_vertices_to_uv[base1 + j].first,
                        0.75 + map_vertices_to_uv[base1 + j].second));
            }
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateTorus(
        double torus_radius /* = 1.0 */,
        double tube_radius /* = 0.5 */,
        int radial_resolution /* = 20 */,
        int tubular_resolution /* = 20 */) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (torus_radius <= 0) {
        utility::LogError("[CreateTorus] torus_radius <= 0");
    }
    if (tube_radius <= 0) {
        utility::LogError("[CreateTorus] tube_radius <= 0");
    }
    if (radial_resolution <= 0) {
        utility::LogError("[CreateTorus] radial_resolution <= 0");
    }
    if (tubular_resolution <= 0) {
        utility::LogError("[CreateTorus] tubular_resolution <= 0");
    }

    mesh->vertices_.resize(radial_resolution * tubular_resolution);
    mesh->triangles_.resize(2 * radial_resolution * tubular_resolution);
    auto vert_idx = [&](int uidx, int vidx) {
        return uidx * tubular_resolution + vidx;
    };
    double u_step = 2 * M_PI / double(radial_resolution);
    double v_step = 2 * M_PI / double(tubular_resolution);
    for (int uidx = 0; uidx < radial_resolution; ++uidx) {
        double u = uidx * u_step;
        Eigen::Vector3d w(cos(u), sin(u), 0);
        for (int vidx = 0; vidx < tubular_resolution; ++vidx) {
            double v = vidx * v_step;
            mesh->vertices_[vert_idx(uidx, vidx)] =
                    torus_radius * w + tube_radius * cos(v) * w +
                    Eigen::Vector3d(0, 0, tube_radius * sin(v));

            int tri_idx = (uidx * tubular_resolution + vidx) * 2;
            mesh->triangles_[tri_idx + 0] = Eigen::Vector3i(
                    vert_idx((uidx + 1) % radial_resolution, vidx),
                    vert_idx((uidx + 1) % radial_resolution,
                             (vidx + 1) % tubular_resolution),
                    vert_idx(uidx, vidx));
            mesh->triangles_[tri_idx + 1] = Eigen::Vector3i(
                    vert_idx(uidx, vidx),
                    vert_idx((uidx + 1) % radial_resolution,
                             (vidx + 1) % tubular_resolution),
                    vert_idx(uidx, (vidx + 1) % tubular_resolution));
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateArrow(
        double cylinder_radius /* = 1.0*/,
        double cone_radius /* = 1.5*/,
        double cylinder_height /* = 5.0*/,
        double cone_height /* = 4.0*/,
        int resolution /* = 20*/,
        int cylinder_split /* = 4*/,
        int cone_split /* = 1*/) {
    if (cylinder_radius <= 0) {
        utility::LogError("[CreateArrow] cylinder_radius <= 0");
    }
    if (cone_radius <= 0) {
        utility::LogError("[CreateArrow] cone_radius <= 0");
    }
    if (cylinder_height <= 0) {
        utility::LogError("[CreateArrow] cylinder_height <= 0");
    }
    if (cone_height <= 0) {
        utility::LogError("[CreateArrow] cone_height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateArrow] resolution <= 0");
    }
    if (cylinder_split <= 0) {
        utility::LogError("[CreateArrow] cylinder_split <= 0");
    }
    if (cone_split <= 0) {
        utility::LogError("[CreateArrow] cone_split <= 0");
    }
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    auto mesh_cylinder = CreateCylinder(cylinder_radius, cylinder_height,
                                        resolution, cylinder_split);
    transformation(2, 3) = cylinder_height * 0.5;
    mesh_cylinder->Transform(transformation);
    auto mesh_cone =
            CreateCone(cone_radius, cone_height, resolution, cone_split);
    transformation(2, 3) = cylinder_height;
    mesh_cone->Transform(transformation);
    auto mesh_arrow = mesh_cylinder;
    *mesh_arrow += *mesh_cone;
    return mesh_arrow;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCoordinateFrame(
        double size /* = 1.0*/,
        const Eigen::Vector3d &origin /* = Eigen::Vector3d(0.0, 0.0, 0.0)*/) {
    if (size <= 0) {
        utility::LogError("[CreateCoordinateFrame] size <= 0");
    }
    auto mesh_frame = CreateSphere(0.06 * size);
    mesh_frame->ComputeVertexNormals();
    mesh_frame->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));

    std::shared_ptr<TriangleMesh> mesh_arrow;
    Eigen::Matrix4d transformation;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
    transformation << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));
    transformation << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(0.0, 0.0, 1.0));
    transformation << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_frame->Translate(origin, true);

    return mesh_frame;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateMobius(
        int length_split /* = 70 */,
        int width_split /* = 15 */,
        int twists /* = 1 */,
        double radius /* = 1 */,
        double flatness /* = 1 */,
        double width /* = 1 */,
        double scale /* = 1 */) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (length_split <= 0) {
        utility::LogError("[CreateMobius] length_split <= 0");
    }
    if (width_split <= 0) {
        utility::LogError("[CreateMobius] width_split <= 0");
    }
    if (twists < 0) {
        utility::LogError("[CreateMobius] twists < 0");
    }
    if (radius <= 0) {
        utility::LogError("[CreateMobius] radius <= 0");
    }
    if (flatness == 0) {
        utility::LogError("[CreateMobius] flatness == 0");
    }
    if (width <= 0) {
        utility::LogError("[CreateMobius] width <= 0");
    }
    if (scale <= 0) {
        utility::LogError("[CreateMobius] scale <= 0");
    }

    mesh->vertices_.resize(length_split * width_split);

    double u_step = 2 * M_PI / length_split;
    double v_step = width / (width_split - 1);
    for (int uidx = 0; uidx < length_split; ++uidx) {
        double u = uidx * u_step;
        double cos_u = std::cos(u);
        double sin_u = std::sin(u);
        for (int vidx = 0; vidx < width_split; ++vidx) {
            int idx = uidx * width_split + vidx;
            double v = -width / 2.0 + vidx * v_step;
            double alpha = twists * 0.5 * u;
            double cos_alpha = std::cos(alpha);
            double sin_alpha = std::sin(alpha);
            mesh->vertices_[idx](0) =
                    scale * ((cos_alpha * cos_u * v) + radius * cos_u);
            mesh->vertices_[idx](1) =
                    scale * ((cos_alpha * sin_u * v) + radius * sin_u);
            mesh->vertices_[idx](2) = scale * sin_alpha * v * flatness;
        }
    }

    for (int uidx = 0; uidx < length_split - 1; ++uidx) {
        for (int vidx = 0; vidx < width_split - 1; ++vidx) {
            if ((uidx + vidx) % 2 == 0) {
                mesh->triangles_.push_back(
                        Eigen::Vector3i(uidx * width_split + vidx,
                                        (uidx + 1) * width_split + vidx + 1,
                                        uidx * width_split + vidx + 1));
                mesh->triangles_.push_back(
                        Eigen::Vector3i(uidx * width_split + vidx,
                                        (uidx + 1) * width_split + vidx,
                                        (uidx + 1) * width_split + vidx + 1));
            } else {
                mesh->triangles_.push_back(
                        Eigen::Vector3i(uidx * width_split + vidx + 1,
                                        uidx * width_split + vidx,
                                        (uidx + 1) * width_split + vidx));
                mesh->triangles_.push_back(
                        Eigen::Vector3i(uidx * width_split + vidx + 1,
                                        (uidx + 1) * width_split + vidx,
                                        (uidx + 1) * width_split + vidx + 1));
            }
        }
    }

    int uidx = length_split - 1;
    for (int vidx = 0; vidx < width_split - 1; ++vidx) {
        if (twists % 2 == 1) {
            if ((uidx + vidx) % 2 == 0) {
                mesh->triangles_.push_back(
                        Eigen::Vector3i((width_split - 1) - (vidx + 1),
                                        uidx * width_split + vidx,
                                        uidx * width_split + vidx + 1));
                mesh->triangles_.push_back(Eigen::Vector3i(
                        (width_split - 1) - vidx, uidx * width_split + vidx,
                        (width_split - 1) - (vidx + 1)));
            } else {
                mesh->triangles_.push_back(
                        Eigen::Vector3i(uidx * width_split + vidx,
                                        uidx * width_split + vidx + 1,
                                        (width_split - 1) - vidx));
                mesh->triangles_.push_back(Eigen::Vector3i(
                        (width_split - 1) - vidx, uidx * width_split + vidx + 1,
                        (width_split - 1) - (vidx + 1)));
            }
        } else {
            if ((uidx + vidx) % 2 == 0) {
                mesh->triangles_.push_back(
                        Eigen::Vector3i(uidx * width_split + vidx, vidx + 1,
                                        uidx * width_split + vidx + 1));
                mesh->triangles_.push_back(Eigen::Vector3i(
                        uidx * width_split + vidx, vidx, vidx + 1));
            } else {
                mesh->triangles_.push_back(
                        Eigen::Vector3i(uidx * width_split + vidx, vidx,
                                        uidx * width_split + vidx + 1));
                mesh->triangles_.push_back(Eigen::Vector3i(
                        uidx * width_split + vidx + 1, vidx, vidx + 1));
            }
        }
    }

    return mesh;
}

}  // namespace geometry
}  // namespace open3d
