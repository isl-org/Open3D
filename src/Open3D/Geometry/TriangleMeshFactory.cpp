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

#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

std::shared_ptr<TriangleMesh> TriangleMesh::CreateTetrahedron(
        double radius /* = 1.0*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogWarning("[CreateTetrahedron] radius <= 0");
        return mesh;
    }
    mesh->vertices_.push_back(radius *
                              Eigen::Vector3d(std::sqrt(8. / 9.), 0, -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-std::sqrt(2. / 9.),
                                                       std::sqrt(2. / 3.),
                                                       -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-std::sqrt(2. / 9.),
                                                       -std::sqrt(2. / 3.),
                                                       -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0., 0., 1.));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 2, 1));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 3, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 3));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 2, 3));
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateOctahedron(
        double radius /* = 1.0*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogWarning("[CreateOctahedron] radius <= 0");
        return mesh;
    }
    mesh->vertices_.push_back(radius * Eigen::Vector3d(1, 0, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, 1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, 0, 1));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(-1, 0, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, -1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3d(0, 0, -1));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 3, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 4, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 0, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 5, 1));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 5, 3));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 5, 4));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 5, 0));
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateIcosahedron(
        double radius /* = 1.0*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogWarning("[CreateIcosahedron] radius <= 0");
        return mesh;
    }
    const double p = (1. + std::sqrt(5.)) / 2.;
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
    mesh->triangles_.push_back(Eigen::Vector3i(0, 4, 1));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 5));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 4, 9));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 9, 10));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 10, 5));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 8, 4));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 11, 8));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 5, 11));
    mesh->triangles_.push_back(Eigen::Vector3i(5, 6, 11));
    mesh->triangles_.push_back(Eigen::Vector3i(5, 10, 6));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 8, 7));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 7, 9));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 6, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 2, 7));
    mesh->triangles_.push_back(Eigen::Vector3i(2, 6, 10));
    mesh->triangles_.push_back(Eigen::Vector3i(2, 10, 9));
    mesh->triangles_.push_back(Eigen::Vector3i(2, 9, 7));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 11, 6));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 8, 11));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 7, 8));
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateBox(double width /* = 1.0*/,
                                                      double height /* = 1.0*/,
                                                      double depth /* = 1.0*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (width <= 0) {
        utility::LogWarning("[CreateBox] width <= 0");
        return mesh_ptr;
    }
    if (height <= 0) {
        utility::LogWarning("[CreateBox] height <= 0");
        return mesh_ptr;
    }
    if (depth <= 0) {
        utility::LogWarning("[CreateBox] depth <= 0");
        return mesh_ptr;
    }
    mesh_ptr->vertices_.resize(8);
    mesh_ptr->vertices_[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
    mesh_ptr->vertices_[1] = Eigen::Vector3d(width, 0.0, 0.0);
    mesh_ptr->vertices_[2] = Eigen::Vector3d(0.0, 0.0, depth);
    mesh_ptr->vertices_[3] = Eigen::Vector3d(width, 0.0, depth);
    mesh_ptr->vertices_[4] = Eigen::Vector3d(0.0, height, 0.0);
    mesh_ptr->vertices_[5] = Eigen::Vector3d(width, height, 0.0);
    mesh_ptr->vertices_[6] = Eigen::Vector3d(0.0, height, depth);
    mesh_ptr->vertices_[7] = Eigen::Vector3d(width, height, depth);
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(4, 7, 5));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(4, 6, 7));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 2, 4));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(2, 6, 4));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 1, 2));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 3, 2));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 5, 7));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 7, 3));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(2, 3, 7));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(2, 7, 6));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 4, 1));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 4, 5));
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateSphere(
        double radius /* = 1.0*/, int resolution /* = 20*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogWarning("[CreateSphere] radius <= 0");
        return mesh_ptr;
    }
    if (resolution <= 0) {
        utility::LogWarning("[CreateSphere] resolution <= 0");
        return mesh_ptr;
    }
    mesh_ptr->vertices_.resize(2 * resolution * (resolution - 1) + 2);
    mesh_ptr->vertices_[0] = Eigen::Vector3d(0.0, 0.0, radius);
    mesh_ptr->vertices_[1] = Eigen::Vector3d(0.0, 0.0, -radius);
    double step = M_PI / (double)resolution;
    for (int i = 1; i < resolution; i++) {
        double alpha = step * i;
        int base = 2 + 2 * resolution * (i - 1);
        for (int j = 0; j < 2 * resolution; j++) {
            double theta = step * j;
            mesh_ptr->vertices_[base + j] =
                    Eigen::Vector3d(sin(alpha) * cos(theta),
                                    sin(alpha) * sin(theta), cos(alpha)) *
                    radius;
        }
    }
    for (int j = 0; j < 2 * resolution; j++) {
        int j1 = (j + 1) % (2 * resolution);
        int base = 2;
        mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, base + j, base + j1));
        base = 2 + 2 * resolution * (resolution - 2);
        mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, base + j1, base + j));
    }
    for (int i = 1; i < resolution - 1; i++) {
        int base1 = 2 + 2 * resolution * (i - 1);
        int base2 = base1 + 2 * resolution;
        for (int j = 0; j < 2 * resolution; j++) {
            int j1 = (j + 1) % (2 * resolution);
            mesh_ptr->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base1 + j1, base1 + j));
            mesh_ptr->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1));
        }
    }
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCylinder(
        double radius /* = 1.0*/,
        double height /* = 2.0*/,
        int resolution /* = 20*/,
        int split /* = 4*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogWarning("[CreateCylinder] radius <= 0");
        return mesh_ptr;
    }
    if (height <= 0) {
        utility::LogWarning("[CreateCylinder] height <= 0");
        return mesh_ptr;
    }
    if (resolution <= 0) {
        utility::LogWarning("[CreateCylinder] resolution <= 0");
        return mesh_ptr;
    }
    if (split <= 0) {
        utility::LogWarning("[CreateCylinder] split <= 0");
        return mesh_ptr;
    }
    mesh_ptr->vertices_.resize(resolution * (split + 1) + 2);
    mesh_ptr->vertices_[0] = Eigen::Vector3d(0.0, 0.0, height * 0.5);
    mesh_ptr->vertices_[1] = Eigen::Vector3d(0.0, 0.0, -height * 0.5);
    double step = M_PI * 2.0 / (double)resolution;
    double h_step = height / (double)split;
    for (int i = 0; i <= split; i++) {
        for (int j = 0; j < resolution; j++) {
            double theta = step * j;
            mesh_ptr->vertices_[2 + resolution * i + j] =
                    Eigen::Vector3d(cos(theta) * radius, sin(theta) * radius,
                                    height * 0.5 - h_step * i);
        }
    }
    for (int j = 0; j < resolution; j++) {
        int j1 = (j + 1) % resolution;
        int base = 2;
        mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, base + j, base + j1));
        base = 2 + resolution * split;
        mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, base + j1, base + j));
    }
    for (int i = 0; i < split; i++) {
        int base1 = 2 + resolution * i;
        int base2 = base1 + resolution;
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            mesh_ptr->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base1 + j1, base1 + j));
            mesh_ptr->triangles_.push_back(
                    Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1));
        }
    }
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCone(double radius /* = 1.0*/,
                                                       double height /* = 2.0*/,
                                                       int resolution /* = 20*/,
                                                       int split /* = 4*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogWarning("[CreateCone] radius <= 0");
        return mesh_ptr;
    }
    if (height <= 0) {
        utility::LogWarning("[CreateCone] height <= 0");
        return mesh_ptr;
    }
    if (resolution <= 0) {
        utility::LogWarning("[CreateCone] resolution <= 0");
        return mesh_ptr;
    }
    if (split <= 0) {
        utility::LogWarning("[CreateCone] split <= 0");
        return mesh_ptr;
    }
    mesh_ptr->vertices_.resize(resolution * split + 2);
    mesh_ptr->vertices_[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
    mesh_ptr->vertices_[1] = Eigen::Vector3d(0.0, 0.0, height);
    double step = M_PI * 2.0 / (double)resolution;
    double h_step = height / (double)split;
    double r_step = radius / (double)split;
    for (int i = 0; i < split; i++) {
        int base = 2 + resolution * i;
        double r = r_step * (split - i);
        for (int j = 0; j < resolution; j++) {
            double theta = step * j;
            mesh_ptr->vertices_[base + j] =
                    Eigen::Vector3d(cos(theta) * r, sin(theta) * r, h_step * i);
        }
    }
    for (int j = 0; j < resolution; j++) {
        int j1 = (j + 1) % resolution;
        int base = 2;
        mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, base + j1, base + j));
        base = 2 + resolution * (split - 1);
        mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, base + j, base + j1));
    }
    for (int i = 0; i < split - 1; i++) {
        int base1 = 2 + resolution * i;
        int base2 = base1 + resolution;
        for (int j = 0; j < resolution; j++) {
            int j1 = (j + 1) % resolution;
            mesh_ptr->triangles_.push_back(
                    Eigen::Vector3i(base2 + j1, base1 + j, base1 + j1));
            mesh_ptr->triangles_.push_back(
                    Eigen::Vector3i(base2 + j1, base2 + j, base1 + j));
        }
    }
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateTorus(
        double torus_radius /* = 1.0 */,
        double tube_radius /* = 0.5 */,
        int radial_resolution /* = 20 */,
        int tubular_resolution /* = 20 */) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (torus_radius <= 0) {
        utility::LogWarning("[CreateTorus] torus_radius <= 0");
        return mesh;
    }
    if (tube_radius <= 0) {
        utility::LogWarning("[CreateTorus] tube_radius <= 0");
        return mesh;
    }
    if (radial_resolution <= 0) {
        utility::LogWarning("[CreateTorus] radial_resolution <= 0");
        return mesh;
    }
    if (tubular_resolution <= 0) {
        utility::LogWarning("[CreateTorus] tubular_resolution <= 0");
        return mesh;
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
        utility::LogWarning("[CreateArrow] cylinder_radius <= 0");
        return std::make_shared<TriangleMesh>();
    }
    if (cone_radius <= 0) {
        utility::LogWarning("[CreateArrow] cone_radius <= 0");
        return std::make_shared<TriangleMesh>();
    }
    if (cylinder_height <= 0) {
        utility::LogWarning("[CreateArrow] cylinder_height <= 0");
        return std::make_shared<TriangleMesh>();
    }
    if (cone_height <= 0) {
        utility::LogWarning("[CreateArrow] cone_height <= 0");
        return std::make_shared<TriangleMesh>();
    }
    if (resolution <= 0) {
        utility::LogWarning("[CreateArrow] resolution <= 0");
        return std::make_shared<TriangleMesh>();
    }
    if (cylinder_split <= 0) {
        utility::LogWarning("[CreateArrow] cylinder_split <= 0");
        return std::make_shared<TriangleMesh>();
    }
    if (cone_split <= 0) {
        utility::LogWarning("[CreateArrow] cone_split <= 0");
        return std::make_shared<TriangleMesh>();
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
        utility::LogWarning("[CreateCoordinateFrame] size <= 0");
        return std::make_shared<TriangleMesh>();
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

    transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 1>(0, 3) = origin;
    mesh_frame->Transform(transformation);

    return mesh_frame;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateMoebius(
        int length_split /* = 70 */,
        int width_split /* = 15 */,
        int twists /* = 1 */,
        double radius /* = 1 */,
        double flatness /* = 1 */,
        double width /* = 1 */,
        double scale /* = 1 */) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (length_split <= 0) {
        utility::LogWarning("[CreateMoebius] length_split <= 0");
        return mesh;
    }
    if (width_split <= 0) {
        utility::LogWarning("[CreateMoebius] width_split <= 0");
        return mesh;
    }
    if (twists < 0) {
        utility::LogWarning("[CreateMoebius] twists < 0");
        return mesh;
    }
    if (radius <= 0) {
        utility::LogWarning("[CreateMoebius] radius <= 0");
        return mesh;
    }
    if (flatness == 0) {
        utility::LogWarning("[CreateMoebius] flatness == 0");
        return mesh;
    }
    if (width <= 0) {
        utility::LogWarning("[CreateMoebius] width <= 0");
        return mesh;
    }
    if (scale <= 0) {
        utility::LogWarning("[CreateMoebius] scale <= 0");
        return mesh;
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
