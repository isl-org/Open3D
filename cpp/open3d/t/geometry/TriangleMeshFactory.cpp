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

#include <vtkLinearExtrusionFilter.h>
#include <vtkNew.h>
#include <vtkTextSource.h>
#include <vtkTriangleFilter.h>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/t/geometry/VtkUtils.h"

namespace open3d {
namespace t {
namespace geometry {

TriangleMesh TriangleMesh::CreateBox(double width,
                                     double height,
                                     double depth,
                                     core::Dtype float_dtype,
                                     core::Dtype int_dtype,
                                     const core::Device &device) {
    // Check width, height, depth.
    if (width <= 0) {
        utility::LogError("width must be > 0, but got {}", width);
    }
    if (height <= 0) {
        utility::LogError("height must be > 0, but got {}", height);
    }
    if (depth <= 0) {
        utility::LogError("depth must be > 0, but got {}", depth);
    }

    // Vertices.
    core::Tensor vertex_positions =
            core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                        {width, 0.0, 0.0},
                                        {0.0, 0.0, depth},
                                        {width, 0.0, depth},
                                        {0.0, height, 0.0},
                                        {width, height, 0.0},
                                        {0.0, height, depth},
                                        {width, height, depth}},
                                       device);

    if (float_dtype == core::Float32) {
        vertex_positions = vertex_positions.To(core::Float32);
    } else if (float_dtype != core::Float64) {
        utility::LogError("float_dtype must be Float32 or Float64, but got {}.",
                          float_dtype.ToString());
    }

    // Triangles.
    core::Tensor triangle_indices = core::Tensor::Init<int64_t>({{4, 7, 5},
                                                                 {4, 6, 7},
                                                                 {0, 2, 4},
                                                                 {2, 6, 4},
                                                                 {0, 1, 2},
                                                                 {1, 3, 2},
                                                                 {1, 5, 7},
                                                                 {1, 7, 3},
                                                                 {2, 3, 7},
                                                                 {2, 7, 6},
                                                                 {0, 4, 1},
                                                                 {1, 4, 5}},
                                                                device);

    if (int_dtype == core::Int32) {
        triangle_indices = triangle_indices.To(core::Int32);
    } else if (int_dtype != core::Int64) {
        utility::LogError("int_dtype must be Int32 or Int64, but got {}.",
                          int_dtype.ToString());
    }

    // Mesh.
    TriangleMesh mesh(vertex_positions, triangle_indices);

    return mesh;
}

TriangleMesh TriangleMesh::CreateSphere(double radius,
                                        int resolution,
                                        core::Dtype float_dtype,
                                        core::Dtype int_dtype,
                                        const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateSphere(radius, resolution);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);
    return mesh;
}

TriangleMesh TriangleMesh::CreateTetrahedron(double radius,
                                             core::Dtype float_dtype,
                                             core::Dtype int_dtype,
                                             const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateTetrahedron(radius);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateOctahedron(double radius,
                                            core::Dtype float_dtype,
                                            core::Dtype int_dtype,
                                            const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateOctahedron(radius);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateIcosahedron(double radius,
                                             core::Dtype float_dtype,
                                             core::Dtype int_dtype,
                                             const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateIcosahedron(radius);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateCylinder(double radius,
                                          double height,
                                          int resolution,
                                          int split,
                                          core::Dtype float_dtype,
                                          core::Dtype int_dtype,
                                          const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateCylinder(radius, height,
                                                           resolution, split);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateCone(double radius,
                                      double height,
                                      int resolution,
                                      int split,
                                      core::Dtype float_dtype,
                                      core::Dtype int_dtype,
                                      const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateCone(radius, height,
                                                       resolution, split);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateTorus(double torus_radius,
                                       double tube_radius,
                                       int radial_resolution,
                                       int tubular_resolution,
                                       core::Dtype float_dtype,
                                       core::Dtype int_dtype,
                                       const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateTorus(
                    torus_radius, tube_radius, radial_resolution,
                    tubular_resolution);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateArrow(double cylinder_radius,
                                       double cone_radius,
                                       double cylinder_height,
                                       double cone_height,
                                       int resolution,
                                       int cylinder_split,
                                       int cone_split,
                                       core::Dtype float_dtype,
                                       core::Dtype int_dtype,
                                       const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateArrow(
                    cylinder_radius, cone_radius, cylinder_height, cone_height,
                    resolution, cylinder_split, cone_split);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateCoordinateFrame(double size,
                                                 const Eigen::Vector3d &origin,
                                                 core::Dtype float_dtype,
                                                 core::Dtype int_dtype,
                                                 const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateCoordinateFrame(size, origin);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateMobius(int length_split,
                                        int width_split,
                                        int twists,
                                        double radius,
                                        double flatness,
                                        double width,
                                        double scale,
                                        core::Dtype float_dtype,
                                        core::Dtype int_dtype,
                                        const core::Device &device) {
    std::shared_ptr<open3d::geometry::TriangleMesh> legacy_mesh =
            open3d::geometry::TriangleMesh::CreateMobius(
                    length_split, width_split, twists, radius, flatness, width,
                    scale);

    TriangleMesh mesh = TriangleMesh::FromLegacy(*legacy_mesh, float_dtype,
                                                 int_dtype, device);

    return mesh;
}

TriangleMesh TriangleMesh::CreateText(const std::string &text,
                                      double depth,
                                      core::Dtype float_dtype,
                                      core::Dtype int_dtype,
                                      const core::Device &device) {
    using namespace vtkutils;

    if (float_dtype != core::Float32 && float_dtype != core::Float64) {
        utility::LogError("float_dtype must be Float32 or Float64, but got {}.",
                          float_dtype.ToString());
    }
    if (int_dtype != core::Int32 && int_dtype != core::Int64) {
        utility::LogError("int_dtype must be Int32 or Int64, but got {}.",
                          int_dtype.ToString());
    }

    vtkNew<vtkTextSource> vector_text;
    vector_text->SetText(text.c_str());
    vector_text->BackingOff();

    vtkNew<vtkLinearExtrusionFilter> extrude;
    vtkNew<vtkTriangleFilter> triangle_filter;
    if (depth > 0) {
        extrude->SetInputConnection(vector_text->GetOutputPort());
        extrude->SetExtrusionTypeToNormalExtrusion();
        extrude->SetVector(0, 0, 1);
        extrude->SetScaleFactor(depth);

        triangle_filter->SetInputConnection(extrude->GetOutputPort());
    } else {
        triangle_filter->SetInputConnection(vector_text->GetOutputPort());
    }

    triangle_filter->Update();
    auto polydata = triangle_filter->GetOutput();
    auto tmesh = CreateTriangleMeshFromVtkPolyData(polydata);
    tmesh.GetVertexPositions() =
            tmesh.GetVertexPositions().To(device, float_dtype);
    tmesh.GetTriangleIndices() =
            tmesh.GetTriangleIndices().To(device, int_dtype);
    return tmesh;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
