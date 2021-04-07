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

#include <iostream>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    utility::LogInfo("Usage :");
    utility::LogInfo("    > TriangleMesh sphere");
    utility::LogInfo("    > TriangleMesh merge <file1> <file2>");
    utility::LogInfo("    > TriangleMesh normal <file1> <file2>");
}

void PaintMesh(open3d::geometry::TriangleMesh &mesh,
               const Eigen::Vector3d &color) {
    mesh.vertex_colors_.resize(mesh.vertices_.size());
    for (size_t i = 0; i < mesh.vertices_.size(); i++) {
        mesh.vertex_colors_[i] = color;
    }
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    std::string option(argv[1]);

    if (option == "box") {
        auto mesh =
                geometry::TriangleMesh::CreateBox(1.0, 1.0, 1.0, true, true);
        // mesh->ComputeVertexNormals();
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Box_each_face.obj", *mesh, false, false, false,
                              true, true, true);
        auto mesh2 =
                geometry::TriangleMesh::CreateBox(1.0, 1.0, 1.0, true, false);
        // mesh->ComputeVertexNormals();
        utility::LogInfo(" Has UV: {}", mesh2->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Box.obj", *mesh2, false, false, false, true,
                              true, true);
    }

    else if (option == "tetrahedron") {
        auto mesh = geometry::TriangleMesh::CreateTetrahedron(1.0, true);
        // mesh->ComputeVertexNormals();
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Tetrahedron.obj", *mesh, false, false, false,
                              true, true, true);
    }

    else if (option == "octahedron") {
        auto mesh = geometry::TriangleMesh::CreateOctahedron(1.0, true);
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Octahedron.obj", *mesh, false, false, false,
                              true, true, true);
    }

    else if (option == "icosahedron") {
        auto mesh = geometry::TriangleMesh::CreateIcosahedron(1.0, true);
        // mesh->ComputeVertexNormals();
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Icosahedron.obj", *mesh, false, false, false,
                              true, true, true);
    }

    else if (option == "cylinder") {
        auto mesh = geometry::TriangleMesh::CreateCylinder(2.0, 5.0, 10, 5);
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Cylinder.obj", *mesh, false, false, false, true,
                              true, true);
    }

    else if (option == "cone") {
        auto mesh = geometry::TriangleMesh::CreateCone(2.0, 5.0, 10, 5);
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Cone.obj", *mesh, false, false, false, true,
                              true, true);
    }

    else if (option == "sphere") {
        auto mesh = geometry::TriangleMesh::CreateSphere(2.0, 10.0);
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("Sphere.obj", *mesh, false, false, false, true,
                              true, true);
    }

    return 0;
}
