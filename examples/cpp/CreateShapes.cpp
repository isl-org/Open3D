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
    if (option == "sphere") {
        auto mesh = geometry::TriangleMesh::CreateSphere(0.05);
        mesh->ComputeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("sphere.ply", *mesh, true, true);
    }    
    
    else if (option == "box") {
        auto mesh = geometry::TriangleMesh::CreateBox(3.0, 4.0, 5.0);
        // mesh->ComputeVertexNormals();
        utility::LogInfo(" Has UV: {}", mesh->HasTriangleUvs());
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("box.obj", *mesh, true, false, false, true, true, true);
    } 
    
    else if (option == "cylinder") {
        auto mesh = geometry::TriangleMesh::CreateCylinder(0.5, 2.0);
        mesh->ComputeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("cylinder.ply", *mesh, true, true);
    } 
    
    else if (option == "cone") {
        auto mesh = geometry::TriangleMesh::CreateCone(0.5, 2.0, 20, 3);
        mesh->ComputeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("cone.ply", *mesh, true, true);
    }
    return 0;
}
