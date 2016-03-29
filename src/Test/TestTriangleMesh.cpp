// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

void PrintHelp()
{
	using namespace three;
	PrintInfo("Usage :\n");
	PrintInfo("    > TestTriangleMesh sphere\n");
	PrintInfo("    > TestTriangleMesh merge <file1> <file2>\n");
	PrintInfo("    > TestTriangleMesh normal <file1> <file2>\n");
}

void PaintMesh(three::TriangleMesh &mesh, const Eigen::Vector3d &color)
{
	mesh.vertex_colors_.resize(mesh.vertices_.size());
	for (size_t i = 0; i < mesh.vertices_.size(); i++) {
		mesh.vertex_colors_[i] = color;
	}
}

int main(int argc, char *argv[])
{
	using namespace three;

	SetVerbosityLevel(VERBOSE_ALWAYS);

	if (argc < 2) {
		PrintHelp();
		return 0;
	}

	std::string option(argv[1]);
	if (option == "sphere") {
		auto mesh = CreateMeshSphere(0.05);
		mesh->ComputeVertexNormals();
		DrawGeometry(mesh);
		WriteTriangleMesh("sphere.ply", *mesh, true, true);
	} else if (option == "cylinder") {
		auto mesh = CreateMeshCylinder(0.5, 2.0);
		mesh->ComputeVertexNormals();
		DrawGeometry(mesh);
		WriteTriangleMesh("cylinder.ply", *mesh, true, true);
	} else if (option == "cone") {
		auto mesh = CreateMeshCone(0.5, 2.0, 20, 3);
		mesh->ComputeVertexNormals();
		DrawGeometry(mesh);
		WriteTriangleMesh("cone.ply", *mesh, true, true);
	} else if (option == "arrow") {
		auto mesh = CreateMeshArrow();
		mesh->ComputeVertexNormals();
		DrawGeometry(mesh);
		WriteTriangleMesh("arrow.ply", *mesh, true, true);
	} else if (option == "frame") {
		if (argc < 3) {
			auto mesh = CreateMeshCoordinateFrame();
			DrawGeometry(mesh);
			WriteTriangleMesh("frame.ply", *mesh, true, true);
		} else {
			auto mesh = CreateMeshFromFile(argv[2]);
			mesh->ComputeVertexNormals();
			BoundingBox boundingbox(*mesh);
			auto mesh_frame = CreateMeshCoordinateFrame(
					boundingbox.GetSize() * 0.2, boundingbox.GetMinBound());
			std::vector<std::shared_ptr<const Geometry>> meshes =
					{mesh, mesh_frame};
			DrawGeometries(meshes);
		}
	} else if (option == "merge") {
		auto mesh1 = CreateMeshFromFile(argv[2]);
		auto mesh2 = CreateMeshFromFile(argv[3]);

		PrintInfo("Mesh1 has %d vertices, %d triangles.\n", mesh1->vertices_.size(),
				mesh1->triangles_.size());
		PrintInfo("Mesh2 has %d vertices, %d triangles.\n", mesh2->vertices_.size(),
				mesh2->triangles_.size());

		*mesh1 += *mesh2;
		PrintInfo("After merge, Mesh1 has %d vertices, %d triangles.\n", 
				mesh1->vertices_.size(), mesh1->triangles_.size());

		mesh1->Purge();
		PrintInfo("After purge vertices, Mesh1 has %d vertices, %d triangles.\n", 
				mesh1->vertices_.size(), mesh1->triangles_.size());

		DrawGeometry(mesh1);

		WriteTriangleMesh("temp.ply", *mesh1, true, true);
	} else if (option == "normal") {
		auto mesh = CreateMeshFromFile(argv[2]);
		mesh->ComputeVertexNormals();
		WriteTriangleMesh(argv[3], *mesh, true, true);
	}
}