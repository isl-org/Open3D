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

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

void PrintHelp()
{
	using namespace three;
	PrintInfo("Usage :\n");
	PrintInfo("    > TestTriangleMesh <file1> <file2>\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	SetVerbosityLevel(VERBOSE_ALWAYS);

	if (argc < 3) {
		PrintHelp();
		return 0;
	}

	auto mesh = std::make_shared<TriangleMesh>();
	mesh->vertices_.push_back(Eigen::Vector3d(1,0,0));
	mesh->vertices_.push_back(Eigen::Vector3d(0,1,0));
	mesh->vertices_.push_back(Eigen::Vector3d(1,0,0));
	mesh->triangles_.push_back(Eigen::Vector3i(0,1,2));
	*mesh += *mesh;
	mesh->RemoveDuplicatedVertices();

	auto mesh1 = std::make_shared<TriangleMesh>();
	auto mesh2 = std::make_shared<TriangleMesh>();

	ReadTriangleMesh(argv[1], *mesh1);
	ReadTriangleMesh(argv[2], *mesh2);

	PrintInfo("Mesh1 has %d vertices, %d triangles.\n", mesh1->vertices_.size(),
			mesh1->triangles_.size());
	PrintInfo("Mesh2 has %d vertices, %d triangles.\n", mesh2->vertices_.size(),
			mesh2->triangles_.size());

	*mesh1 += *mesh1;
	PrintInfo("After merge, Mesh1 has %d vertices, %d triangles.\n", 
			mesh1->vertices_.size(), mesh1->triangles_.size());

	mesh1->RemoveDuplicatedVertices();
	PrintInfo("After purge vertices, Mesh1 has %d vertices, %d triangles.\n", 
			mesh1->vertices_.size(), mesh1->triangles_.size());

	DrawGeometry(mesh1);
}