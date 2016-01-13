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

#include "TriangleMesh.h"

namespace three{

std::shared_ptr<TriangleMesh> CreateMeshSphere(double radius/* = 1.0*/, 
		int resolution/* = 20*/)
{
	auto mesh_ptr = std::make_shared<TriangleMesh>();
	mesh_ptr->vertices_.resize(2 * resolution * (resolution -1 ) + 2);
	mesh_ptr->vertices_[0] = Eigen::Vector3d(0.0, 0.0, radius);
	mesh_ptr->vertices_[1] = Eigen::Vector3d(0.0, 0.0, -radius);
	double step = M_PI / (double)resolution;
	for (int i = 1; i < resolution; i++) {
		for (int j = 0; j < 2 * resolution; j++) {
			double alpha = step * i;
			double theta = step * j;
			mesh_ptr->vertices_[2 + 2 * resolution * (i - 1) + j] = 
					Eigen::Vector3d(sin(alpha) * cos(theta), 
					sin(alpha) * sin(theta), cos(alpha)) * radius;
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
			mesh_ptr->triangles_.push_back(Eigen::Vector3i(base2 + j, 
					base1 + j1, base1 + j));
			mesh_ptr->triangles_.push_back(Eigen::Vector3i(base2 + j, 
					base2 + j1, base1 + j1));
		}
	}
	return mesh_ptr;
}

}	// namespace three
