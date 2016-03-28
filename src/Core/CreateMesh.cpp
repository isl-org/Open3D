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
	if (radius <= 0.0 || resolution <= 0) {
		return mesh_ptr;
	}
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

std::shared_ptr<TriangleMesh> CreateMeshCylinder(double radius/* = 1.0*/,
		double height/* = 2.0*/, int resolution/* = 20*/, int split/* = 4*/)
{
	auto mesh_ptr = std::make_shared<TriangleMesh>();
	if (radius <= 0.0 || height <= 0.0 || resolution <= 0 || split <= 0) {
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
			mesh_ptr->triangles_.push_back(Eigen::Vector3i(base2 + j,
					base1 + j1, base1 + j));
			mesh_ptr->triangles_.push_back(Eigen::Vector3i(base2 + j,
					base2 + j1, base1 + j1));
		}
	}
	return mesh_ptr;
}

std::shared_ptr<TriangleMesh> CreateMeshCone(double radius/* = 1.0*/,
		double height/* = 2.0*/, int resolution/* = 20*/, int split/* = 4*/)
{
	auto mesh_ptr = std::make_shared<TriangleMesh>();
	if (radius <= 0.0 || height <= 0.0 || resolution <= 0 || split <= 0) {
		return mesh_ptr;
	}
	mesh_ptr->vertices_.resize(resolution * split + 2);
	mesh_ptr->vertices_[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
	mesh_ptr->vertices_[1] = Eigen::Vector3d(0.0, 0.0, height);
	double step = M_PI * 2.0 / (double)resolution;
	double h_step = height / (double)split;
	double r_step = radius / (double)split;
	for (int i = 0; i < split; i++) {
		for (int j = 0; j < resolution; j++) {
			double theta = step * j;
			double r = r_step * (split - i);
			mesh_ptr->vertices_[2 + resolution * i + j] =
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
			mesh_ptr->triangles_.push_back(Eigen::Vector3i(base2 + j1,
					base1 + j, base1 + j1));
			mesh_ptr->triangles_.push_back(Eigen::Vector3i(base2 + j1,
					base2 + j, base1 + j));
		}
	}
	return mesh_ptr;
}

}	// namespace three
