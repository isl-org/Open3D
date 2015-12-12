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

#include <Eigen/Dense>

namespace three{

TriangleMesh::TriangleMesh() : Geometry(GEOMETRY_TRIANGLEMESH)
{
}

TriangleMesh::~TriangleMesh()
{
}

Eigen::Vector3d TriangleMesh::GetMinBound() const
{
	if (!HasVertices()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x = std::min_element(vertices_.begin(), vertices_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(0) < b(0); });
	auto itr_y = std::min_element(vertices_.begin(), vertices_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(1) < b(1); });
	auto itr_z = std::min_element(vertices_.begin(), vertices_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(2) < b(2); });
	return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d TriangleMesh::GetMaxBound() const
{
	if (!HasVertices()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x = std::max_element(vertices_.begin(), vertices_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(0) < b(0); });
	auto itr_y = std::max_element(vertices_.begin(), vertices_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(1) < b(1); });
	auto itr_z = std::max_element(vertices_.begin(), vertices_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(2) < b(2); });
	return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}
	
void TriangleMesh::Clear()
{
	vertices_.clear();
	triangles_.clear();
}

bool TriangleMesh::IsEmpty() const
{
	return !HasVertices();
}

void TriangleMesh::Transform(const Eigen::Matrix4d &transformation)
{
	for (size_t i = 0; i < vertices_.size(); i++) {
		Eigen::Vector4d new_point = transformation * Eigen::Vector4d(
				vertices_[i](0), vertices_[i](1), vertices_[i](2), 1.0);
		vertices_[i] = new_point.block<3, 1>(0, 0);
	}
	
	for (size_t i = 0; i < vertex_normals_.size(); i++) {
		Eigen::Vector4d new_normal = transformation * Eigen::Vector4d(
				vertex_normals_[i](0), vertex_normals_[i](1),
				vertex_normals_[i](2), 0.0);
		vertex_normals_[i] = new_normal.block<3, 1>(0, 0);
	}

	for (size_t i = 0; i < triangle_normals_.size(); i++) {
		Eigen::Vector4d new_normal = transformation * Eigen::Vector4d(
				triangle_normals_[i](0), triangle_normals_[i](1),
				triangle_normals_[i](2), 0.0);
		triangle_normals_[i] = new_normal.block<3, 1>(0, 0);
	}
}

TriangleMesh &TriangleMesh::operator+=(const TriangleMesh &mesh)
{
	size_t old_vert_num = vertices_.size();
	size_t old_tri_num = triangles_.size();
	if (HasVertexNormals() && mesh.HasVertexNormals()) {
		vertex_normals_.insert(vertex_normals_.end(),
				mesh.vertex_normals_.begin(), mesh.vertex_normals_.end());
	}
	if (HasVertexColors() && mesh.HasVertexColors()) {
		vertex_colors_.insert(vertex_colors_.end(),
				mesh.vertex_colors_.begin(), mesh.vertex_colors_.end());
	}
	vertices_.insert(vertices_.end(), mesh.vertices_.begin(), 
			mesh.vertices_.end());
	if (HasTriangleNormals() && mesh.HasTriangleNormals()) {
		triangle_normals_.insert(triangle_normals_.end(),
				mesh.triangle_normals_.begin(), mesh.triangle_normals_.end());
	}
	triangles_.resize(triangles_.size() + mesh.triangles_.size());
	Eigen::Vector3i index_shift((int)old_vert_num, (int)old_vert_num, 
			(int)old_vert_num);
	for (size_t i = 0; i < mesh.triangles_.size(); i++) {
		triangles_[old_tri_num + i] = mesh.triangles_[i] + index_shift;
	}
	return (*this);
}

const TriangleMesh TriangleMesh::operator+(const TriangleMesh &mesh)
{
	return (TriangleMesh(*this) += mesh);
}

void TriangleMesh::ComputeTriangleNormals(bool normalized/* = true*/)
{
	triangle_normals_.resize(triangles_.size());
	for (size_t i = 0; i < triangles_.size(); i++) {
		auto &triangle = triangles_[i];
		Eigen::Vector3d v01 = vertices_[triangle(1)] - vertices_[triangle(0)];
		Eigen::Vector3d v02 = vertices_[triangle(2)] - vertices_[triangle(0)];
		triangle_normals_[i] = v01.cross(v02);
	}
	if (normalized) {
		NormalizeNormals();
	}
}

void TriangleMesh::ComputeVertexNormals(bool normalized/* = true*/)
{
	if (HasTriangleNormals() == false) {
		ComputeTriangleNormals(false);
	}
	vertex_normals_.resize(vertices_.size(), Eigen::Vector3d::Zero());
	for (size_t i = 0; i < triangles_.size(); i++) {
		auto &triangle = triangles_[i];
		vertex_normals_[triangle(0)] += triangle_normals_[i];
		vertex_normals_[triangle(1)] += triangle_normals_[i];
		vertex_normals_[triangle(2)] += triangle_normals_[i];
	}
	if (normalized) {
		NormalizeNormals();
	}
}

}	// namespace three
