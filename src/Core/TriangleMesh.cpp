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

#include <unordered_map>
#include <tuple>

#include <Eigen/Dense>
#include "CoreHelper.h"
#include "Console.h"

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
	size_t add_vert_num = mesh.vertices_.size();
	size_t new_vert_num = old_vert_num + add_vert_num;
	size_t old_tri_num = triangles_.size();
	size_t add_tri_num = mesh.triangles_.size();
	size_t new_tri_num = old_tri_num + add_tri_num;
	if (HasVertexNormals() && mesh.HasVertexNormals()) {
		vertex_normals_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			vertex_normals_[old_vert_num + i] = mesh.vertex_normals_[i];
	}
	if (HasVertexColors() && mesh.HasVertexColors()) {
		vertex_colors_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			vertex_colors_[old_vert_num + i] = mesh.vertex_colors_[i];
	}
	vertices_.resize(new_vert_num);
	for (size_t i = 0; i < add_vert_num; i++)
		vertices_[old_vert_num + i] = mesh.vertices_[i];

	if (HasTriangleNormals() && mesh.HasTriangleNormals()) {
		triangle_normals_.resize(new_tri_num);
		for (size_t i = 0; i < add_tri_num; i++)
			triangle_normals_[old_tri_num + i] = mesh.triangle_normals_[i];
	}
	triangles_.resize(triangles_.size() + mesh.triangles_.size());
	Eigen::Vector3i index_shift((int)old_vert_num, (int)old_vert_num, 
			(int)old_vert_num);
	for (size_t i = 0; i < add_tri_num; i++) {
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

void TriangleMesh::RemoveDuplicatedVertices()
{
	typedef std::tuple<double, double, double> Coordinate3;
	std::unordered_map<Coordinate3, size_t, hash_tuple::hash<Coordinate3>> 
			point_to_old_index;
	std::vector<size_t> index_old_to_new(vertices_.size());
	bool has_vert_normal = HasVertexNormals();
	bool has_vert_color = HasVertexColors();
	size_t k = 0;											// new index
	for (size_t i = 0; i < vertices_.size(); i++) {			// old index
		Coordinate3 coord = std::make_tuple(vertices_[i](0), vertices_[i](1), 
				vertices_[i](2));
		if (point_to_old_index.find(coord) == point_to_old_index.end()) {
			point_to_old_index[coord] = i;
			vertices_[k] = vertices_[i];
			if (has_vert_normal) vertex_normals_[k] = vertex_normals_[i];
			if (has_vert_color) vertex_colors_[k] = vertex_colors_[i];
			index_old_to_new[i] = k;
			k++;
		} else {
			index_old_to_new[i] = index_old_to_new[point_to_old_index[coord]];
		}
	}
	vertices_.resize(k);
	if (has_vert_normal) vertex_normals_.resize(k);
	if (has_vert_color) vertex_colors_.resize(k);
	for (auto &triangle : triangles_) {
		triangle(0) = (int)index_old_to_new[triangle(0)];
		triangle(1) = (int)index_old_to_new[triangle(1)];
		triangle(2) = (int)index_old_to_new[triangle(2)];
	}
	PrintDebug("[RemoveDuplicatedVertices] %d vertices have been removed.\n", 
			index_old_to_new.size() - vertices_.size());
}

void TriangleMesh::RemoveDuplicatedTriangles()
{
}

}	// namespace three
