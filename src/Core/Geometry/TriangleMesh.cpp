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

#include "TriangleMesh.h"

#include <unordered_map>
#include <tuple>

#include <Eigen/Dense>
#include <Core/Utility/Helper.h>
#include <Core/Utility/Console.h>

namespace three{

void TriangleMesh::Clear()
{
	vertices_.clear();
	triangles_.clear();
}

bool TriangleMesh::IsEmpty() const
{
	return !HasVertices();
}

Eigen::Vector3d TriangleMesh::GetMinBound() const
{
	if (!HasVertices()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x = std::min_element(vertices_.begin(), vertices_.end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
	auto itr_y = std::min_element(vertices_.begin(), vertices_.end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
	auto itr_z = std::min_element(vertices_.begin(), vertices_.end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
	return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d TriangleMesh::GetMaxBound() const
{
	if (!HasVertices()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x = std::max_element(vertices_.begin(), vertices_.end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
	auto itr_y = std::max_element(vertices_.begin(), vertices_.end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
	auto itr_z = std::max_element(vertices_.begin(), vertices_.end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
	return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

void TriangleMesh::Transform(const Eigen::Matrix4d &transformation)
{
	for (auto &vertex : vertices_) {
		Eigen::Vector4d new_point = transformation * Eigen::Vector4d(
				vertex(0), vertex(1), vertex(2), 1.0);
		vertex = new_point.block<3, 1>(0, 0);
	}
	for (auto &vertex_normal : vertex_normals_) {
		Eigen::Vector4d new_normal = transformation * Eigen::Vector4d(
				vertex_normal(0), vertex_normal(1), vertex_normal(2), 0.0);
		vertex_normal = new_normal.block<3, 1>(0, 0);
	}
	for (auto &triangle_normal : triangle_normals_) {
		Eigen::Vector4d new_normal = transformation * Eigen::Vector4d(
				triangle_normal(0), triangle_normal(1),
				triangle_normal(2), 0.0);
		triangle_normal = new_normal.block<3, 1>(0, 0);
	}
}

TriangleMesh &TriangleMesh::operator+=(const TriangleMesh &mesh)
{
	if (mesh.IsEmpty()) return (*this);
	size_t old_vert_num = vertices_.size();
	size_t add_vert_num = mesh.vertices_.size();
	size_t new_vert_num = old_vert_num + add_vert_num;
	size_t old_tri_num = triangles_.size();
	size_t add_tri_num = mesh.triangles_.size();
	size_t new_tri_num = old_tri_num + add_tri_num;
	if ((!HasVertices() || HasVertexNormals()) && mesh.HasVertexNormals()) {
		vertex_normals_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			vertex_normals_[old_vert_num + i] = mesh.vertex_normals_[i];
	} else {
		vertex_normals_.clear();
	}
	if ((!HasVertices() || HasVertexColors()) && mesh.HasVertexColors()) {
		vertex_colors_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			vertex_colors_[old_vert_num + i] = mesh.vertex_colors_[i];
	} else {
		vertex_colors_.clear();
	}
	vertices_.resize(new_vert_num);
	for (size_t i = 0; i < add_vert_num; i++)
		vertices_[old_vert_num + i] = mesh.vertices_[i];

	if ((!HasTriangles() || HasTriangleNormals()) &&
			mesh.HasTriangleNormals()) {
		triangle_normals_.resize(new_tri_num);
		for (size_t i = 0; i < add_tri_num; i++)
			triangle_normals_[old_tri_num + i] = mesh.triangle_normals_[i];
	} else {
		triangle_normals_.clear();
	}
	triangles_.resize(triangles_.size() + mesh.triangles_.size());
	Eigen::Vector3i index_shift((int)old_vert_num, (int)old_vert_num,
			(int)old_vert_num);
	for (size_t i = 0; i < add_tri_num; i++) {
		triangles_[old_tri_num + i] = mesh.triangles_[i] + index_shift;
	}
	return (*this);
}

TriangleMesh TriangleMesh::operator+(const TriangleMesh &mesh) const
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

void TriangleMesh::Purge()
{
	RemoveDuplicatedVertices();
	RemoveDuplicatedTriangles();
	RemoveNonManifoldTriangles();
	RemoveNonManifoldVertices();
}

void TriangleMesh::RemoveDuplicatedVertices()
{
	typedef std::tuple<double, double, double> Coordinate3;
	std::unordered_map<Coordinate3, size_t, hash_tuple::hash<Coordinate3>>
			point_to_old_index;
	std::vector<int> index_old_to_new(vertices_.size());
	bool has_vert_normal = HasVertexNormals();
	bool has_vert_color = HasVertexColors();
	size_t old_vertex_num = vertices_.size();
	size_t k = 0;											// new index
	for (size_t i = 0; i < old_vertex_num; i++) {			// old index
		Coordinate3 coord = std::make_tuple(vertices_[i](0), vertices_[i](1),
				vertices_[i](2));
		if (point_to_old_index.find(coord) == point_to_old_index.end()) {
			point_to_old_index[coord] = i;
			vertices_[k] = vertices_[i];
			if (has_vert_normal) vertex_normals_[k] = vertex_normals_[i];
			if (has_vert_color) vertex_colors_[k] = vertex_colors_[i];
			index_old_to_new[i] = (int)k;
			k++;
		} else {
			index_old_to_new[i] = index_old_to_new[point_to_old_index[coord]];
		}
	}
	vertices_.resize(k);
	if (has_vert_normal) vertex_normals_.resize(k);
	if (has_vert_color) vertex_colors_.resize(k);
	if (k < old_vertex_num) {
		for (auto &triangle : triangles_) {
			triangle(0) = index_old_to_new[triangle(0)];
			triangle(1) = index_old_to_new[triangle(1)];
			triangle(2) = index_old_to_new[triangle(2)];
		}
	}
	PrintDebug("[RemoveDuplicatedVertices] %d vertices have been removed.\n",
			(int)(old_vertex_num - k));
}

void TriangleMesh::RemoveDuplicatedTriangles()
{
	typedef std::tuple<int, int, int> Index3;
	std::unordered_map<Index3, size_t, hash_tuple::hash<Index3>>
			triangle_to_old_index;
	bool has_tri_normal = HasTriangleNormals();
	size_t old_triangle_num = triangles_.size();
	size_t k = 0;
	for (size_t i = 0; i < old_triangle_num; i++) {
		Index3 index;
		// We first need to find the minimum index. Because triangle (0-1-2) and
		// triangle (2-0-1) are the same.
		if (triangles_[i](0) <= triangles_[i](1)) {
			if (triangles_[i](0) <= triangles_[i](2)) {
				index = std::make_tuple(triangles_[i](0), triangles_[i](1),
						triangles_[i](2));
			} else {
				index = std::make_tuple(triangles_[i](2), triangles_[i](0),
						triangles_[i](1));
			}
		} else {
			if (triangles_[i](1) <= triangles_[i](2)) {
				index = std::make_tuple(triangles_[i](1), triangles_[i](2),
						triangles_[i](0));
			} else {
				index = std::make_tuple(triangles_[i](2), triangles_[i](0),
						triangles_[i](1));
			}
		}
		if (triangle_to_old_index.find(index) == triangle_to_old_index.end()) {
			triangle_to_old_index[index] = i;
			triangles_[k] = triangles_[i];
			if (has_tri_normal) triangle_normals_[k] = triangle_normals_[i];
			k++;
		}
	}
	triangles_.resize(k);
	if (has_tri_normal) triangle_normals_.resize(k);
	PrintDebug("[RemoveDuplicatedTriangles] %d triangles have been removed.\n",
			(int)(old_triangle_num - k));
}

void TriangleMesh::RemoveNonManifoldVertices()
{
	// Non-manifold vertices are vertices without a triangle reference. They
	// should not exist in a valid triangle mesh.
	std::vector<bool> vertex_has_reference(vertices_.size(), false);
	for (const auto &triangle : triangles_) {
		vertex_has_reference[triangle(0)] = true;
		vertex_has_reference[triangle(1)] = true;
		vertex_has_reference[triangle(2)] = true;
	}
	std::vector<int> index_old_to_new(vertices_.size());
	bool has_vert_normal = HasVertexNormals();
	bool has_vert_color = HasVertexColors();
	size_t old_vertex_num = vertices_.size();
	size_t k = 0;											// new index
	for (size_t i = 0; i < old_vertex_num; i++) {			// old index
		if (vertex_has_reference[i]) {
			vertices_[k] = vertices_[i];
			if (has_vert_normal) vertex_normals_[k] = vertex_normals_[i];
			if (has_vert_color) vertex_colors_[k] = vertex_colors_[i];
			index_old_to_new[i] = (int)k;
			k++;
		} else {
			index_old_to_new[i] = -1;
		}
	}
	vertices_.resize(k);
	if (has_vert_normal) vertex_normals_.resize(k);
	if (has_vert_color) vertex_colors_.resize(k);
	if (k < old_vertex_num) {
		for (auto &triangle : triangles_) {
			triangle(0) = index_old_to_new[triangle(0)];
			triangle(1) = index_old_to_new[triangle(1)];
			triangle(2) = index_old_to_new[triangle(2)];
		}
	}
	PrintDebug("[RemoveNonManifoldVertices] %d vertices have been removed.\n",
			(int)(old_vertex_num - k));
}

void TriangleMesh::RemoveNonManifoldTriangles()
{
	// Non-manifold triangles are degenerate triangles that have one vertex as
	// its multiple end-points. They are usually the product of removing
	// duplicated vertices.
	bool has_tri_normal = HasTriangleNormals();
	size_t old_triangle_num = triangles_.size();
	size_t k = 0;
	for (size_t i = 0; i < old_triangle_num; i++) {
		const auto &triangle = triangles_[i];
		if (triangle(0) != triangle(1) && triangle(1) != triangle(2) &&
				triangle(2) != triangle(0)) {
			triangles_[k] = triangles_[i];
			if (has_tri_normal) triangle_normals_[k] = triangle_normals_[i];
			k++;
		}
	}
	triangles_.resize(k);
	if (has_tri_normal) triangle_normals_.resize(k);
	PrintDebug("[RemoveNonManifoldTriangles] %d triangles have been removed.\n",
			(int)(old_triangle_num - k));
}

}	// namespace three
