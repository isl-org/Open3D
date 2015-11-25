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

#pragma once

#include <vector>
#include <Eigen/Core>

#include "Geometry.h"

namespace three {

class TriangleMesh : public Geometry
{
public:
	TriangleMesh();
	virtual ~TriangleMesh();

public:
	bool CloneFrom(const Geometry &mesh) override;
	Eigen::Vector3d GetMinBound() const override;
	Eigen::Vector3d GetMaxBound() const override;
	void Clear() override;
	bool IsEmpty() const override;
	void Transform(const Eigen::Matrix4d &transformation) override;

	void ComputeTriangleNormals(bool normalized = true);
	void ComputeVertexNormals(bool normalized = true);
	
public:
	bool HasVertices() const {
		return vertices_.size() > 0;
	}
	
	bool HasTriangles() const {
		return vertices_.size() > 0 && triangles_.size() > 0;
	}
	
	bool HasVertexNormals() const {
		return vertices_.size() > 0 &&
				vertex_normals_.size() == vertices_.size();
	}
	
	bool HasVertexColors() const {
		return vertices_.size() > 0 &&
				vertex_colors_.size() == vertices_.size();
	}

	bool HasTriangleNormals() const {
		return HasTriangles() &&
				triangles_.size() == triangle_normals_.size();
	}
	
	void NormalizeNormals() {
		for (size_t i = 0; i < vertex_normals_.size(); i++) {
			vertex_normals_[i].normalize();
		}
		for (size_t i = 0; i < triangle_normals_.size(); i++) {
			triangle_normals_[i].normalize();
		}
	}

public:
	std::vector<Eigen::Vector3d> vertices_;
	std::vector<Eigen::Vector3d> vertex_normals_;
	std::vector<Eigen::Vector3d> vertex_colors_;
	std::vector<Eigen::Vector3i> triangles_;
	std::vector<Eigen::Vector3d> triangle_normals_;
};

}	// namespace three
