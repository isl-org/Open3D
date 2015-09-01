// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
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

#include "ShaderPointCloud.h"

#include "Shader.h"

namespace three{

namespace glsl {

ShaderPointCloudDefault::ShaderPointCloudDefault() :
		count_(0)
{
}

ShaderPointCloudDefault::~ShaderPointCloudDefault()
{
}

bool ShaderPointCloudDefault::Compile()
{
	if (CompileShaders(
			&PointCloudVertexShader,
			NULL,
			&PointCloudFragmentShader) == false)
	{
		return false;
	}
	
	vertex_position_ = glGetAttribLocation(program_, "vertex_position");
	vertex_color_ = glGetAttribLocation(program_, "vertex_color");
	MVP_ = glGetUniformLocation(program_, "MVP");

	return true;
}

void ShaderPointCloudDefault::Release()
{
	UnbindGeometry();
	ReleaseProgram();
}

bool ShaderPointCloudDefault::BindGeometry(const Geometry &geometry)
{
	// Sanity check to see if this geometry is worth binding.
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD) {
		return false;
	}
	const PointCloud &pointcloud = (const PointCloud&)geometry;
	if (pointcloud.HasPoints() == false || pointcloud.HasColors() == false) {
		return false;
	}

	// If there is already geometry, we first unbind it.
	// In the default PointCloud render mode, we use GL_STATIC_DRAW. When
	// geometry changes, we clear buffers and rebind the geometry. Note that
	// this approach is slow. If the geomtry is changing per frame, consider
	// implementing a new ShaderWrapper using GL_STREAM_DRAW, and replace
	// UnbindGeometry() with Buffer Object Streaming mechanisms.
	UnbindGeometry();
	
	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			pointcloud.points_.size() * sizeof(Eigen::Vector3d),
			pointcloud.points_.data(),
			GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_color_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			pointcloud.colors_.size() * sizeof(Eigen::Vector3d),
			pointcloud.colors_.data(),
			GL_STATIC_DRAW);
	
	count_ = (GLsizei)pointcloud.points_.size();
	bound_ = true;
	
	return true;
}

void ShaderPointCloudDefault::UnbindGeometry()
{
	if (bound_) {
		glDeleteBuffers(1, &vertex_position_buffer_);
		glDeleteBuffers(1, &vertex_color_buffer_);
	}
	count_ = 0;
	bound_ = false;
}

bool ShaderPointCloudDefault::Render(const ViewControl &view)
{
	if (compiled_ == false || bound_ == false) {
		return false;
	}
	
	glUseProgram(program_);
	glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
	glEnableVertexAttribArray(vertex_position_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glVertexAttribPointer(vertex_position_, 3, GL_DOUBLE, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_color_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glVertexAttribPointer(vertex_color_, 3, GL_DOUBLE, GL_FALSE, 0, NULL);
	glDrawArrays(GL_POINTS, 0, count_);
	glDisableVertexAttribArray(vertex_position_);
	glDisableVertexAttribArray(vertex_color_);
	
	return true;
}

}

}	// namespace three
