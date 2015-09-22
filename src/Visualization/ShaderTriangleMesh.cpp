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

#include "ShaderTriangleMesh.h"

#include "Shader.h"

namespace three{

namespace glsl {

ShaderTriangleMeshDefault::ShaderTriangleMeshDefault()
{
}

ShaderTriangleMeshDefault::~ShaderTriangleMeshDefault()
{
}

bool ShaderTriangleMeshDefault::Compile()
{
	if (CompileShaders(
			TriangleMeshVertexShader,
			NULL,
			TriangleMeshFragmentShader) == false)
	{
		return false;
	}
	
	vertex_position_ = glGetAttribLocation(program_, "vertex_position");
	vertex_color_ = glGetAttribLocation(program_, "vertex_color");
	MVP_ = glGetUniformLocation(program_, "MVP");

	return true;
}

void ShaderTriangleMeshDefault::Release()
{
	UnbindGeometry();
	ReleaseProgram();
}

bool ShaderTriangleMeshDefault::BindGeometry(
		const Geometry &geometry, 
		const RenderMode &mode,
		const ViewControl &view)
{
	// Sanity check to see if this geometry is worth binding.
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_TRIANGLEMESH ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_TRIANGLEMESH) {
		return false;
	}
	const TriangleMesh &mesh = (const TriangleMesh &)geometry;
    if (mesh.HasTriangles() == false) {
		return false;
	}
	//const PointCloudRenderMode &pointcloud_render_mode = 
		//	(const PointCloudRenderMode &)mode;

	// If there is already geometry, we first unbind it.
	// In the default PointCloud render mode, we use GL_STATIC_DRAW. When
	// geometry changes, we clear buffers and rebind the geometry. Note that
	// this approach is slow. If the geomtry is changing per frame, consider
	// implementing a new ShaderWrapper using GL_STREAM_DRAW, and replace
	// UnbindGeometry() with Buffer Object Streaming mechanisms.
	UnbindGeometry();

	/*
	// Copy data to renderer's own container. A double-to-float cast is
	// performed for performance reason.
	point_num_ = (GLsizei)pointcloud.points_.size();
	points_copy_.resize(pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		points_copy_[i] = pointcloud.points_[i].cast<float>();
	}
	colors_copy_.resize(pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		auto point = pointcloud.points_[i];
		Eigen::Vector3d color;
		switch (pointcloud_render_mode.GetPointColorOption()) {
		case PointCloudRenderMode::POINTCOLOR_X:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetXPercentage(point(0)));
			break;
		case PointCloudRenderMode::POINTCOLOR_Y:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetYPercentage(point(1)));
			break;
		case PointCloudRenderMode::POINTCOLOR_Z:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetZPercentage(point(2)));
			break;
		case PointCloudRenderMode::POINTCOLOR_COLOR:
		case PointCloudRenderMode::POINTCOLOR_DEFAULT:
		default:
			if (pointcloud.HasColors()) {
				color = pointcloud.colors_[i];
			} else {
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetZPercentage(point(2)));
			}
			break;
		}
		colors_copy_[i] = color.cast<float>();
	}

	// Set up normal if it is enabled
	if (pointcloud_render_mode.IsNormalShown() && pointcloud.HasNormals()) {
		show_normal_ = true;
		points_copy_.resize(point_num_ * 3);
		colors_copy_.resize(point_num_ * 3);
		double line_length = pointcloud_render_mode.GetPointSize() *
				0.01 * view.GetBoundingBox().GetSize();
		const Eigen::Vector3f default_normal_color(0.1f, 0.1f, 0.1f);
		for (size_t i = 0; i < pointcloud.points_.size(); i++) {
			auto point = pointcloud.points_[i];
			auto normal = pointcloud.normals_[i];
			points_copy_[point_num_ + i * 2] = point.cast<float>();
			points_copy_[point_num_ + i * 2 + 1] = 
					(point + normal * line_length).cast<float>();
			colors_copy_[point_num_ + i * 2] = default_normal_color;
			colors_copy_[point_num_ + i * 2 + 1] = default_normal_color;
		}
	} else {
		show_normal_ = false;
	}
	
	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			points_copy_.size() * sizeof(Eigen::Vector3f),
			points_copy_.data(),
			GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_color_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			colors_copy_.size() * sizeof(Eigen::Vector3f),
			colors_copy_.data(),
			GL_STATIC_DRAW);
	
	*/
	bound_ = true;	
	return true;
}

void ShaderTriangleMeshDefault::UnbindGeometry()
{
	if (bound_) {
		glDeleteBuffers(1, &vertex_position_buffer_);
		glDeleteBuffers(1, &vertex_color_buffer_);
		bound_ = false;
	}
}

bool ShaderTriangleMeshDefault::Render(
		const RenderMode &mode,
		const ViewControl &view)
{
	if (compiled_ == false || bound_ == false) {
		return false;
	}

	return true;
	
	glUseProgram(program_);
	glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
	glEnableVertexAttribArray(vertex_position_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_color_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	//glDrawArrays(GL_POINTS, 0, point_num_);
	glDisableVertexAttribArray(vertex_position_);
	glDisableVertexAttribArray(vertex_color_);
	
	return true;
}

}

}	// namespace three
