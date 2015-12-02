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

#include "SimpleShader.h"

#include "Shader.h"
#include "ColorMap.h"

namespace three{

namespace glsl {

bool SimpleShader::Compile()
{
	if (CompileShaders(
			SimpleVertexShader,
			NULL,
			SimpleFragmentShader) == false)
	{
		PrintWarning("[%s] Compiling shaders failed.\n",
				GetShaderName().c_str());
		return false;
	}
	
	vertex_position_ = glGetAttribLocation(program_, "vertex_position");
	vertex_color_ = glGetAttribLocation(program_, "vertex_color");
	MVP_ = glGetUniformLocation(program_, "MVP");

	return true;
}

void SimpleShader::Release()
{
	UnbindGeometry();
	ReleaseProgram();
}

bool SimpleShader::BindGeometry(const Geometry &geometry,
		const RenderMode &mode, const ViewControl &view)
{
	// If there is already geometry, we first unbind it.
	// We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
	// rebind the geometry. Note that this approach is slow. If the geomtry is
	// changing per frame, consider implementing a new ShaderWrapper using
	// GL_STREAM_DRAW, and replace InvalidateGeometry() with Buffer Object
	// Streaming mechanisms.
	UnbindGeometry();

	// Prepare data to be passed to GPU
	std::vector<Eigen::Vector3f> points;
	std::vector<Eigen::Vector3f> colors;
	if (PrepareBinding(geometry, mode, view, points, colors) == false) {
		PrintWarning("[%s] Binding failed when preparing data.\n",
				GetShaderName().c_str());
		return false;
	}
	
	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			points.size() * sizeof(Eigen::Vector3f),
			points.data(),
			GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_color_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			colors.size() * sizeof(Eigen::Vector3f),
			colors.data(),
			GL_STATIC_DRAW);
	
	bound_ = true;	
	return true;
}

bool SimpleShader::RenderGeometry(const Geometry &geometry,
		const RenderMode &mode, const ViewControl &view)
{
	if (PrepareRendering(geometry, mode, view) == false) {
		PrintWarning("[%s] Rendering failed during preparation.\n",
				GetShaderName().c_str());
		return false;
	}
	glUseProgram(program_);
	glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
	glEnableVertexAttribArray(vertex_position_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_color_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
	glDisableVertexAttribArray(vertex_position_);
	glDisableVertexAttribArray(vertex_color_);
	return true;
}

void SimpleShader::UnbindGeometry()
{
	if (bound_) {
		glDeleteBuffers(1, &vertex_position_buffer_);
		glDeleteBuffers(1, &vertex_color_buffer_);
		bound_ = false;
	}
}

bool SimpleShaderForPointCloud::PrepareRendering(const Geometry &geometry,
			const RenderMode &mode, const ViewControl &view)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_POINTCLOUD) {
		PrintWarning("[%s] Rendering type is not PointCloud.\n",
				GetShaderName().c_str());
		return false;
	}
	const auto &rendermode = (const PointCloudRenderMode &)mode;
	glPointSize(GLfloat(rendermode.GetPointSize()));
	return true;
}

bool SimpleShaderForPointCloud::PrepareBinding(const Geometry &geometry,
			const RenderMode &mode, const ViewControl &view,
			std::vector<Eigen::Vector3f> &points,
			std::vector<Eigen::Vector3f> &colors)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_POINTCLOUD) {
		PrintWarning("[%s] Binding type is not PointCloud.\n",
				GetShaderName().c_str());
		return false;
	}
	const PointCloud &pointcloud = (const PointCloud &)geometry;
	if (pointcloud.HasPoints() == false) {
		PrintWarning("[%s] Binding failed with empty pointcloud.\n",
				GetShaderName().c_str());
		return false;
	}
	const auto &rendermode = (const PointCloudRenderMode &)mode;
	const ColorMap &global_color_map = *GetGlobalColorMap();
	points.resize(pointcloud.points_.size());
	colors.resize(pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		const auto &point = pointcloud.points_[i];
		points[i] = point.cast<float>();
		Eigen::Vector3d color;
		switch (rendermode.GetPointColorOption()) {
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
		colors[i] = color.cast<float>();
	}
	draw_arrays_mode_ = GL_POINTS;
	draw_arrays_size_ = GLsizei(points.size());
	return true;
}

}

}	// namespace three
