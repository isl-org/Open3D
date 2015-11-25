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

#include "SimpleBlackShader.h"

#include "Shader.h"
#include "ColorMap.h"

namespace three{

namespace glsl {

bool SimpleBlackShader::Compile()
{
	if (CompileShaders(
			SimpleBlackVertexShader,
			NULL,
			SimpleBlackFragmentShader) == false)
	{
		PrintWarning("[SimpleBlackShader] Compiling shaders failed.\n");
		return false;
	}
	
	vertex_position_ = glGetAttribLocation(program_, "vertex_position");
	MVP_ = glGetUniformLocation(program_, "MVP");

	return true;
}

void SimpleBlackShader::Release()
{
	UnbindGeometry();
	ReleaseProgram();
}

bool SimpleBlackShader::BindGeometry(
		const Geometry &geometry, 
		const RenderMode &mode,
		const ViewControl &view)
{
	// If there is already geometry, we first unbind it.
	// We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
	// rebind the geometry. Note that this approach is slow. If the geomtry is
	// changing per frame, consider implementing a new ShaderWrapper using
	// GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
	// Streaming mechanisms.
	UnbindGeometry();

	// Prepare data to be passed to GPU
	std::vector<Eigen::Vector3f> points;
	if (PrepareBinding(geometry, mode, view, points) == false) {
		PrintWarning("[SimpleBlackShader] Binding failed when preparing data.\n");
		return false;
	}
	
	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			points.size() * sizeof(Eigen::Vector3f),
			points.data(),
			GL_STATIC_DRAW);
	
	bound_ = true;	
	return true;
}

void SimpleBlackShader::UnbindGeometry()
{
	if (bound_) {
		glDeleteBuffers(1, &vertex_position_buffer_);
		bound_ = false;
	}
}

bool SimpleBlackShader::Render(
		const Geometry &geometry,
		const RenderMode &mode,
		const ViewControl &view)
{
	if (compiled_ == false) {
		Compile();
	}
	if (bound_ == false) {
		BindGeometry(geometry, mode, view);
	}
	if (compiled_ == false || bound_ == false) {
		PrintWarning("[SimpleBlackShader] Something is wrong in compiling or binding.\n");
		return false;
	}
	if (PrepareRendering(geometry, mode, view) == false) {
		PrintWarning("[SimpleBlackShader] Rendering failed during preparation.\n");
		return false;
	}
	glUseProgram(program_);
	glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
	glEnableVertexAttribArray(vertex_position_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
	glDisableVertexAttribArray(vertex_position_);
	return true;
}

bool SimpleBlackShaderForPointCloudNormal::PrepareRendering(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_POINTCLOUD) {
		PrintWarning("[SimpleBlackShaderForPointCloudNormal] Rendering type is not PointCloud.\n");
		return false;
	}
	return true;
}

bool SimpleBlackShaderForPointCloudNormal::PrepareBinding(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view,
			std::vector<Eigen::Vector3f> &points)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_POINTCLOUD) {
		PrintWarning("[SimpleBlackShaderForPointCloudNormal] Binding type is not PointCloud.\n");
		return false;
	}
	const PointCloud &pointcloud = (const PointCloud &)geometry;
	if (pointcloud.HasPoints() == false) {
		PrintWarning("[SimpleBlackShaderForPointCloudNormal] Binding failed with empty pointcloud.\n");
		return false;
	}
	const auto &rendermode = (const PointCloudRenderMode &)mode;
	points.resize(pointcloud.points_.size() * 2);
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		double line_length = rendermode.GetPointSize() *
				0.01 * view.GetBoundingBox().GetSize();
		const Eigen::Vector3f default_normal_color(0.1f, 0.1f, 0.1f);
		for (size_t i = 0; i < pointcloud.points_.size(); i++) {
			const auto &point = pointcloud.points_[i];
			const auto &normal = pointcloud.normals_[i];
			points[i * 2] = point.cast<float>();
			points[i * 2 + 1] = (point + normal * line_length).cast<float>();
		}
	}
	draw_arrays_mode_ = GL_LINES;
	draw_arrays_size_ = pointcloud.points_.size() * 2;
	return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareRendering(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_TRIANGLEMESH ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_TRIANGLEMESH) {
		PrintWarning("[SimpleBlackShaderForTriangleMeshWireFrame] Rendering type is not PointCloud.\n");
		return false;
	}
	return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareBinding(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view,
			std::vector<Eigen::Vector3f> &points)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_POINTCLOUD) {
		PrintWarning("[SimpleBlackShaderForTriangleMeshWireFrame] Binding type is not PointCloud.\n");
		return false;
	}
	const PointCloud &pointcloud = (const PointCloud &)geometry;
	if (pointcloud.HasPoints() == false) {
		PrintWarning("[SimpleBlackShaderForTriangleMeshWireFrame] Binding failed with empty pointcloud.\n");
		return false;
	}
	const auto &rendermode = (const PointCloudRenderMode &)mode;
	points.resize(pointcloud.points_.size() * 2);
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		double line_length = rendermode.GetPointSize() *
				0.01 * view.GetBoundingBox().GetSize();
		const Eigen::Vector3f default_normal_color(0.1f, 0.1f, 0.1f);
		for (size_t i = 0; i < pointcloud.points_.size(); i++) {
			const auto &point = pointcloud.points_[i];
			const auto &normal = pointcloud.normals_[i];
			points[i * 2] = point.cast<float>();
			points[i * 2 + 1] = (point + normal * line_length).cast<float>();
		}
	}
	draw_arrays_mode_ = GL_LINES;
	draw_arrays_size_ = pointcloud.points_.size() * 2;
	return true;
}

}

}	// namespace three
