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

#include "SimpleShader.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/LineSet.h>
#include <Core/Geometry/TriangleMesh.h>
#include <Visualization/Shader/Shader.h>
#include <Visualization/Utility/ColorMap.h>

namespace three{

namespace glsl {

bool SimpleShader::Compile()
{
	if (CompileShaders(SimpleVertexShader, NULL,
			SimpleFragmentShader) == false) {
		PrintShaderWarning("Compiling shaders failed.");
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
		const RenderOption &option, const ViewControl &view)
{
	// If there is already geometry, we first unbind it.
	// We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
	// rebind the geometry. Note that this approach is slow. If the geometry is
	// changing per frame, consider implementing a new ShaderWrapper using
	// GL_STREAM_DRAW, and replace InvalidateGeometry() with Buffer Object
	// Streaming mechanisms.
	UnbindGeometry();

	// Prepare data to be passed to GPU
	std::vector<Eigen::Vector3f> points;
	std::vector<Eigen::Vector3f> colors;
	if (PrepareBinding(geometry, option, view, points, colors) == false) {
		PrintShaderWarning("Binding failed when preparing data.");
		return false;
	}

	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
			points.data(), GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_color_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f),
			colors.data(), GL_STATIC_DRAW);
	bound_ = true;
	return true;
}

bool SimpleShader::RenderGeometry(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view)
{
	if (PrepareRendering(geometry, option, view) == false) {
		PrintShaderWarning("Rendering failed during preparation.");
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
		const RenderOption &option, const ViewControl &view)
{
	if (geometry.GetGeometryType() !=
			Geometry::GeometryType::PointCloud) {
		PrintShaderWarning("Rendering type is not PointCloud.");
		return false;
	}
	glPointSize(GLfloat(option.point_size_));
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	return true;
}

bool SimpleShaderForPointCloud::PrepareBinding(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view,
		std::vector<Eigen::Vector3f> &points,
		std::vector<Eigen::Vector3f> &colors)
{
	if (geometry.GetGeometryType() !=
			Geometry::GeometryType::PointCloud) {
		PrintShaderWarning("Rendering type is not PointCloud.");
		return false;
	}
	const PointCloud &pointcloud = (const PointCloud &)geometry;
	if (pointcloud.HasPoints() == false) {
		PrintShaderWarning("Binding failed with empty pointcloud.");
		return false;
	}
	const ColorMap &global_color_map = *GetGlobalColorMap();
	points.resize(pointcloud.points_.size());
	colors.resize(pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		const auto &point = pointcloud.points_[i];
		points[i] = point.cast<float>();
		Eigen::Vector3d color;
		switch (option.point_color_option_) {
		case RenderOption::PointColorOption::XCoordinate:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetXPercentage(point(0)));
			break;
		case RenderOption::PointColorOption::YCoordinate:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetYPercentage(point(1)));
			break;
		case RenderOption::PointColorOption::ZCoordinate:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetZPercentage(point(2)));
			break;
		case RenderOption::PointColorOption::Color:
		case RenderOption::PointColorOption::Default:
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

bool SimpleShaderForLineSet::PrepareRendering(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view)
{
	if (geometry.GetGeometryType() !=
			Geometry::GeometryType::LineSet) {
		PrintShaderWarning("Rendering type is not LineSet.");
		return false;
	}
	glLineWidth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	return true;
}

bool SimpleShaderForLineSet::PrepareBinding(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view,
		std::vector<Eigen::Vector3f> &points,
		std::vector<Eigen::Vector3f> &colors)
{
	if (geometry.GetGeometryType() !=
			Geometry::GeometryType::LineSet) {
		PrintShaderWarning("Rendering type is not LineSet.");
		return false;
	}
	const LineSet &lineset = (const LineSet &)geometry;
	if (lineset.HasLines() == false) {
		PrintShaderWarning("Binding failed with empty LineSet.");
		return false;
	}
	points.resize(lineset.lines_.size() * 2);
	colors.resize(lineset.lines_.size() * 2);
	for (size_t i = 0; i < lineset.lines_.size(); i++) {
		const auto point_pair = lineset.GetLineCoordinate(i);
		points[i * 2] = point_pair.first.cast<float>();
		points[i * 2 + 1] = point_pair.second.cast<float>();
		Eigen::Vector3d color;
		if (lineset.HasColors()) {
			color = lineset.colors_[i];
		} else {
			color = Eigen::Vector3d::Zero();
		}
		colors[i * 2] = colors[i * 2 + 1] = color.cast<float>();
	}
	draw_arrays_mode_ = GL_LINES;
	draw_arrays_size_ = GLsizei(points.size());
	return true;
}

bool SimpleShaderForTriangleMesh::PrepareRendering(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view)
{
	if (geometry.GetGeometryType() !=
			Geometry::GeometryType::TriangleMesh) {
		PrintShaderWarning("Rendering type is not TriangleMesh.");
		return false;
	}
	if (option.mesh_show_back_face_) {
		glDisable(GL_CULL_FACE);
	} else {
		glEnable(GL_CULL_FACE);
	}
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (option.mesh_show_wireframe_) {
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.0, 1.0);
	} else {
		glDisable(GL_POLYGON_OFFSET_FILL);
	}
	return true;
}

bool SimpleShaderForTriangleMesh::PrepareBinding(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view,
		std::vector<Eigen::Vector3f> &points,
		std::vector<Eigen::Vector3f> &colors)
{
	if (geometry.GetGeometryType() !=
			Geometry::GeometryType::TriangleMesh) {
		PrintShaderWarning("Rendering type is not TriangleMesh.");
		return false;
	}
	const TriangleMesh &mesh = (const TriangleMesh &)geometry;
	if (mesh.HasTriangles() == false) {
		PrintShaderWarning("Binding failed with empty triangle mesh.");
		return false;
	}
	const ColorMap &global_color_map = *GetGlobalColorMap();
	points.resize(mesh.triangles_.size() * 3);
	colors.resize(mesh.triangles_.size() * 3);

	for (size_t i = 0; i < mesh.triangles_.size(); i++) {
		const auto &triangle = mesh.triangles_[i];
		for (size_t j = 0; j < 3; j++) {
			size_t idx = i * 3 + j;
			size_t vi = triangle(j);
			const auto &vertex = mesh.vertices_[vi];
			points[idx] = vertex.cast<float>();

			Eigen::Vector3d color;
			switch (option.mesh_color_option_) {
			case RenderOption::MeshColorOption::XCoordinate:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetXPercentage(vertex(0)));
				break;
			case RenderOption::MeshColorOption::YCoordinate:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetYPercentage(vertex(1)));
				break;
			case RenderOption::MeshColorOption::ZCoordinate:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetZPercentage(vertex(2)));
				break;
			case RenderOption::MeshColorOption::Color:
				if (mesh.HasVertexColors()) {
					color = mesh.vertex_colors_[vi];
					break;
				}
			case RenderOption::MeshColorOption::Default:
			default:
				color = option.default_mesh_color_;
				break;
			}
			colors[idx] = color.cast<float>();
		}
	}
	draw_arrays_mode_ = GL_TRIANGLES;
	draw_arrays_size_ = GLsizei(points.size());
	return true;
}

}	// namespace three::glsl

}	// namespace three
