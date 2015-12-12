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

#include "PhongShader.h"

#include "Shader.h"
#include "ColorMap.h"

namespace three{

namespace glsl {

bool PhongShader::Compile()
{
	if (CompileShaders(
			PhongVertexShader,
			NULL,
			PhongFragmentShader) == false)
	{
		PrintWarning("[%s] Compiling shaders failed.\n",
				GetShaderName().c_str());
		return false;
	}
	
	vertex_position_ = glGetAttribLocation(program_, "vertex_position");
	vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
	vertex_color_ = glGetAttribLocation(program_, "vertex_color");
	MVP_ = glGetUniformLocation(program_, "MVP");
	V_ = glGetUniformLocation(program_, "V");
	M_ = glGetUniformLocation(program_, "M");
	light_position_world_ = 
			glGetUniformLocation(program_, "light_position_world_4");
	light_color_ = glGetUniformLocation(program_, "light_color_4");
	light_power_ = glGetUniformLocation(program_, "light_power_4");
	light_ambient_ = glGetUniformLocation(program_, "light_ambient");

	return true;
}

void PhongShader::Release()
{
	UnbindGeometry();
	ReleaseProgram();
}

bool PhongShader::BindGeometry(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view)
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
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector3f> colors;
	if (PrepareBinding(geometry, option, view, points, normals, colors) ==
			false) {
		PrintWarning("[%s] Binding failed when preparing data.\n",
				GetShaderName().c_str());
		return false;
	}
	
	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
			points.data(), GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_normal_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
	glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(Eigen::Vector3f),
			normals.data(), GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_color_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f),
			colors.data(), GL_STATIC_DRAW);
	
	bound_ = true;	
	return true;
}
	
bool PhongShader::RenderGeometry(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view)
{
	if (PrepareRendering(geometry, option, view) == false) {
		PrintWarning("[%s] Rendering failed during preparation.\n",
				GetShaderName().c_str());
		return false;
	}
	
	glUseProgram(program_);
	glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
	glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
	glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
	glUniformMatrix4fv(light_position_world_, 1, GL_FALSE, 
			light_position_world_data_.data());
	glUniformMatrix4fv(light_color_, 1, GL_FALSE, light_color_data_.data());
	glUniform4fv(light_power_, 1, light_power_data_.data());
	glUniform4fv(light_ambient_, 1, light_ambient_data_.data());
	glEnableVertexAttribArray(vertex_position_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_normal_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
	glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_color_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
	glDisableVertexAttribArray(vertex_position_);
	glDisableVertexAttribArray(vertex_normal_);
	glDisableVertexAttribArray(vertex_color_);

	return true;
}

void PhongShader::UnbindGeometry()
{
	if (bound_) {
		glDeleteBuffers(1, &vertex_position_buffer_);
		glDeleteBuffers(1, &vertex_normal_buffer_);
		glDeleteBuffers(1, &vertex_color_buffer_);
		bound_ = false;
	}
}

void PhongShader::SetBoundingBoxLight(const ViewControl &view,
		bool light_on/* = true*/)
{
	const auto &box = view.GetBoundingBox();
	light_position_world_data_.setOnes();
	light_position_world_data_.block<3, 1>(0, 0) = 
			(box.GetCenter() + Eigen::Vector3d(2, 2, 2) * box.GetSize()).
			cast<float>();
	light_position_world_data_.block<3, 1>(0, 1) =
			(box.GetCenter() + Eigen::Vector3d(2, -2, -2) * box.GetSize()).
			cast<float>();
	light_position_world_data_.block<3, 1>(0, 2) =
			(box.GetCenter() + Eigen::Vector3d(-2, 2, -2) * box.GetSize()).
			cast<float>();
	light_position_world_data_.block<3, 1>(0, 3) =
			(box.GetCenter() + Eigen::Vector3d(-2, -2, 2) * box.GetSize()).
			cast<float>();

	light_color_data_.setOnes();

	if (light_on) {
		light_power_data_ = GLHelper::GLVector4f(0.6f, 0.6f, 0.6f, 0.6f);
		light_ambient_data_ = GLHelper::GLVector4f(0.1f, 0.1f, 0.1f, 1.0f);
	} else {
		light_power_data_ = GLHelper::GLVector4f::Zero();
		light_ambient_data_ = GLHelper::GLVector4f(1.0f, 1.0f, 1.0f, 1.0f);
	}
}

bool PhongShaderForPointCloud::PrepareRendering(const Geometry &geometry,
		const RenderOption &option,const ViewControl &view)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD) {
		PrintWarning("[%s] Rendering type is not PointCloud.\n",
				GetShaderName().c_str());
		return false;
	}
	glDepthFunc(GL_LESS);
	glPointSize(GLfloat(option.GetPointSize()));
	SetBoundingBoxLight(view, option.IsLightOn());
	return true;
}

bool PhongShaderForPointCloud::PrepareBinding(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view,
		std::vector<Eigen::Vector3f> &points,
		std::vector<Eigen::Vector3f> &normals,
		std::vector<Eigen::Vector3f> &colors)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD) {
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
	if (pointcloud.HasNormals() == false) {
		PrintWarning("[%s] Binding failed with pointcloud with no normals.\n",
				GetShaderName().c_str());
		return false;
	}
	const ColorMap &global_color_map = *GetGlobalColorMap();
	points.resize(pointcloud.points_.size());
	normals.resize(pointcloud.points_.size());
	colors.resize(pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		const auto &point = pointcloud.points_[i];
		const auto &normal = pointcloud.normals_[i];
		points[i] = point.cast<float>();
		normals[i] = normal.cast<float>();
		Eigen::Vector3d color;
		switch (option.GetPointColorOption()) {
		case RenderOption::POINTCOLOR_X:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetXPercentage(point(0)));
			break;
		case RenderOption::POINTCOLOR_Y:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetYPercentage(point(1)));
			break;
		case RenderOption::POINTCOLOR_Z:
			color = global_color_map.GetColor(
					view.GetBoundingBox().GetZPercentage(point(2)));
			break;
		case RenderOption::POINTCOLOR_COLOR:
		case RenderOption::POINTCOLOR_DEFAULT:
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

bool PhongShaderForTriangleMesh::PrepareRendering(const Geometry &geometry,
		const RenderOption &option,const ViewControl &view)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_TRIANGLEMESH) {
		PrintWarning("[%s] Rendering type is not TriangleMesh.\n",
				GetShaderName().c_str());
		return false;
	}
	SetBoundingBoxLight(view, option.IsLightOn());
	if (option.IsMeshBackFaceShown()) {
		glDisable(GL_CULL_FACE);
	} else {
		glEnable(GL_CULL_FACE);
	}
	glDepthFunc(GL_LESS);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.0, 1.0);
	return true;
}

bool PhongShaderForTriangleMesh::PrepareBinding(const Geometry &geometry,
		const RenderOption &option, const ViewControl &view,
		std::vector<Eigen::Vector3f> &points,
		std::vector<Eigen::Vector3f> &normals,
		std::vector<Eigen::Vector3f> &colors)
{
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_TRIANGLEMESH) {
		PrintWarning("[%s] Binding type is not TriangleMesh.\n",
				GetShaderName().c_str());
		return false;
	}
	const TriangleMesh &mesh = (const TriangleMesh &)geometry;
	if (mesh.HasTriangles() == false) {
		PrintWarning("[%s] Binding failed with empty triangle mesh.\n",
				GetShaderName().c_str());
		return false;
	}
	if (mesh.HasTriangleNormals() == false || mesh.HasVertexNormals() == false)
	{
		PrintWarning("[%s] Binding failed because mesh has no normals.\n",
				GetShaderName().c_str());
		PrintWarning("[%s] Call ComputeVertexNormals() before binding.\n",
				GetShaderName().c_str());
		return false;
	}
	const ColorMap &global_color_map = *GetGlobalColorMap();
	points.resize(mesh.triangles_.size() * 3);
	normals.resize(mesh.triangles_.size() * 3);
	colors.resize(mesh.triangles_.size() * 3);
	
	for (size_t i = 0; i < mesh.triangles_.size(); i++) {
		const auto &triangle = mesh.triangles_[i];
		for (size_t j = 0; j < 3; j++) {
			size_t idx = i * 3 + j;
			size_t vi = triangle(j);
			const auto &vertex = mesh.vertices_[vi];
			points[idx] = vertex.cast<float>();

			Eigen::Vector3d color;
			switch (option.GetMeshColorOption()) {
			case RenderOption::TRIANGLEMESH_X:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetXPercentage(vertex(0)));
				break;
			case RenderOption::TRIANGLEMESH_Y:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetYPercentage(vertex(1)));
				break;
			case RenderOption::TRIANGLEMESH_Z:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetZPercentage(vertex(2)));
				break;
			case RenderOption::TRIANGLEMESH_COLOR:
				if (mesh.HasVertexColors()) {
					color = mesh.vertex_colors_[vi];
					break;
				}
			case RenderOption::TRIANGLEMESH_DEFAULT:
			default:
				color = RenderOption::DEFAULT_MESH_COLOR;
				break;
			}
			colors[idx] = color.cast<float>();

			if (option.GetMeshShadeOption() ==
					RenderOption::MESHSHADE_FLATSHADE) {
				normals[idx] = mesh.triangle_normals_[i].cast<float>();
			} else {
				normals[idx] = mesh.vertex_normals_[vi].cast<float>();
			}
		}
	}
	draw_arrays_mode_ = GL_TRIANGLES;
	draw_arrays_size_ = GLsizei(points.size());
	return true;
}

}	// namespace glsl

}	// namespace three
