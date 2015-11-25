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

#include "ShaderTriangleMesh.h"

#include "Shader.h"
#include "ColorMap.h"

namespace three{

namespace glsl {

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
	vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
	vertex_color_ = glGetAttribLocation(program_, "vertex_color");
	MVP_ = glGetUniformLocation(program_, "MVP");
	V_ = glGetUniformLocation(program_, "V");
	M_ = glGetUniformLocation(program_, "M");
	light_position_world_ = 
			glGetUniformLocation(program_, "light_position_world_4");
	light_color_ = glGetUniformLocation(program_, "light_color_4");
	light_power_ = glGetUniformLocation(program_, "light_power_4");

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
		PrintWarning("[ShaderTriangleMeshDefault] Binding failed with wrong binding type.\n");
		return false;
	}
	const TriangleMesh &mesh = (const TriangleMesh &)geometry;
	if (mesh.HasTriangles() == false) {
		PrintWarning("[ShaderTriangleMeshDefault] Binding failed with empty mesh.\n");
		return false;
	}
	if (mesh.HasTriangleNormals() == false || 
			mesh.HasVertexNormals() == false) {
		PrintWarning("[ShaderTriangleMeshDefault] Binding failed because mesh has no normals.\n");
		PrintWarning("[ShaderTriangleMeshDefault] Call ComputeVertexNormals() before binding.\n");
		return false;
	}
	const auto &mesh_render_mode = (const TriangleMeshRenderMode &)mode;
	const ColorMap &global_color_map = *GetGlobalColorMap();

	// If there is already geometry, we first unbind it.
	// In the default PointCloud render mode, we use GL_STATIC_DRAW. When
	// geometry changes, we clear buffers and rebind the geometry. Note that
	// this approach is slow. If the geomtry is changing per frame, consider
	// implementing a new ShaderWrapper using GL_STREAM_DRAW, and replace
	// UnbindGeometry() with Buffer Object Streaming mechanisms.
	UnbindGeometry();

	// Copy data to renderer's own container. A double-to-float cast is
	// performed for performance reason.
	vertex_num_ = (GLsizei)mesh.triangles_.size() * 3;
	std::vector<Eigen::Vector3f> vertices_copy(vertex_num_);
	std::vector<Eigen::Vector3f> colors_copy(vertex_num_);
	std::vector<Eigen::Vector3f> normals_copy(vertex_num_);

	for (size_t i = 0; i < mesh.triangles_.size(); i++) {
		const auto &triangle = mesh.triangles_[i];
		for (size_t j = 0; j < 3; j++) {
			size_t idx = i * 3 + j;
			size_t vi = triangle(j);
			const auto &vertex = mesh.vertices_[vi];
			vertices_copy[idx] = vertex.cast<float>();

			Eigen::Vector3d color;
			switch (mesh_render_mode.GetMeshColorOption()) {
			case TriangleMeshRenderMode::TRIANGLEMESH_X:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetXPercentage(vertex(0)));
				break;
			case TriangleMeshRenderMode::TRIANGLEMESH_Y:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetYPercentage(vertex(1)));
				break;
			case TriangleMeshRenderMode::TRIANGLEMESH_Z:
				color = global_color_map.GetColor(
						view.GetBoundingBox().GetZPercentage(vertex(2)));
				break;
			case TriangleMeshRenderMode::TRIANGLEMESH_COLOR:
				if (mesh.HasVertexColors()) {
					color = mesh.vertex_colors_[vi] * 10.0;
					break;
				}
			case TriangleMeshRenderMode::TRIANGLEMESH_DEFAULT:
			default:
				color = default_color_;
				break;
			}
			colors_copy[idx] = color.cast<float>();

			if (mesh_render_mode.GetMeshShadeOption() == 
					TriangleMeshRenderMode::MESHSHADE_FLATSHADE) {
				normals_copy[idx] = mesh.triangle_normals_[i].cast<float>();
			} else {
				normals_copy[idx] = mesh.vertex_normals_[vi].cast<float>();
			}
		}
	}

	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			vertices_copy.size() * sizeof(Eigen::Vector3f),
			vertices_copy.data(),
			GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_normal_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			normals_copy.size() * sizeof(Eigen::Vector3f),
			normals_copy.data(),
			GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_color_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			colors_copy.size() * sizeof(Eigen::Vector3f),
			colors_copy.data(),
			GL_STATIC_DRAW);
	
	lights_on_ = !(mesh_render_mode.GetMeshColorOption() == 
			TriangleMeshRenderMode::TRIANGLEMESH_COLOR && 
			mesh.HasVertexColors());
	bound_ = true;	
	return true;
}

void ShaderTriangleMeshDefault::UnbindGeometry()
{
	if (bound_) {
		glDeleteBuffers(1, &vertex_position_buffer_);
		glDeleteBuffers(1, &vertex_normal_buffer_);
		glDeleteBuffers(1, &vertex_color_buffer_);
		bound_ = false;
	}
}

void ShaderTriangleMeshDefault::SetLight(
		const TriangleMeshRenderMode &mode, 
		const ViewControl &view)
{
	if (mode.GetLightingOption() == TriangleMeshRenderMode::LIGHTING_DEFAULT) {
		const auto &box = view.GetBoundingBox();
		light_position_world_data_.setOnes();
		light_position_world_data_.block<3, 1>(0, 0) = 
				(box.GetCenter() + Eigen::Vector3d(2, 2, 2) * box.GetSize()).
				cast<float>();
		light_color_data_.block<4, 1>(0, 0) = 
				GLHelper::GLVector4f(1.0, 1.0, 1.0, 1.0);
		light_power_data_(0) = lights_on_ ? 0.5f : 0.0f;
		light_position_world_data_.block<3, 1>(0, 1) = 
				(box.GetCenter() + Eigen::Vector3d(2, -2, -2) * box.GetSize()).
				cast<float>();
		light_color_data_.block<4, 1>(0, 1) = 
				GLHelper::GLVector4f(1.0, 1.0, 1.0, 1.0);
		light_power_data_(1) = lights_on_ ? 0.5f : 0.0f;
		light_position_world_data_.block<3, 1>(0, 2) = 
				(box.GetCenter() + Eigen::Vector3d(-2, 2, -2) * box.GetSize()).
				cast<float>();
		light_color_data_.block<4, 1>(0, 2) = 
				GLHelper::GLVector4f(1.0, 1.0, 1.0, 1.0);
		light_power_data_(2) = lights_on_ ? 0.5f : 0.0f;
		light_position_world_data_.block<3, 1>(0, 3) = 
				(box.GetCenter() + Eigen::Vector3d(-2, -2, 2) * box.GetSize()).
				cast<float>();
		light_color_data_.block<4, 1>(0, 3) = 
				GLHelper::GLVector4f(1.0, 1.0, 1.0, 1.0);
		light_power_data_(3) = lights_on_ ? 0.7f : 0.0f;
	}
}

bool ShaderTriangleMeshDefault::Render(
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
	if (compiled_ == false || bound_ == false || vertex_num_ <= 0 ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_TRIANGLEMESH) {
		return false;
	}
	
	const auto &rendermode = (const TriangleMeshRenderMode &)mode;
	SetLight(rendermode, view);
	if (rendermode.IsBackFaceShown()) {
		glDisable(GL_CULL_FACE);
	} else {
		glEnable(GL_CULL_FACE);
	}

	glUseProgram(program_);
	glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
	glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
	glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
	glUniformMatrix4fv(light_position_world_, 1, GL_FALSE, 
			light_position_world_data_.data());
	glUniformMatrix4fv(light_color_, 1, GL_FALSE, light_color_data_.data());
	glUniform4fv(light_power_, 1, light_power_data_.data());
	glEnableVertexAttribArray(vertex_position_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_normal_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
	glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_color_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glDrawArrays(GL_TRIANGLES, 0, vertex_num_);
	glDisableVertexAttribArray(vertex_position_);
	glDisableVertexAttribArray(vertex_normal_);
	glDisableVertexAttribArray(vertex_color_);

	return true;
}

}

}	// namespace three
