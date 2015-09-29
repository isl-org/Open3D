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

#include "ShaderImage.h"

#include "Shader.h"
#include "ColorMap.h"

namespace three{

namespace glsl {

bool ShaderImageDefault::Compile()
{
	if (CompileShaders(
			ImageVertexShader,
			NULL,
			ImageFragmentShader) == false)
	{
		PrintWarning("[ShaderImageDefault] Compiling shaders failed.\n");
		return false;
	}
	
	vertex_position_ = glGetAttribLocation(program_, "vertex_position");
	vertex_UV_ = glGetAttribLocation(program_, "vertex_UV");
	image_texture_ = glGetUniformLocation(program_, "image_texture");

	return true;
}

void ShaderImageDefault::Release()
{
	UnbindGeometry();
	ReleaseProgram();
}

bool ShaderImageDefault::BindGeometry(
		const Geometry &geometry, 
		const RenderMode &mode,
		const ViewControl &view)
{
	// Sanity check to see if this geometry is worth binding.
	if (geometry.GetGeometryType() != Geometry::GEOMETRY_IMAGE ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_IMAGE) {
		PrintWarning("[ShaderImageDefault] Binding failed with wrong binding type.\n");
		return false;
	}
	const Image &image = (const Image &)geometry;
	if (image.HasData() == false) {
		PrintWarning("[ShaderImageDefault] Binding failed with empty image.\n");
		return false;
	}
	const auto &image_render_mode = (const ImageRenderMode &)mode;

	// If there is already geometry, we first unbind it.
	// In the default PointCloud render mode, we use GL_STATIC_DRAW. When
	// geometry changes, we clear buffers and rebind the geometry. Note that
	// this approach is slow. If the geomtry is changing per frame, consider
	// implementing a new ShaderWrapper using GL_STREAM_DRAW, and replace
	// UnbindGeometry() with Buffer Object Streaming mechanisms.
	UnbindGeometry();

	// Copy data to renderer's own container. A double-to-float cast is
	// performed for performance reason.
	
	/*
	// Create buffers and bind the geometry
	glGenBuffers(1, &vertex_position_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			points_copy.size() * sizeof(Eigen::Vector3f),
			points_copy.data(),
			GL_STATIC_DRAW);
	glGenBuffers(1, &vertex_color_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glBufferData(GL_ARRAY_BUFFER,
			colors_copy.size() * sizeof(Eigen::Vector3f),
			colors_copy.data(),
			GL_STATIC_DRAW);
	*/

	bound_ = true;	
	return true;
}

void ShaderImageDefault::UnbindGeometry()
{
	if (bound_) {
		glDeleteBuffers(1, &vertex_position_buffer_);
		glDeleteBuffers(1, &vertex_UV_buffer_);
		glDeleteTextures(1, &image_texture_buffer_);
		bound_ = false;
	}
}

bool ShaderImageDefault::Render(
		const RenderMode &mode, 
		const ViewControl &view)
{
	if (compiled_ == false || bound_ == false ||
			mode.GetRenderModeType() != RenderMode::RENDERMODE_IMAGE) {
		return false;
	}
	
	const auto &rendermode = (const ImageRenderMode &)mode;

	/*
	glUseProgram(program_);
	glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
	glEnableVertexAttribArray(vertex_position_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
	glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertex_color_);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
	glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glDrawArrays(GL_POINTS, 0, point_num_);
	if (show_normal_) {
		glDrawArrays(GL_LINES, point_num_, point_num_ * 2);
	}
	glDisableVertexAttribArray(vertex_position_);
	glDisableVertexAttribArray(vertex_color_);
	*/
	
	return true;
}

}

}	// namespace three
