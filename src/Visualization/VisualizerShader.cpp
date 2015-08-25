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

#include "Visualizer.h"
#include "Shader.h"

namespace three{
	
bool Visualizer::CompileShaders()
{
	shaders_.vertex_shader_pointcloud_default =
			glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_default, 1,
			&glsl::PointCloudVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_default);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_default) == false) {
		return false;
	}
	
	shaders_.vertex_shader_pointcloud_x = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_x, 1,
			&glsl::PointCloudXVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_x);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_x) == false) {
		return false;
	}

	shaders_.vertex_shader_pointcloud_y = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_y, 1,
			&glsl::PointCloudYVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_y);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_y) == false) {
		return false;
	}

	shaders_.vertex_shader_pointcloud_z = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_z, 1,
			&glsl::PointCloudZVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_z);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_z) == false) {
		return false;
	}
	
	shaders_.fragment_shader_pointcloud_default =
			glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(shaders_.fragment_shader_pointcloud_default, 1,
			&glsl::PointCloudFragmentShader, NULL);
	glCompileShader(shaders_.fragment_shader_pointcloud_default);
	if (ValidateShader(shaders_.fragment_shader_pointcloud_default) == false) {
		return false;
	}
	
	return true;
}
	
bool Visualizer::ValidateShader(GLuint shader_index)
{
	GLint result = GL_FALSE;
	int info_log_length;
	glGetShaderiv(shader_index, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		glGetShaderiv(shader_index, GL_INFO_LOG_LENGTH, &info_log_length);
		if (info_log_length > 0) {
			std::vector<char> error_message(info_log_length + 1);
			glGetShaderInfoLog(shader_index, info_log_length, NULL,
					&error_message[0]);
			PrintError("Shader error: %s\n", &error_message[0]);
		}
		return false;
	}
	return true;
}

}	// namespace three
