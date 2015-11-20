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

#include <GL/glew.h>
#include <Core/Core.h>

#include "ViewControl.h"
#include "RenderMode.h"

namespace three {

namespace glsl {
	
class ShaderWrapper {
public:
	ShaderWrapper() {}
	virtual ~ShaderWrapper() {}
	ShaderWrapper(const ShaderWrapper &) = delete;
	ShaderWrapper &operator=(const ShaderWrapper &) = delete;
	
public:
	virtual bool Compile() = 0;
	virtual bool Render(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view) = 0;
	virtual void Release() = 0;
	void UpdateGeometry() { bound_ = false; }

protected:
	virtual bool BindGeometry(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view) = 0;
	virtual void UnbindGeometry() = 0;
	
protected:
	bool ValidateShader(GLuint shader_index);
	bool ValidateProgram(GLuint program_index);
	bool CompileShaders(
			const char * const vertex_shader_code,
			const char * const geometry_shader_code,
			const char * const fragment_shader_code);
	void ReleaseProgram();
	
protected:
	GLuint vertex_shader_;
	GLuint geometry_shader_;
	GLuint fragment_shader_;
	GLuint program_;
	bool compiled_ = false;
	bool bound_ = false;
};

}	// namespace three::glsl

}	// namespace three
