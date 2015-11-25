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

#include "ShaderWrapper.h"

namespace three {

namespace glsl {
	
class ShaderImageDefault : public ShaderWrapper {
public:
	ShaderImageDefault() {}
	virtual ~ShaderImageDefault() {}
	
public:
	virtual bool Compile();
	virtual bool Render(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view);
	virtual void Release();

protected:
	virtual bool BindGeometry(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view);
	virtual void UnbindGeometry();
	virtual bool PrepareRendering(
			const Geometry &geometry,
			const RenderMode &mode,
			const ViewControl &view) { return true; }


protected:
	GLuint vertex_position_;
	GLuint vertex_position_buffer_;
	GLuint vertex_UV_;
	GLuint vertex_UV_buffer_;
	GLuint image_texture_;
	GLuint image_texture_buffer_;
	GLuint vertex_scale_;
	
	GLHelper::GLVector3f vertex_scale_data_;
};
	
}	// namespace three::glsl

}	// namespace three
