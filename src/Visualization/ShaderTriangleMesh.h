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
	
class ShaderTriangleMeshDefault : public ShaderWrapper {
public:
	ShaderTriangleMeshDefault() {}
	virtual ~ShaderTriangleMeshDefault() {}
	
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
	virtual void SetLight(
			const TriangleMeshRenderMode &mode, 
			const ViewControl &view);

protected:
	GLuint vertex_position_;
	GLuint vertex_position_buffer_;
	GLuint vertex_color_;
	GLuint vertex_color_buffer_;
	GLuint vertex_normal_;
	GLuint vertex_normal_buffer_;
	GLuint MVP_;
	GLuint V_;
	GLuint M_;
	GLuint light_position_world_;
	GLuint light_color_;
	GLuint light_power_;

	// At most support 4 lights
	GLHelper::GLMatrix4f light_position_world_data_;
	GLHelper::GLMatrix4f light_color_data_;
	GLHelper::GLVector4f light_power_data_;

	Eigen::Vector3d default_color_ = 
			Eigen::Vector3d(0.439216, 0.858824, 0.858824);
	
	GLsizei vertex_num_ = 0;
	bool lights_on_ = true;
};
	
}	// namespace three::glsl

}	// namespace three
