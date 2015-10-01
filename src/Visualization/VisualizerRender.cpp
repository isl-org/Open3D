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

namespace three{

bool Visualizer::InitOpenGL()
{
	if (glewInit() != GLEW_OK) {
		PrintError("Failed to initialize GLEW.\n");
		return false;
	}
		
	// depth test
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0f);

	// pixel alignment
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// polygon rendering
	glEnable(GL_CULL_FACE);

	return true;
}

void Visualizer::Render()
{
	glfwMakeContextCurrent(window_);

	if (is_shader_update_required_) {
		UpdateShaders();
		is_shader_update_required_ = false;
	}

	view_control_.SetViewPoint();

	glClearColor((GLclampf)background_color_(0),
			(GLclampf)background_color_(1),
			(GLclampf)background_color_(2), 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (size_t i = 0; i < geometry_ptrs_.size(); i++) {
		// dispatch geometry
		const Geometry &geometry = *geometry_ptrs_[i];
		switch (geometry.GetGeometryType()) {
		case Geometry::GEOMETRY_POINTCLOUD:
			shader_ptrs_[i]->Render(pointcloud_render_mode_, view_control_);
			break;
		case Geometry::GEOMETRY_TRIANGLEMESH:
			shader_ptrs_[i]->Render(mesh_render_mode_, view_control_);
			break;
		case Geometry::GEOMETRY_IMAGE:
			shader_ptrs_[i]->Render(image_render_mode_, view_control_);
			break;
		case Geometry::GEOMETRY_UNKNOWN:
		default:
			break;
		}
	}

	glfwSwapBuffers(window_);
}

void Visualizer::UpdateShaders()
{
	for (size_t i = 0; i < geometry_ptrs_.size(); i++) {
		const Geometry &geometry = *geometry_ptrs_[i];
		switch (geometry.GetGeometryType()) {
		case Geometry::GEOMETRY_POINTCLOUD:
			shader_ptrs_[i]->BindGeometry(
					*geometry_ptrs_[i], 
					pointcloud_render_mode_,
					view_control_);
			break;
		case Geometry::GEOMETRY_TRIANGLEMESH:
			shader_ptrs_[i]->BindGeometry(
					*geometry_ptrs_[i],
					mesh_render_mode_,
					view_control_);
			break;
		case Geometry::GEOMETRY_IMAGE:
			shader_ptrs_[i]->BindGeometry(
					*geometry_ptrs_[i],
					image_render_mode_,
					view_control_);
			break;
		case Geometry::GEOMETRY_UNKNOWN:
		default:
			break;
		}
	}
}

void Visualizer::ResetViewPoint()
{
	view_control_.Reset();
	is_redraw_required_ = true;
}

}	// namespace three
