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

#pragma once

#include <string>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Core/Core.h>

#include "ColorMap.h"
#include "BoundingBox.h"
#include "ViewControl.h"
#include "RenderMode.h"
#include "ShaderWrapper.h"

namespace three {

class Visualizer  {
public:
	struct MouseControl {
	public:
		bool is_mouse_left_button_down;
		bool is_control_key_down;
		double mouse_position_x;
		double mouse_position_y;

		MouseControl() :
				is_mouse_left_button_down(false),
				is_control_key_down(false),
				mouse_position_x(0.0),
				mouse_position_y(0.0)
		{}
	};
	
public:
	Visualizer();
	virtual ~Visualizer();
	Visualizer(const Visualizer &) = delete;
	Visualizer &operator=(const Visualizer &) = delete;

public:
	/// Function to create a window and initialize GLFW
	/// This function MUST be called from the main thread.
	bool CreateWindow(const std::string window_name = "Open3DV", 
			const int width = 640, const int height = 480,
			const int left = 50, const int top = 50);
	
	/// Function to activate the window
	/// This function will block the current thread until the window is closed.
	void Run();

	/// Function to process the event queue and return if the window is closed
	/// Use this function if you want to manage the while loop yourself. This
	/// function will block the thread.
	bool WaitEvents();
	
	/// Function to process the event queue and return if the window is closed
	/// Use this function if you want to manage the while loop yourself. This
	/// function will NOT block the thread. Thus it is suitable for computation
	/// heavy task behind the scene.
	bool PollEvents();

	/// Function to add geometry to the scene and create corresponding shaders
	/// This function MUST be called after CreateWindow().
	/// This function returns FALSE when the geometry is of an unsupported type.
	bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr);

	/// Function to update shaders
	/// Call this function when geometry or rendermode has been changed.
	void UpdateGeometry();

	/// Function to set the redraw flag as dirty
	void UpdateRender();

	bool HasGeometry();
	virtual void PrintVisualizerHelp();

	ViewControl &GetViewControl() { return view_control_; }

protected:
	// rendering functions

	/// Function to initialize OpenGL
	/// The following things will be set:
	/// PolygonMode
	/// DepthTest
	/// PixelStorei
	/// Mesh material
	/// Note that we use a view point dependent lighting scheme, thus light 
	/// should be set during rendering.
	virtual bool InitOpenGL();

	/// Function to do the main rendering
	/// The function first sets view point, then draw geometry (pointclouds and
	/// meshes individually).
	virtual void Render();

	/// Function to update VBOs in shader containers
	/// This function is called in lazy mode, i.e., only triggers when
	/// is_shader_update_required is true.
	void UpdateShaders();
	void ResetViewPoint();
	void SetDefaultMeshMaterial();
	void SetDefaultLighting(const BoundingBox &bounding_box);

	// callback functions
	virtual void WindowRefreshCallback(GLFWwindow *window);
	virtual void WindowResizeCallback(GLFWwindow *window, int w, int h);
	virtual void MouseMoveCallback(GLFWwindow* window, double x, double y);
	virtual void MouseScrollCallback(GLFWwindow* window, double x, double y);
	virtual void MouseButtonCallback(GLFWwindow* window,
			int button, int action, int mods);
	virtual void KeyPressCallback(GLFWwindow *window,
			int key, int scancode, int action, int mods);
	virtual void WindowCloseCallback(GLFWwindow *window);

protected:
	// window
	GLFWwindow* window_ = NULL;

	// control
	MouseControl mouse_control_;
	ViewControl view_control_;
	bool is_redraw_required_ = true;
	bool is_shader_update_required_ = true;
	bool is_initialized_ = false;

	// rendering properties
	PointCloudRenderMode pointcloud_render_mode_;
	TriangleMeshRenderMode mesh_render_mode_;
	Eigen::Vector3d background_color_ = Eigen::Vector3d(1.0, 1.0, 1.0);

	// geometry to be rendered
	std::vector<std::shared_ptr<const Geometry>> geometry_ptrs_;
	
	// shaders
	std::vector<std::shared_ptr<glsl::ShaderWrapper>> shader_ptrs_;
};

}	// namespace three
