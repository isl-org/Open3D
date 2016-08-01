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

#include <string>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Core/Geometry/Geometry.h>

#include <Visualization/Utility/ColorMap.h>
#include <Visualization/Utility/BoundingBox.h>
#include <Visualization/Visualizer/ViewControl.h>
#include <Visualization/Visualizer/RenderOption.h>
#include <Visualization/Shader/GeometryRenderer.h>

namespace three {

class Visualizer
{
public:
	struct MouseControl {
	public:
		bool is_mouse_left_button_down;
		bool is_control_key_down;
		double mouse_position_x;
		double mouse_position_y;

		MouseControl() :
				is_mouse_left_button_down(false),
				is_control_key_down(false),///
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
	bool CreateWindow(const std::string &window_name = "Open3D", 
			const int width = 640, const int height = 480,
			const int left = 50, const int top = 50);

	/// Function to destroy a window
	/// This function MUST be called from the main thread.
	void DestroyWindow();
	
	/// Function to register a callback function for animation
	/// The callback function returns if UpdateGeometry() needs to be run
	void RegisterAnimationCallback(
			std::function<bool(Visualizer &)> callback_func);
	
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
	/// 1. After calling this function, the Visualizer owns the geometry object.
	/// 2. This function MUST be called after CreateWindow().
	/// 3. This function returns FALSE when the geometry is of an unsupported
	/// type.
	/// 4. If an added geometry is changed, the behavior of Visualizer is
	/// undefined. Programmers are responsible for calling UpdateGeometry() to
	/// notify the Visualizer that the geometry has been changed and the 
	/// Visualizer should be updated accordingly.
	bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr,
			const bool update_boundingbox = true);

	/// Function to update geometry
	/// This function must be called when geometry has been changed. Otherwise
	/// the behavior of Visualizer is undefined.
	bool UpdateGeometry();
	bool HasGeometry() const;

	virtual void PrintVisualizerHelp();
	virtual void UpdateWindowTitle();

	ViewControl &GetViewControl() { return *view_control_ptr_; }
	RenderOption &GetRenderOption() { return *render_option_ptr_; }
	void CaptureScreenImage(const std::string &filename = "",
			bool do_render = true);
	void CaptureDepthImage(const std::string &filename = "",
			bool do_render = true, double depth_scale = 1000.0);
	void CaptureDepthPointCloud(const std::string &filename = "",
			bool do_render = true);
	void CaptureRenderOption(const std::string &filename = "");

protected:
	/// Function to initialize OpenGL
	virtual bool InitOpenGL();

	/// Function to initialize ViewControl
	virtual bool InitViewControl();

	/// Function to initialize RenderOption
	virtual bool InitRenderOption();

	/// Function to do the main rendering
	/// The function first sets view point, then draw geometry (pointclouds and
	/// meshes individually).
	virtual void Render();

	/// Function to set the redraw flag as dirty
	void UpdateRender();

	void ResetViewPoint();

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
	std::string window_name_ = "Open3D";
	std::function<bool(Visualizer &)> animation_callback_func_in_loop_ 
			= nullptr;
	std::function<bool(Visualizer &)> animation_callback_func_ = nullptr;

	// control
	MouseControl mouse_control_;
	bool is_redraw_required_ = true;
	bool is_initialized_ = false;

	// view control
	std::unique_ptr<ViewControl> view_control_ptr_;

	// rendering properties
	std::unique_ptr<RenderOption> render_option_ptr_;

	// geometry to be rendered
	std::vector<std::shared_ptr<const Geometry>> geometry_ptrs_;
	
	// renderers
	std::vector<std::shared_ptr<glsl::GeometryRenderer>> renderer_ptrs_;
	
#ifdef __APPLE__
	// MacBook with Retina display does not have a 1:1 mapping from screen
	// coordinates to pixels. Thus we hack it back.
	// http://www.glfw.org/faq.html#why-is-my-output-in-the-lower-left-corner-of-the-window
	double pixel_to_screen_coordinate_ = 1.0;
#endif //__APPLE__
};

}	// namespace three
