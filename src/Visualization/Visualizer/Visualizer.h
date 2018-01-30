// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

class TriangleMesh;
class Image;

class Visualizer
{
public:
	struct MouseControl {
	public:
		bool is_mouse_left_button_down = false;
		bool is_control_key_down = false;
		bool is_shift_key_down = false;
		bool is_alt_key_down = false;
		bool is_super_key_down = false;
		double mouse_position_x = 0.0;
		double mouse_position_y = 0.0;
	};

public:
	Visualizer();
	virtual ~Visualizer();
	Visualizer(Visualizer &&) = delete;
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
			std::function<bool(Visualizer *)> callback_func);

	/// Function to activate the window
	/// This function will block the current thread until the window is closed.
	void Run();

	/// Function to to notify the window to be closed
	void Close();

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
	virtual bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr);

	/// Function to update geometry
	/// This function must be called when geometry has been changed. Otherwise
	/// the behavior of Visualizer is undefined.
	virtual bool UpdateGeometry();
	virtual bool HasGeometry() const;

	/// Function to set the redraw flag as dirty
	virtual void UpdateRender();

	virtual void PrintVisualizerHelp();
	virtual void UpdateWindowTitle();
	virtual void BuildUtilities();

	ViewControl &GetViewControl() { return *view_control_ptr_; }
	RenderOption &GetRenderOption() { return *render_option_ptr_; }
	std::shared_ptr<Image> CaptureScreenFloatBuffer(bool do_render = true);
	void CaptureScreenImage(const std::string &filename = "",
			bool do_render = true);
	std::shared_ptr<Image> CaptureDepthFloatBuffer(bool do_render = true);
	void CaptureDepthImage(const std::string &filename = "",
			bool do_render = true, double depth_scale = 1000.0);
	void CaptureDepthPointCloud(const std::string &filename = "",
			bool do_render = true);
	void CaptureRenderOption(const std::string &filename = "");
	void ResetViewPoint(bool reset_bounding_box = false);

	const std::string &GetWindowName() const {
		return window_name_;
	}

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

	void CopyViewStatusToClipboard();

	void CopyViewStatusFromClipboard();

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
	std::function<bool(Visualizer *)> animation_callback_func_ = nullptr;
	// Auxiliary internal backup of the callback function.
	// It copies animation_callback_func_ in each PollEvent() or WaitEvent()
	// so that even if user calls RegisterAnimationCallback() within the
	// callback function it is still safe.
	std::function<bool(Visualizer *)> animation_callback_func_in_loop_
		= nullptr;

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

	// geometry renderers
	std::vector<std::shared_ptr<glsl::GeometryRenderer>>
			geometry_renderer_ptrs_;

	// utilities owned by the Visualizer
	std::vector<std::shared_ptr<const Geometry>> utility_ptrs_;

	// utility renderers
	std::vector<std::shared_ptr<glsl::GeometryRenderer>> utility_renderer_ptrs_;

	// coordinate frame
	std::shared_ptr<TriangleMesh> coordinate_frame_mesh_ptr_;
	std::shared_ptr<glsl::CoordinateFrameRenderer>
			coordinate_frame_mesh_renderer_ptr_;

#ifdef __APPLE__
	// MacBook with Retina display does not have a 1:1 mapping from screen
	// coordinates to pixels. Thus we hack it back.
	// http://www.glfw.org/faq.html#why-is-my-output-in-the-lower-left-corner-of-the-window
	double pixel_to_screen_coordinate_ = 1.0;
#endif //__APPLE__
};

}	// namespace three
