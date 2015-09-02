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

namespace {

class GLFWEnvironmentSingleton {
public:
	GLFWEnvironmentSingleton() { PrintDebug("GLFW init.\n");}
	~GLFWEnvironmentSingleton() {
		glfwTerminate();
		PrintDebug("GLFW destruct.\n");
	}

public:
	static int InitGLFW() {
		static GLFWEnvironmentSingleton singleton;
		return glfwInit();
	}

	static void GLFWErrorCallback(int error, const char* description) {
		PrintError("GLFW Error: %s\n", description);
	}
};

}	// unnamed namespace

Visualizer::Visualizer() : 
		window_(NULL),
		mouse_control_(),
		view_control_(),
		is_redraw_required_(true),
		pointcloud_render_mode_(),
		mesh_render_mode_(),
		color_map_ptr_(new ColorMapJet),
		background_color_(1.0, 1.0, 1.0)
{
}

Visualizer::~Visualizer()
{
	pointcloud_default_shader_.Release();
	glfwTerminate();	// to be safe
}

bool Visualizer::CreateWindow(const std::string window_name/* = "Open3DV"*/, 
		const int width/* = 640*/, const int height/* = 480*/,
		const int left/* = 50*/, const int top/* = 50*/)
{
	if (window_) {	// window already created
		glfwSetWindowTitle(window_, window_name.c_str());
		glfwSetWindowPos(window_, left, top);
		glfwSetWindowSize(window_, width, height);
		return true;
	}

	glfwSetErrorCallback(GLFWEnvironmentSingleton::GLFWErrorCallback);
	if (!GLFWEnvironmentSingleton::InitGLFW()) {
		PrintError("Failed to initialize GLFW\n");
		return false;
	}
	
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	window_ = glfwCreateWindow(width, height, window_name.c_str(), NULL, NULL);
	if (!window_) {
		PrintError("Failed to create window\n");
		return false;
	}
	glfwSetWindowPos(window_, left, top);
	glfwSetWindowUserPointer(window_, this);

	auto window_refresh_callback = [](GLFWwindow *window) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				WindowRefreshCallback(window);
	};
	glfwSetWindowRefreshCallback(window_, window_refresh_callback);

	auto window_resize_callback = [](GLFWwindow *window, int w, int h) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				WindowResizeCallback(window, w, h);
	};
	glfwSetFramebufferSizeCallback(window_, window_resize_callback);

	auto mouse_move_callback = [](GLFWwindow *window, double x, double y) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				MouseMoveCallback(window, x, y);
	};
	glfwSetCursorPosCallback(window_, mouse_move_callback);

	auto mouse_scroll_callback = [](GLFWwindow *window, double x, double y) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				MouseScrollCallback(window, x, y);
	};
	glfwSetScrollCallback(window_, mouse_scroll_callback);

	auto mouse_button_callback = [](GLFWwindow *window,
			int button, int action, int mods)
	{
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				MouseButtonCallback(window, button, action, mods);
	};
	glfwSetMouseButtonCallback(window_, mouse_button_callback);

	auto key_press_callback = [](GLFWwindow *window,
			int key, int scancode, int action, int mods)
	{
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				KeyPressCallback(window, key, scancode, action, mods);
	};
	glfwSetKeyCallback(window_, key_press_callback);

	auto window_close_callback = [](GLFWwindow *window) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				WindowCloseCallback(window);
	};
	glfwSetWindowCloseCallback(window_, window_close_callback);
	
	glfwMakeContextCurrent(window_);
	glfwSwapInterval(1);

	int window_width, window_height;
	glfwGetFramebufferSize(window_, &window_width, &window_height);
	WindowResizeCallback(window_, window_width, window_height);

	if (InitOpenGL() == false) {
		return false;
	}
	
	ResetViewPoint();

	return true;
}

void Visualizer::Run()
{
	glfwMakeContextCurrent(window_);
	while (WaitEvents()) {
	}
}

bool Visualizer::WaitEvents()
{
	glfwMakeContextCurrent(window_);
	if (is_redraw_required_) {
		WindowRefreshCallback(window_);
	}
	glfwWaitEvents();
	return !glfwWindowShouldClose(window_);
}

bool Visualizer::PollEvents()
{
	glfwMakeContextCurrent(window_);
	if (is_redraw_required_) {
		WindowRefreshCallback(window_);
	}
	glfwPollEvents();
	return !glfwWindowShouldClose(window_);
}

void Visualizer::AddGeometry(std::shared_ptr<const Geometry> geometry_ptr)
{
	geometry_ptrs_.push_back(geometry_ptr);
	//pointcloud_default_shader_.BindGeometry(*geometry_ptr);
	view_control_.AddGeometry(*geometry_ptr);
	ResetViewPoint();
	is_redraw_required_ = true;
}

bool Visualizer::HasGeometry()
{
	return !geometry_ptrs_.empty();
}

void Visualizer::PrintVisualizerHelp()
{
	PrintInfo("Mouse options:\n");
	PrintInfo("    Left btn + drag        : Rotate.\n");
	PrintInfo("    Ctrl + left btn + drag : Translate.\n");
	PrintInfo("    Wheel                  : Zoom in/out.\n");
	PrintInfo("\n");
	PrintInfo("Keyboard options:\n");
	PrintInfo("    Q, Esc       : Exit window.\n");
	PrintInfo("    R            : Reset view point.\n");
	PrintInfo("    [/]          : Increase/decrease field of view.\n");
	PrintInfo("    +/-          : Increase/decrease point size.\n");
	PrintInfo("    0..4         : Point color options.\n");
	PrintInfo("                   0 - Default behavior, use z value to render.\n");
	PrintInfo("                   1 - Render point color.\n");
	PrintInfo("                   2 - x coordinate as color.\n");
	PrintInfo("                   3 - y coordinate as color.\n");
	PrintInfo("                   4 - z coordinate as color.\n");
	PrintInfo("    Shift + 0..3 : Color map options.\n");
	PrintInfo("                   0 - Gray scale color.\n");
	PrintInfo("                   1 - JET color map.\n");
	PrintInfo("                   2 - SUMMER color map.\n");
	PrintInfo("                   3 - WINTER color map.\n");
	PrintInfo("    Ctrl + 0..3  : Mesh rendering options.\n");
	PrintInfo("                   0 - Render with vertex colors.\n");
	PrintInfo("                   1 - Render flat polygons.\n");
	PrintInfo("                   2 - Render smooth polygons.\n");
	PrintInfo("                   3 - Render wireframe.\n");
	PrintInfo("    N            : Turn on/off normal rendering.\n");
	PrintInfo("\n");
}

}	// namespace three
