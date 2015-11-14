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
#include "ShaderPointCloud.h"
#include "ShaderTriangleMesh.h"
#include "ShaderImage.h"

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

Visualizer::Visualizer()
{
}

Visualizer::~Visualizer()
{
	for (auto shader_ptr : shader_ptrs_) {
		shader_ptr->Release();
	}
	glfwTerminate();	// to be safe
}

bool Visualizer::CreateWindow(const std::string window_name/* = "Open3DV"*/, 
		const int width/* = 640*/, const int height/* = 480*/,
		const int left/* = 50*/, const int top/* = 50*/)
{
	window_name_ = window_name;
	if (window_) {	// window already created
		UpdateWindowTitle();
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

	window_ = glfwCreateWindow(width, height, window_name_.c_str(), NULL, NULL);
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

	if (InitOpenGL() == false) {
		return false;
	}

	if (InitViewControl() == false) {
		return false;
	}

	int window_width, window_height;
	glfwGetFramebufferSize(window_, &window_width, &window_height);
	WindowResizeCallback(window_, window_width, window_height);

	UpdateWindowTitle();
	is_initialized_ = true;
	return true;
}

void Visualizer::RegisterAnimationCallback(
		std::function<bool (Visualizer &)> callback_func)
{
	animation_callback_func_ = callback_func;
}

bool Visualizer::InitViewControl()
{
	view_control_ptr_ = std::unique_ptr<ViewControl>(new ViewControl);
	ResetViewPoint();
	return true;
}

void Visualizer::UpdateWindowTitle()
{
	if (window_ != NULL) {
		glfwSetWindowTitle(window_, window_name_.c_str());
	}
}

void Visualizer::Run()
{
	while (bool(animation_callback_func_) ? PollEvents() : WaitEvents()) {
		if (bool(animation_callback_func_in_loop_)) {
			if (animation_callback_func_in_loop_(*this)) {
				UpdateGeometry();
			}
			// Set render flag as dirty anyways, because when we use callback
			// functions, we assume something has been changed in the callback
			// and the redraw event should be triggered.
			UpdateRender();
		}
	}
}

bool Visualizer::WaitEvents()
{
	if (is_initialized_ == false) {
		return false;
	}
	glfwMakeContextCurrent(window_);
	if (is_redraw_required_) {
		WindowRefreshCallback(window_);
	}
	animation_callback_func_in_loop_ = animation_callback_func_;
	glfwWaitEvents();
	return !glfwWindowShouldClose(window_);
}

bool Visualizer::PollEvents()
{
	if (is_initialized_ == false) {
		return false;
	}
	glfwMakeContextCurrent(window_);
	if (is_redraw_required_) {
		WindowRefreshCallback(window_);
	}
	animation_callback_func_in_loop_ = animation_callback_func_;
	glfwPollEvents();
	return !glfwWindowShouldClose(window_);
}

bool Visualizer::AddGeometry(std::shared_ptr<const Geometry> geometry_ptr)
{
	if (is_initialized_ == false) {
		return false;
	}

	if (geometry_ptr->GetGeometryType() == 
			Geometry::GEOMETRY_UNKNOWN) {
		return false;
	} else if (geometry_ptr->GetGeometryType() == 
			Geometry::GEOMETRY_POINTCLOUD) {
		auto shader_ptr = std::make_shared<glsl::ShaderPointCloudDefault>();
		if (shader_ptr->Compile() == false) {
			return false;
		}
		shader_ptrs_.push_back(shader_ptr);
	} else if (geometry_ptr->GetGeometryType() == 
			Geometry::GEOMETRY_TRIANGLEMESH) {
		auto shader_ptr = std::make_shared<glsl::ShaderTriangleMeshDefault>();
		if (shader_ptr->Compile() == false) {
			return false;
		}
		shader_ptrs_.push_back(shader_ptr);
	} else if (geometry_ptr->GetGeometryType() ==
			Geometry::GEOMETRY_IMAGE) {
		auto shader_ptr = std::make_shared<glsl::ShaderImageDefault>();
		if (shader_ptr->Compile() == false) {
			return false;
		}
		shader_ptrs_.push_back(shader_ptr);
	} else {
		return false;
	}

	geometry_ptrs_.push_back(geometry_ptr);
	view_control_ptr_->AddGeometry(*geometry_ptr);
	ResetViewPoint();
	PrintDebug("Add geometry and update bounding box to %s\n", 
			view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
	return UpdateGeometry();
}

bool Visualizer::UpdateGeometry()
{
	UpdateShaders();
	return true;
}

void Visualizer::UpdateShaders()
{
	is_shader_update_required_ = true;
	is_redraw_required_ = true;
}

void Visualizer::UpdateRender()
{
	is_redraw_required_ = true;
}

bool Visualizer::HasGeometry() const
{
	return !geometry_ptrs_.empty();
}

void Visualizer::PrintVisualizerHelp()
{
	PrintInfo("Mouse options:\n");
	PrintInfo("\n");
	PrintInfo("    Left btn + drag        : Rotate.\n");
	PrintInfo("    Ctrl + left btn + drag : Translate.\n");
	PrintInfo("    Wheel                  : Zoom in/out.\n");
	PrintInfo("\n");
	PrintInfo("Keyboard options:\n");
	PrintInfo("\n");
	PrintInfo("  -- General control --\n");
	PrintInfo("    Q, Esc       : Exit window.\n");
	PrintInfo("    H            : Print help message.\n");
	PrintInfo("    P, PrtScn    : Take a screen capture.\n");
	PrintInfo("    R            : Reset view point.\n");
	PrintInfo("    [/]          : Increase/decrease field of view.\n");
	PrintInfo("\n");
	PrintInfo("  -- Render mode control --\n");
	PrintInfo("    +/-          : Increase/decrease point size.\n");
	PrintInfo("    N            : Turn on/off point cloud normal rendering.\n");
	PrintInfo("    S            : Toggle between mesh flat shading and smooth shading.\n");
	PrintInfo("    B            : Turn on/off back face rendering.\n");
	PrintInfo("    I            : Turn on/off image zoom in interpolation.\n");
	PrintInfo("    T            : Toggle among image render: no stretch / keep ratio / freely stretch.\n");
	PrintInfo("\n");
	PrintInfo("  -- Color control --\n");
	PrintInfo("    0..4         : Set point cloud color option.\n");
	PrintInfo("                   0 - Default behavior, render point color.\n");
	PrintInfo("                   1 - Render point color.\n");
	PrintInfo("                   2 - x coordinate as color.\n");
	PrintInfo("                   3 - y coordinate as color.\n");
	PrintInfo("                   4 - z coordinate as color.\n");
	PrintInfo("    Ctrl + 0..4  : Set mesh color option.\n");
	PrintInfo("                   0 - Default behavior, render uniform turquoise color.\n");
	PrintInfo("                   1 - Render point color.\n");
	PrintInfo("                   2 - x coordinate as color.\n");
	PrintInfo("                   3 - y coordinate as color.\n");
	PrintInfo("                   4 - z coordinate as color.\n");
	PrintInfo("    Shift + 0..3 : Color map options.\n");
	PrintInfo("                   0 - Gray scale color.\n");
	PrintInfo("                   1 - JET color map.\n");
	PrintInfo("                   2 - SUMMER color map.\n");
	PrintInfo("                   3 - WINTER color map.\n");
	PrintInfo("\n");
}

}	// namespace three
