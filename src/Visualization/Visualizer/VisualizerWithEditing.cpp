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

#include "VisualizerWithEditing.h"

#include <Visualization/Visualizer/ViewControlWithEditing.h>
#include <IO/ClassIO/IJsonConvertibleIO.h>

namespace three{

VisualizerWithEditing::VisualizerWithEditing()
{
}

VisualizerWithEditing::~VisualizerWithEditing()
{
}

void VisualizerWithEditing::PrintVisualizerHelp()
{
	Visualizer::PrintVisualizerHelp();
	PrintInfo("  -- Editing control --\n");
	PrintInfo("    F            : Enter freeview mode.\n");
	PrintInfo("    X            : Enter orthogonal editing view along X axis, press again to flip.\n");
	PrintInfo("    Y            : Enter orthogonal editing view along Y axis, press again to flip.\n");
	PrintInfo("    Z            : Enter orthogonal editing view along Z axis, press again to flip.\n");
	PrintInfo("\n");
	PrintInfo("    -- In orthogonal editing view --\n");
	PrintInfo("    Ctrl + <-/-> : Go backward/forward a keyframe.\n");
	PrintInfo("\n");
}

void VisualizerWithEditing::UpdateWindowTitle()
{
	if (window_ != NULL) {
		auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
		std::string new_window_title = window_name_ + " - " + 
				view_control.GetStatusString();
		glfwSetWindowTitle(window_, new_window_title.c_str());
	}
}

bool VisualizerWithEditing::InitViewControl()
{
	view_control_ptr_ = std::unique_ptr<ViewControlWithEditing>(
			new ViewControlWithEditing);
	ResetViewPoint();
	return true;
}

void VisualizerWithEditing::KeyPressCallback(GLFWwindow *window,
		int key, int scancode, int action, int mods)
{
	auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
	if (action == GLFW_RELEASE) {
		return;
	}

	switch (key) {
	case GLFW_KEY_F:
		view_control.SetEditingMode(ViewControlWithEditing::EDITING_FREEMODE);
		PrintDebug("[Visualizer] Enter freeview mode.\n");
		break;
	case GLFW_KEY_X:
		view_control.ToggleEditingX();
		PrintDebug("[Visualizer] Enter orthogonal X editing mode.\n");
		break;
	case GLFW_KEY_Y:
		view_control.ToggleEditingY();
		PrintDebug("[Visualizer] Enter orthogonal Y editing mode.\n");
		break;
	case GLFW_KEY_Z:
		view_control.ToggleEditingZ();
		PrintDebug("[Visualizer] Enter orthogonal Z editing mode.\n");
		break;
	default:
		Visualizer::KeyPressCallback(window, key, scancode, action, mods);
		break;
	}
	is_redraw_required_ = true;
	UpdateWindowTitle();
}

void VisualizerWithEditing::MouseMoveCallback(GLFWwindow* window, 
		double x, double y)
{
	auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
	Visualizer::MouseMoveCallback(window, x, y);
}

void VisualizerWithEditing::MouseScrollCallback(GLFWwindow* window, 
		double x, double y)
{
	auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
	Visualizer::MouseScrollCallback(window, x, y);
}

void VisualizerWithEditing::MouseButtonCallback(GLFWwindow* window,
		int button, int action, int mods)
{
	auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
	Visualizer::MouseButtonCallback(window, button, action, mods);
}

}	// namespace three
