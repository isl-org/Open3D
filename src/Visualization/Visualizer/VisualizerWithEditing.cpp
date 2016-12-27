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

#include <IO/ClassIO/IJsonConvertibleIO.h>
#include <Visualization/Visualizer/ViewControlWithEditing.h>
#include <Visualization/Utility/SelectionPolygon.h>

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
	PrintInfo("    X            : Enter orthogonal view along X axis, press again to flip.\n");
	PrintInfo("    Y            : Enter orthogonal view along Y axis, press again to flip.\n");
	PrintInfo("    Z            : Enter orthogonal view along Z axis, press again to flip.\n");
	PrintInfo("    K            : Lock / unlock camera.\n");
	PrintInfo("\n");
	PrintInfo("    -- When camera is locked --\n");
	PrintInfo("    Mouse left button + drag        : Create a selection rectangle.\n");
	PrintInfo("    Ctrl + mouse left button + drag : Hold Ctrl key to draw a selection polygon.\n");
	PrintInfo("                                      Release Ctrl key to close the polygon.\n");
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

void VisualizerWithEditing::BuildUtilities()
{
	Visualizer::BuildUtilities();
	
	// 1. Build selection polygon
	selection_polygon_ptr_ = std::make_shared<SelectionPolygon>();
	utility_ptrs_.push_back(selection_polygon_ptr_);
	selection_polygon_renderer_ptr_ =
			std::make_shared<glsl::SelectionPolygonRenderer>();
	if (selection_polygon_renderer_ptr_->AddGeometry(selection_polygon_ptr_) ==
			false) {
		return;
	}
	utility_renderer_ptrs_.push_back(selection_polygon_renderer_ptr_);
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
		if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
			if (view_control.IsLocked() &&
					selection_mode_ == SELECTION_POLYGON) {
				selection_mode_ = SELECTION_NONE;
				selection_polygon_ptr_->polygon_.pop_back();
				if (selection_polygon_ptr_->IsEmpty()) {
					selection_polygon_ptr_->Clear();
				} else {
					selection_polygon_ptr_->FillPolygon(
							view_control.GetWindowWidth(),
							view_control.GetWindowHeight());
					selection_polygon_ptr_->polygon_type_ =
							SelectionPolygon::POLYGON_POLYGON;
				}
				selection_polygon_renderer_ptr_->UpdateGeometry();
				is_redraw_required_ = true;
			}
		}
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
	case GLFW_KEY_K:
		view_control.ToggleLocking();
		InvalidateSelectionPolygon();
		PrintDebug("[Visualizer] Camera %s.\n",
				view_control.IsLocked() ? "Lock" : "Unlock");
		break;
	default:
		Visualizer::KeyPressCallback(window, key, scancode, action, mods);
		break;
	}
	is_redraw_required_ = true;
	UpdateWindowTitle();
}

void VisualizerWithEditing::WindowResizeCallback(GLFWwindow *window, int w,
		int h)
{
	InvalidateSelectionPolygon();
	Visualizer::WindowResizeCallback(window, w, h);
}

void VisualizerWithEditing::MouseMoveCallback(GLFWwindow* window, 
		double x, double y)
{
	auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
	if (view_control.IsLocked()) {
#ifdef __APPLE__
		x /= pixel_to_screen_coordinate_;
		y /= pixel_to_screen_coordinate_;
#endif
		double y_inv = view_control.GetWindowHeight() - y;
		if (selection_mode_ == SELECTION_NONE) {
		} else if (selection_mode_ == SELECTION_RECTANGLE) {
			selection_polygon_ptr_->polygon_[1](0) = x;
			selection_polygon_ptr_->polygon_[2](0) = x;
			selection_polygon_ptr_->polygon_[2](1) = y_inv;
			selection_polygon_ptr_->polygon_[3](1) = y_inv;
			selection_polygon_renderer_ptr_->UpdateGeometry();
			is_redraw_required_ = true;
		} else if (selection_mode_ == SELECTION_POLYGON) {
			selection_polygon_ptr_->polygon_.back() = Eigen::Vector2d(x, y_inv);
			selection_polygon_renderer_ptr_->UpdateGeometry();
			is_redraw_required_ = true;
		}
	} else {
		Visualizer::MouseMoveCallback(window, x, y);
	}
}

void VisualizerWithEditing::MouseScrollCallback(GLFWwindow* window, 
		double x, double y)
{
	auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
	if (view_control.IsLocked()) {
	} else {
		Visualizer::MouseScrollCallback(window, x, y);
	}
}

void VisualizerWithEditing::MouseButtonCallback(GLFWwindow* window,
		int button, int action, int mods)
{
	auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
	if (view_control.IsLocked()) {
		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			double x, y;
			glfwGetCursorPos(window, &x, &y);
#ifdef __APPLE__
			x /= pixel_to_screen_coordinate_;
			y /= pixel_to_screen_coordinate_;
#endif
			if (action == GLFW_PRESS) {
				double y_inv = view_control.GetWindowHeight() - y;
				if (selection_mode_ == SELECTION_NONE) {
					InvalidateSelectionPolygon();
					if (mods & GLFW_MOD_CONTROL) {
						selection_mode_ = SELECTION_POLYGON;
						selection_polygon_ptr_->polygon_.push_back(
								Eigen::Vector2d(x, y_inv));
						selection_polygon_ptr_->polygon_.push_back(
								Eigen::Vector2d(x, y_inv));
					} else {
						selection_mode_ = SELECTION_RECTANGLE;
						selection_polygon_ptr_->is_closed_ = true;
						selection_polygon_ptr_->polygon_.push_back(
								Eigen::Vector2d(x, y_inv));
						selection_polygon_ptr_->polygon_.push_back(
								Eigen::Vector2d(x, y_inv));
						selection_polygon_ptr_->polygon_.push_back(
								Eigen::Vector2d(x, y_inv));
						selection_polygon_ptr_->polygon_.push_back(
								Eigen::Vector2d(x, y_inv));
					}
					selection_polygon_renderer_ptr_->UpdateGeometry();
				} else if (selection_mode_ == SELECTION_RECTANGLE) {
				} else if (selection_mode_ == SELECTION_POLYGON) {
					if (mods & GLFW_MOD_CONTROL) {
						selection_polygon_ptr_->polygon_.back() =
								Eigen::Vector2d(x, y_inv);
						selection_polygon_ptr_->polygon_.push_back(
								Eigen::Vector2d(x, y_inv));
						selection_polygon_renderer_ptr_->UpdateGeometry();
					}
				}
			} else if (action == GLFW_RELEASE) {
				if (selection_mode_ == SELECTION_NONE) {
				} else if (selection_mode_ == SELECTION_RECTANGLE) {
					selection_mode_ = SELECTION_NONE;
					selection_polygon_ptr_->FillPolygon(
							view_control.GetWindowWidth(),
							view_control.GetWindowHeight());
					selection_polygon_ptr_->polygon_type_ =
							SelectionPolygon::POLYGON_RECTANGLE;
					selection_polygon_renderer_ptr_->UpdateGeometry();
				} else if (selection_mode_ == SELECTION_POLYGON) {
				}
			}
			is_redraw_required_ = true;
		}
	} else {
		Visualizer::MouseButtonCallback(window, button, action, mods);
	}
}

void VisualizerWithEditing::InvalidateSelectionPolygon()
{
	if (selection_polygon_ptr_) selection_polygon_ptr_->Clear();
	if (selection_polygon_renderer_ptr_) {
		selection_polygon_renderer_ptr_->UpdateGeometry();
	}
	selection_mode_ = SELECTION_NONE;
}

}	// namespace three
