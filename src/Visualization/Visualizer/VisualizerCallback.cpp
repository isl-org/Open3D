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

#include "Visualizer.h"

namespace three{

void Visualizer::WindowRefreshCallback(GLFWwindow *window)
{
	if (is_redraw_required_) {
		Render();
		is_redraw_required_ = false;
	}
}

void Visualizer::WindowResizeCallback(GLFWwindow *window, int w, int h)
{
	view_control_ptr_->ChangeWindowSize(w, h);
	is_redraw_required_ = true;
}

void Visualizer::MouseMoveCallback(GLFWwindow *window, double x, double y)
{
#ifdef __APPLE__
	x /= pixel_to_screen_coordinate_;
	y /= pixel_to_screen_coordinate_;
#endif
	if (mouse_control_.is_mouse_left_button_down) {
		if (mouse_control_.is_control_key_down) {
			view_control_ptr_->Translate(
					x - mouse_control_.mouse_position_x,
					y - mouse_control_.mouse_position_y,
					mouse_control_.mouse_position_x,
					mouse_control_.mouse_position_y);
		} else {
			view_control_ptr_->Rotate(
					x - mouse_control_.mouse_position_x,
					y - mouse_control_.mouse_position_y,
					mouse_control_.mouse_position_x,
					mouse_control_.mouse_position_y);
		}
		is_redraw_required_ = true;
	}
	mouse_control_.mouse_position_x = x;
	mouse_control_.mouse_position_y = y;
}

void Visualizer::MouseScrollCallback(GLFWwindow* window, double x, double y)
{
	view_control_ptr_->Scale(y);
	is_redraw_required_ = true;
}

void Visualizer::MouseButtonCallback(GLFWwindow* window,
		int button, int action, int mods)
{
	double x, y;
	glfwGetCursorPos(window, &x, &y);
#ifdef __APPLE__
	x /= pixel_to_screen_coordinate_;
	y /= pixel_to_screen_coordinate_;
#endif
	mouse_control_.mouse_position_x = x;
	mouse_control_.mouse_position_y = y;
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouse_control_.is_mouse_left_button_down = true;
			mouse_control_.is_control_key_down =
					(mods & GLFW_MOD_CONTROL) != 0;
			mouse_control_.is_shift_key_down = (mods & GLFW_MOD_SHIFT) != 0;
			mouse_control_.is_alt_key_down = (mods & GLFW_MOD_ALT) != 0;
			mouse_control_.is_super_key_down = (mods & GLFW_MOD_SUPER) != 0;
		} else {
			mouse_control_.is_mouse_left_button_down = false;
			mouse_control_.is_control_key_down = false;
			mouse_control_.is_shift_key_down = false;
			mouse_control_.is_alt_key_down = false;
			mouse_control_.is_super_key_down = false;
		}
	}
}

void Visualizer::KeyPressCallback(GLFWwindow *window,
		int key, int scancode, int action, int mods)
{
	if (action == GLFW_RELEASE) {
		return;
	}

	switch (key) {
	case GLFW_KEY_LEFT_BRACKET:
		view_control_ptr_->ChangeFieldOfView(-1.0);
		PrintDebug("[Visualizer] Field of view set to %.2f.\n",
				view_control_ptr_->GetFieldOfView());
		break;
	case GLFW_KEY_RIGHT_BRACKET:
		view_control_ptr_->ChangeFieldOfView(1.0);
		PrintDebug("[Visualizer] Field of view set to %.2f.\n",
				view_control_ptr_->GetFieldOfView());
		break;
	case GLFW_KEY_R:
		ResetViewPoint();
		PrintDebug("[Visualizer] Reset view point.\n");
		break;
	case GLFW_KEY_C:
		if (mods & GLFW_MOD_CONTROL || mods & GLFW_MOD_SUPER) {
			CopyViewStatusToClipboard();
		}
		break;
	case GLFW_KEY_V:
		if (mods & GLFW_MOD_CONTROL || mods & GLFW_MOD_SUPER) {
			CopyViewStatusFromClipboard();
		}
		break;
	case GLFW_KEY_ESCAPE:
	case GLFW_KEY_Q:
		Close();
		break;
	case GLFW_KEY_H:
		PrintVisualizerHelp();
		break;
	case GLFW_KEY_P:
	case GLFW_KEY_PRINT_SCREEN:
		CaptureScreenImage();
		break;
	case GLFW_KEY_D:
		CaptureDepthImage();
		break;
	case GLFW_KEY_O:
		CaptureRenderOption();
		break;
	case GLFW_KEY_L:
		render_option_ptr_->ToggleLightOn();
		PrintDebug("[Visualizer] Lighting %s.\n",
				render_option_ptr_->light_on_ ? "ON" : "OFF");
		break;
	case GLFW_KEY_EQUAL:
		render_option_ptr_->ChangePointSize(1.0);
		if (render_option_ptr_->point_show_normal_) {
			UpdateGeometry();
		}
		PrintDebug("[Visualizer] Point size set to %.2f.\n",
				render_option_ptr_->point_size_);
		break;
	case GLFW_KEY_MINUS:
		render_option_ptr_->ChangePointSize(-1.0);
		if (render_option_ptr_->point_show_normal_) {
			UpdateGeometry();
		}
		PrintDebug("[Visualizer] Point size set to %.2f.\n",
				render_option_ptr_->point_size_);
		break;
	case GLFW_KEY_N:
		render_option_ptr_->TogglePointShowNormal();
		if (render_option_ptr_->point_show_normal_) {
			UpdateGeometry();
		}
		PrintDebug("[Visualizer] Point normal rendering %s.\n",
				render_option_ptr_->point_show_normal_ ? "ON" : "OFF");
		break;
	case GLFW_KEY_S:
		render_option_ptr_->ToggleShadingOption();
		UpdateGeometry();
		PrintDebug("[Visualizer] Mesh shading mode is %s.\n",
				render_option_ptr_->mesh_shade_option_ ==
				RenderOption::MeshShadeOption::FlatShade ?
				"FLAT" : "SMOOTH");
		break;
	case GLFW_KEY_W:
		render_option_ptr_->ToggleMeshShowWireframe();
		PrintDebug("[Visualizer] Mesh wireframe rendering %s.\n",
				render_option_ptr_->mesh_show_wireframe_ ? "ON" : "OFF");
		break;
	case GLFW_KEY_B:
		render_option_ptr_->ToggleMeshShowBackFace();
		PrintDebug("[Visualizer] Mesh back face rendering %s.\n",
				render_option_ptr_->mesh_show_back_face_ ? "ON" : "OFF");
		break;
	case GLFW_KEY_I:
		render_option_ptr_->ToggleInterpolationOption();
		UpdateGeometry();
		PrintDebug("[Visualizer] Image interpolation mode is %s.\n",
				render_option_ptr_->interpolation_option_ ==
				RenderOption::TextureInterpolationOption::Nearest ?
				"NEARST" : "LINEAR");
		break;
	case GLFW_KEY_T:
		render_option_ptr_->ToggleImageStretchOption();
		PrintDebug("[Visualizer] Image stretch mode is #%d.\n",
				int(render_option_ptr_->image_stretch_option_));
		break;
	case GLFW_KEY_0:
		if (mods & GLFW_MOD_CONTROL) {
			render_option_ptr_->mesh_color_option_ =
					RenderOption::MeshColorOption::Default;
			UpdateGeometry();
			PrintDebug("[Visualizer] Mesh color set to DEFAULT.\n");
		} else if (mods & GLFW_MOD_SHIFT) {
			SetGlobalColorMap(ColorMap::ColorMapOption::Gray);
			UpdateGeometry();
			PrintDebug("[Visualizer] Color map set to GRAY.\n");
		} else {
			render_option_ptr_->point_color_option_ = 
					RenderOption::PointColorOption::Default;
			UpdateGeometry();
			PrintDebug("[Visualizer] Point color set to DEFAULT.\n");
		}
		break;
	case GLFW_KEY_1:
		if (mods & GLFW_MOD_CONTROL) {
			render_option_ptr_->mesh_color_option_ = 
					RenderOption::MeshColorOption::Color;
			UpdateGeometry();
			PrintDebug("[Visualizer] Mesh color set to COLOR.\n");
		} else if (mods & GLFW_MOD_SHIFT) {
			SetGlobalColorMap(ColorMap::ColorMapOption::Jet);
			UpdateGeometry();
			PrintDebug("[Visualizer] Color map set to JET.\n");
		} else {
			render_option_ptr_->point_color_option_ =
					RenderOption::PointColorOption::Color;
			UpdateGeometry();
			PrintDebug("[Visualizer] Point color set to COLOR.\n");
		}
		break;
	case GLFW_KEY_2:
		if (mods & GLFW_MOD_CONTROL) {
			render_option_ptr_->mesh_color_option_ =
					RenderOption::MeshColorOption::XCoordinate;
			UpdateGeometry();
			PrintDebug("[Visualizer] Mesh color set to X.\n");
		} else if (mods & GLFW_MOD_SHIFT) {
			SetGlobalColorMap(ColorMap::ColorMapOption::Summer);
			UpdateGeometry();
			PrintDebug("[Visualizer] Color map set to SUMMER.\n");
		} else {
			render_option_ptr_->point_color_option_ =
					RenderOption::PointColorOption::XCoordinate;
			UpdateGeometry();
			PrintDebug("[Visualizer] Point color set to X.\n");
		}
		break;
	case GLFW_KEY_3:
		if (mods & GLFW_MOD_CONTROL) {
			render_option_ptr_->mesh_color_option_ =
					RenderOption::MeshColorOption::YCoordinate;
			UpdateGeometry();
			PrintDebug("[Visualizer] Mesh color set to Y.\n");
		} else if (mods & GLFW_MOD_SHIFT) {
			SetGlobalColorMap(ColorMap::ColorMapOption::Winter);
			UpdateGeometry();
			PrintDebug("[Visualizer] Color map set to WINTER.\n");
		} else {
			render_option_ptr_->point_color_option_ =
					RenderOption::PointColorOption::YCoordinate;
			UpdateGeometry();
			PrintDebug("[Visualizer] Point color set to Y.\n");
		}
		break;
	case GLFW_KEY_4:
		if (mods & GLFW_MOD_CONTROL) {
			render_option_ptr_->mesh_color_option_ =
					RenderOption::MeshColorOption::ZCoordinate;
			UpdateGeometry();
			PrintDebug("[Visualizer] Mesh color set to Z.\n");
		} else if (mods & GLFW_MOD_SHIFT) {
			SetGlobalColorMap(ColorMap::ColorMapOption::Hot);
			UpdateGeometry();
			PrintDebug("[Visualizer] Color map set to HOT.\n");
		} else {
			render_option_ptr_->point_color_option_ =
					RenderOption::PointColorOption::ZCoordinate;
			UpdateGeometry();
			PrintDebug("[Visualizer] Point color set to Z.\n");
		}
		break;
	case GLFW_KEY_9:
		if (mods & GLFW_MOD_CONTROL) {
			render_option_ptr_->mesh_color_option_ =
					RenderOption::MeshColorOption::Normal;
			UpdateGeometry();
			PrintDebug("[Visualizer] Mesh color set to NORMAL.\n");
		} else if (mods & GLFW_MOD_SHIFT) {
		} else {
			render_option_ptr_->point_color_option_ =
					RenderOption::PointColorOption::Normal;
			UpdateGeometry();
			PrintDebug("[Visualizer] Point color set to NORMAL.\n");
		}
		break;
	default:
		break;
	}

	is_redraw_required_ = true;
}

void Visualizer::WindowCloseCallback(GLFWwindow *window)
{
	// happens when user click the close icon to close the window
}

}	// namespace three
