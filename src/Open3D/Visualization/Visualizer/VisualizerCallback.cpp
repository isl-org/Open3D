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

#include "Open3D/Visualization/Visualizer/Visualizer.h"

namespace open3d {
namespace visualization {

void Visualizer::WindowRefreshCallback(GLFWwindow *window) {
    if (is_redraw_required_) {
        Render();
        is_redraw_required_ = false;
    }
}

void Visualizer::WindowResizeCallback(GLFWwindow *window, int w, int h) {
    view_control_ptr_->ChangeWindowSize(w, h);
    is_redraw_required_ = true;
}

void Visualizer::MouseMoveCallback(GLFWwindow *window, double x, double y) {
#ifdef __APPLE__
    x /= pixel_to_screen_coordinate_;
    y /= pixel_to_screen_coordinate_;
#endif
    if (mouse_control_.is_mouse_left_button_down) {
        if (mouse_control_.is_control_key_down) {
            view_control_ptr_->Translate(x - mouse_control_.mouse_position_x,
                                         y - mouse_control_.mouse_position_y,
                                         mouse_control_.mouse_position_x,
                                         mouse_control_.mouse_position_y);
        } else if (mouse_control_.is_shift_key_down) {
            view_control_ptr_->Roll(x - mouse_control_.mouse_position_x);
        } else {
            view_control_ptr_->Rotate(x - mouse_control_.mouse_position_x,
                                      y - mouse_control_.mouse_position_y,
                                      mouse_control_.mouse_position_x,
                                      mouse_control_.mouse_position_y);
        }
        is_redraw_required_ = true;
    }
    if (mouse_control_.is_mouse_middle_button_down) {
        view_control_ptr_->Translate(x - mouse_control_.mouse_position_x,
                                     y - mouse_control_.mouse_position_y,
                                     mouse_control_.mouse_position_x,
                                     mouse_control_.mouse_position_y);
        is_redraw_required_ = true;
    }
    mouse_control_.mouse_position_x = x;
    mouse_control_.mouse_position_y = y;
}

void Visualizer::MouseScrollCallback(GLFWwindow *window, double x, double y) {
    view_control_ptr_->Scale(y);
    is_redraw_required_ = true;
}

void Visualizer::MouseButtonCallback(GLFWwindow *window,
                                     int button,
                                     int action,
                                     int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);
#ifdef __APPLE__
    x /= pixel_to_screen_coordinate_;
    y /= pixel_to_screen_coordinate_;
#endif
    mouse_control_.mouse_position_x = x;
    mouse_control_.mouse_position_y = y;
    if (action == GLFW_PRESS) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mouse_control_.is_mouse_left_button_down = true;
            mouse_control_.is_control_key_down = (mods & GLFW_MOD_CONTROL) != 0;
            mouse_control_.is_shift_key_down = (mods & GLFW_MOD_SHIFT) != 0;
            mouse_control_.is_alt_key_down = (mods & GLFW_MOD_ALT) != 0;
            mouse_control_.is_super_key_down = (mods & GLFW_MOD_SUPER) != 0;
        } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
            mouse_control_.is_mouse_middle_button_down = true;
        }
    } else {
        mouse_control_.is_mouse_left_button_down = false;
        mouse_control_.is_mouse_middle_button_down = false;
        mouse_control_.is_control_key_down = false;
        mouse_control_.is_shift_key_down = false;
        mouse_control_.is_alt_key_down = false;
        mouse_control_.is_super_key_down = false;
    }
}

void Visualizer::KeyPressCallback(
        GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_RELEASE) {
        return;
    }

    switch (key) {
        case GLFW_KEY_LEFT_BRACKET:
            view_control_ptr_->ChangeFieldOfView(-1.0);
            utility::LogDebug("[Visualizer] Field of view set to {:.2f}.",
                              view_control_ptr_->GetFieldOfView());
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            view_control_ptr_->ChangeFieldOfView(1.0);
            utility::LogDebug("[Visualizer] Field of view set to {:.2f}.",
                              view_control_ptr_->GetFieldOfView());
            break;
        case GLFW_KEY_R:
            ResetViewPoint();
            utility::LogDebug("[Visualizer] Reset view point.");
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
            utility::LogDebug("[Visualizer] Lighting {}.",
                              render_option_ptr_->light_on_ ? "ON" : "OFF");
            break;
        case GLFW_KEY_EQUAL:
            if (mods & GLFW_MOD_SHIFT) {
                render_option_ptr_->ChangeLineWidth(1.0);
                utility::LogDebug("[Visualizer] Line width set to {:.2f}.",
                                  render_option_ptr_->line_width_);
            } else {
                render_option_ptr_->ChangePointSize(1.0);
                if (render_option_ptr_->point_show_normal_) {
                    UpdateGeometry();
                }
                utility::LogDebug("[Visualizer] Point size set to {:.2f}.",
                                  render_option_ptr_->point_size_);
            }
            break;
        case GLFW_KEY_MINUS:
            if (mods & GLFW_MOD_SHIFT) {
                render_option_ptr_->ChangeLineWidth(-1.0);
                utility::LogDebug("[Visualizer] Line width set to {:.2f}.",
                                  render_option_ptr_->line_width_);
            } else {
                render_option_ptr_->ChangePointSize(-1.0);
                if (render_option_ptr_->point_show_normal_) {
                    UpdateGeometry();
                }
                utility::LogDebug("[Visualizer] Point size set to {:.2f}.",
                                  render_option_ptr_->point_size_);
            }
            break;
        case GLFW_KEY_N:
            render_option_ptr_->TogglePointShowNormal();
            if (render_option_ptr_->point_show_normal_) {
                UpdateGeometry();
            }
            utility::LogDebug(
                    "[Visualizer] Point normal rendering {}.",
                    render_option_ptr_->point_show_normal_ ? "ON" : "OFF");
            break;
        case GLFW_KEY_S:
            render_option_ptr_->ToggleShadingOption();
            UpdateGeometry();
            utility::LogDebug(
                    "[Visualizer] Mesh shading mode is {}.",
                    render_option_ptr_->mesh_shade_option_ ==
                                    RenderOption::MeshShadeOption::FlatShade
                            ? "FLAT"
                            : "SMOOTH");
            break;
        case GLFW_KEY_W:
            render_option_ptr_->ToggleMeshShowWireframe();
            utility::LogDebug(
                    "[Visualizer] Mesh wireframe rendering {}.",
                    render_option_ptr_->mesh_show_wireframe_ ? "ON" : "OFF");
            break;
        case GLFW_KEY_B:
            render_option_ptr_->ToggleMeshShowBackFace();
            utility::LogDebug(
                    "[Visualizer] Mesh back face rendering {}.",
                    render_option_ptr_->mesh_show_back_face_ ? "ON" : "OFF");
            break;
        case GLFW_KEY_I:
            render_option_ptr_->ToggleInterpolationOption();
            UpdateGeometry();
            utility::LogDebug(
                    "[Visualizer] geometry::Image interpolation mode is {}.",
                    render_option_ptr_->interpolation_option_ ==
                                    RenderOption::TextureInterpolationOption::
                                            Nearest
                            ? "NEARST"
                            : "LINEAR");
            break;
        case GLFW_KEY_T:
            render_option_ptr_->ToggleImageStretchOption();
            utility::LogDebug(
                    "[Visualizer] geometry::Image stretch mode is #{}.",
                    int(render_option_ptr_->image_stretch_option_));
            break;
        case GLFW_KEY_0:
            if (mods & GLFW_MOD_CONTROL) {
                render_option_ptr_->mesh_color_option_ =
                        RenderOption::MeshColorOption::Default;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Mesh color set to DEFAULT.");
            } else if (mods & GLFW_MOD_SHIFT) {
                SetGlobalColorMap(ColorMap::ColorMapOption::Gray);
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Color map set to GRAY.");
            } else {
                render_option_ptr_->point_color_option_ =
                        RenderOption::PointColorOption::Default;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Point color set to DEFAULT.");
            }
            break;
        case GLFW_KEY_1:
            if (mods & GLFW_MOD_CONTROL) {
                render_option_ptr_->mesh_color_option_ =
                        RenderOption::MeshColorOption::Color;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Mesh color set to COLOR.");
            } else if (mods & GLFW_MOD_SHIFT) {
                SetGlobalColorMap(ColorMap::ColorMapOption::Jet);
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Color map set to JET.");
            } else {
                render_option_ptr_->point_color_option_ =
                        RenderOption::PointColorOption::Color;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Point color set to COLOR.");
            }
            break;
        case GLFW_KEY_2:
            if (mods & GLFW_MOD_CONTROL) {
                render_option_ptr_->mesh_color_option_ =
                        RenderOption::MeshColorOption::XCoordinate;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Mesh color set to X.");
            } else if (mods & GLFW_MOD_SHIFT) {
                SetGlobalColorMap(ColorMap::ColorMapOption::Summer);
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Color map set to SUMMER.");
            } else {
                render_option_ptr_->point_color_option_ =
                        RenderOption::PointColorOption::XCoordinate;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Point color set to X.");
            }
            break;
        case GLFW_KEY_3:
            if (mods & GLFW_MOD_CONTROL) {
                render_option_ptr_->mesh_color_option_ =
                        RenderOption::MeshColorOption::YCoordinate;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Mesh color set to Y.");
            } else if (mods & GLFW_MOD_SHIFT) {
                SetGlobalColorMap(ColorMap::ColorMapOption::Winter);
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Color map set to WINTER.");
            } else {
                render_option_ptr_->point_color_option_ =
                        RenderOption::PointColorOption::YCoordinate;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Point color set to Y.");
            }
            break;
        case GLFW_KEY_4:
            if (mods & GLFW_MOD_CONTROL) {
                render_option_ptr_->mesh_color_option_ =
                        RenderOption::MeshColorOption::ZCoordinate;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Mesh color set to Z.");
            } else if (mods & GLFW_MOD_SHIFT) {
                SetGlobalColorMap(ColorMap::ColorMapOption::Hot);
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Color map set to HOT.");
            } else {
                render_option_ptr_->point_color_option_ =
                        RenderOption::PointColorOption::ZCoordinate;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Point color set to Z.");
            }
            break;
        case GLFW_KEY_9:
            if (mods & GLFW_MOD_CONTROL) {
                render_option_ptr_->mesh_color_option_ =
                        RenderOption::MeshColorOption::Normal;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Mesh color set to NORMAL.");
            } else if (mods & GLFW_MOD_SHIFT) {
            } else {
                render_option_ptr_->point_color_option_ =
                        RenderOption::PointColorOption::Normal;
                UpdateGeometry();
                utility::LogDebug("[Visualizer] Point color set to NORMAL.");
            }
            break;
        default:
            break;
    }

    is_redraw_required_ = true;
}

void Visualizer::WindowCloseCallback(GLFWwindow *window) {
    // happens when user click the close icon to close the window
}

}  // namespace visualization
}  // namespace open3d
