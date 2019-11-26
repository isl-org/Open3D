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

#include "Open3D/Visualization/Visualizer/VisualizerWithEditing.h"

#include <tinyfiledialogs/tinyfiledialogs.h>

#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/IO/ClassIO/IJsonConvertibleIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Utility/FileSystem.h"
#include "Open3D/Visualization/Utility/GLHelper.h"
#include "Open3D/Visualization/Utility/PointCloudPicker.h"
#include "Open3D/Visualization/Utility/SelectionPolygon.h"
#include "Open3D/Visualization/Utility/SelectionPolygonVolume.h"
#include "Open3D/Visualization/Visualizer/RenderOptionWithEditing.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithEditing.h"

namespace open3d {
namespace visualization {

bool VisualizerWithEditing::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr,
        bool reset_bounding_box) {
    if (is_initialized_ == false || geometry_ptrs_.empty() == false) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    original_geometry_ptr_ = geometry_ptr;
    if (geometry_ptr->GetGeometryType() ==
        geometry::Geometry::GeometryType::Unspecified) {
        return false;
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::PointCloud) {
        auto ptr = std::make_shared<geometry::PointCloud>();
        *ptr = (const geometry::PointCloud &)*original_geometry_ptr_;
        editing_geometry_ptr_ = ptr;
        editing_geometry_renderer_ptr_ =
                std::make_shared<glsl::PointCloudRenderer>();
        if (editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::LineSet) {
        auto ptr = std::make_shared<geometry::LineSet>();
        *ptr = (const geometry::LineSet &)*original_geometry_ptr_;
        editing_geometry_ptr_ = ptr;
        editing_geometry_renderer_ptr_ =
                std::make_shared<glsl::LineSetRenderer>();
        if (editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
                       geometry::Geometry::GeometryType::TriangleMesh ||
               geometry_ptr->GetGeometryType() ==
                       geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        auto ptr = std::make_shared<geometry::TriangleMesh>();
        *ptr = (const geometry::TriangleMesh &)*original_geometry_ptr_;
        editing_geometry_ptr_ = ptr;
        editing_geometry_renderer_ptr_ =
                std::make_shared<glsl::TriangleMeshRenderer>();
        if (editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Image) {
        auto ptr = std::make_shared<geometry::Image>();
        *ptr = (const geometry::Image &)*original_geometry_ptr_;
        editing_geometry_ptr_ = ptr;
        editing_geometry_renderer_ptr_ =
                std::make_shared<glsl::ImageRenderer>();
        if (editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_) == false) {
            return false;
        }
    } else {
        return false;
    }
    geometry_ptrs_.insert(editing_geometry_ptr_);
    geometry_renderer_ptrs_.insert(editing_geometry_renderer_ptr_);
    if (reset_bounding_box) {
        ResetViewPoint(true);
    }
    utility::LogDebug(
            "Add geometry and update bounding box to {}",
            view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
    return UpdateGeometry();
}

void VisualizerWithEditing::PrintVisualizerHelp() {
    Visualizer::PrintVisualizerHelp();
    // clang-format off
    utility::LogInfo("  -- Editing control --");
    utility::LogInfo("    F            : Enter freeview mode.");
    utility::LogInfo("    X            : Enter orthogonal view along X axis, press again to flip.");
    utility::LogInfo("    Y            : Enter orthogonal view along Y axis, press again to flip.");
    utility::LogInfo("    Z            : Enter orthogonal view along Z axis, press again to flip.");
    utility::LogInfo("    K            : Lock / unlock camera.");
    utility::LogInfo("    Ctrl + D     : Downsample point cloud with a voxel grid.");
    utility::LogInfo("    Ctrl + R     : Reset geometry to its initial state.");
    utility::LogInfo("    Shift + +/-  : Increase/decrease picked point size..");
    utility::LogInfo("    Shift + mouse left button   : Pick a point and add in queue.");
    utility::LogInfo("    Shift + mouse right button  : Remove last picked point from queue.");
    utility::LogInfo("");
    utility::LogInfo("    -- When camera is locked --");
    utility::LogInfo("    Mouse left button + drag    : Create a selection rectangle.");
    utility::LogInfo("    Ctrl + mouse buttons + drag : Hold Ctrl key to draw a selection polygon.");
    utility::LogInfo("                                  Left mouse button to add point. Right mouse");
    utility::LogInfo("                                  button to remove point. Release Ctrl key to");
    utility::LogInfo("                                  close the polygon.");
    utility::LogInfo("    C                           : Crop the geometry with selection region.");
    utility::LogInfo("");
    // clang-format on
}

void VisualizerWithEditing::UpdateWindowTitle() {
    if (window_ != NULL) {
        auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
        std::string new_window_title =
                window_name_ + " - " + view_control.GetStatusString();
        glfwSetWindowTitle(window_, new_window_title.c_str());
    }
}

void VisualizerWithEditing::BuildUtilities() {
    Visualizer::BuildUtilities();
    bool success;

    // 1. Build selection polygon
    success = true;
    selection_polygon_ptr_ = std::make_shared<SelectionPolygon>();
    selection_polygon_renderer_ptr_ =
            std::make_shared<glsl::SelectionPolygonRenderer>();
    if (selection_polygon_renderer_ptr_->AddGeometry(selection_polygon_ptr_) ==
        false) {
        success = false;
    }
    if (success) {
        utility_ptrs_.push_back(selection_polygon_ptr_);
        utility_renderer_ptrs_.push_back(selection_polygon_renderer_ptr_);
    }

    // 2. Build pointcloud picker
    success = true;
    pointcloud_picker_ptr_ = std::make_shared<PointCloudPicker>();
    if (geometry_ptrs_.empty() ||
        pointcloud_picker_ptr_->SetPointCloud(editing_geometry_ptr_) == false) {
        success = false;
    }
    pointcloud_picker_renderer_ptr_ =
            std::make_shared<glsl::PointCloudPickerRenderer>();
    if (pointcloud_picker_renderer_ptr_->AddGeometry(pointcloud_picker_ptr_) ==
        false) {
        success = false;
    }
    if (success) {
        utility_ptrs_.push_back(pointcloud_picker_ptr_);
        utility_renderer_ptrs_.push_back(pointcloud_picker_renderer_ptr_);
    }
}

int VisualizerWithEditing::PickPoint(double x, double y) {
    auto renderer_ptr = std::make_shared<glsl::PointCloudPickingRenderer>();
    if (renderer_ptr->AddGeometry(editing_geometry_ptr_) == false) {
        return -1;
    }
    const auto &view = GetViewControl();
    // Render to FBO and disable anti-aliasing
    glDisable(GL_MULTISAMPLE);
    GLuint frame_buffer_name = 0;
    glGenFramebuffers(1, &frame_buffer_name);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_name);
    GLuint fbo_texture;
    glGenTextures(1, &fbo_texture);
    glBindTexture(GL_TEXTURE_2D, fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, view.GetWindowWidth(),
                 view.GetWindowHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (!GLEW_ARB_framebuffer_object) {
        // OpenGL 2.1 doesn't require this, 3.1+ does
        utility::LogWarning(
                "[PickPoint] Your GPU does not provide framebuffer objects. "
                "Use a texture instead.");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glEnable(GL_MULTISAMPLE);
        return -1;
    }
    GLuint depth_render_buffer;
    glGenRenderbuffers(1, &depth_render_buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_render_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
                          view.GetWindowWidth(), view.GetWindowHeight());
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, depth_render_buffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           fbo_texture, 0);
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);  // "1" is the size of DrawBuffers
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        utility::LogWarning("[PickPoint] Something is wrong with FBO.");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glEnable(GL_MULTISAMPLE);
        return -1;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_name);
    view_control_ptr_->SetViewMatrices();
    glDisable(GL_BLEND);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderer_ptr->Render(GetRenderOption(), GetViewControl());
    glFinish();
    uint8_t rgba[4];
    glReadPixels((int)(x + 0.5), (int)(view.GetWindowHeight() - y + 0.5), 1, 1,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    int index = GLHelper::ColorCodeToPickIndex(
            Eigen::Vector4i(rgba[0], rgba[1], rgba[2], rgba[3]));
    // Recover rendering state
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_MULTISAMPLE);
    return index;
}

std::vector<size_t> &VisualizerWithEditing::GetPickedPoints() {
    return pointcloud_picker_ptr_->picked_indices_;
}

bool VisualizerWithEditing::InitViewControl() {
    view_control_ptr_ =
            std::unique_ptr<ViewControlWithEditing>(new ViewControlWithEditing);
    ResetViewPoint();
    return true;
}

bool VisualizerWithEditing::InitRenderOption() {
    render_option_ptr_ = std::unique_ptr<RenderOptionWithEditing>(
            new RenderOptionWithEditing);
    return true;
}

void VisualizerWithEditing::KeyPressCallback(
        GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
    auto &option = (RenderOptionWithEditing &)(*render_option_ptr_);
    if (action == GLFW_RELEASE) {
        if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
            if (view_control.IsLocked() &&
                selection_mode_ == SelectionMode::Polygon) {
                selection_mode_ = SelectionMode::None;
                selection_polygon_ptr_->polygon_.pop_back();
                if (selection_polygon_ptr_->IsEmpty()) {
                    selection_polygon_ptr_->Clear();
                } else {
                    selection_polygon_ptr_->FillPolygon(
                            view_control.GetWindowWidth(),
                            view_control.GetWindowHeight());
                    selection_polygon_ptr_->polygon_type_ =
                            SelectionPolygon::SectionPolygonType::Polygon;
                }
                selection_polygon_renderer_ptr_->UpdateGeometry();
                is_redraw_required_ = true;
            }
        }
        return;
    }

    switch (key) {
        case GLFW_KEY_F:
            view_control.SetEditingMode(
                    ViewControlWithEditing::EditingMode::FreeMode);
            utility::LogDebug("[Visualizer] Enter freeview mode.");
            break;
        case GLFW_KEY_X:
            view_control.ToggleEditingX();
            utility::LogDebug("[Visualizer] Enter orthogonal X editing mode.");
            break;
        case GLFW_KEY_Y:
            view_control.ToggleEditingY();
            utility::LogDebug("[Visualizer] Enter orthogonal Y editing mode.");
            break;
        case GLFW_KEY_Z:
            view_control.ToggleEditingZ();
            utility::LogDebug("[Visualizer] Enter orthogonal Z editing mode.");
            break;
        case GLFW_KEY_K:
            view_control.ToggleLocking();
            InvalidateSelectionPolygon();
            utility::LogDebug("[Visualizer] Camera %s.",
                              view_control.IsLocked() ? "Lock" : "Unlock");
            break;
        case GLFW_KEY_R:
            if (mods & GLFW_MOD_CONTROL) {
                (geometry::PointCloud &)*editing_geometry_ptr_ =
                        (const geometry::PointCloud &)*original_geometry_ptr_;
                editing_geometry_renderer_ptr_->UpdateGeometry();
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
            }
            break;
        case GLFW_KEY_D:
            if (mods & GLFW_MOD_CONTROL) {
                if (use_dialog_) {
                    std::string buff;
                    buff = fmt::format("{:.4f}", voxel_size_);
                    const char *str = tinyfd_inputBox(
                            "Set voxel size",
                            "Set voxel size (ignored if it is non-positive)",
                            buff.c_str());
                    if (str == NULL) {
                        utility::LogWarning(
                                "Illegal input, using default voxel size.");
                    } else {
                        char *end;
                        errno = 0;
                        double l = std::strtod(str, &end);
                        if (errno == ERANGE &&
                            (l == HUGE_VAL || l == -HUGE_VAL)) {
                            utility::LogWarning(
                                    "Illegal input, using default voxel "
                                    "size.");
                        } else {
                            voxel_size_ = l;
                        }
                    }
                }
                if (voxel_size_ > 0.0 && editing_geometry_ptr_ &&
                    editing_geometry_ptr_->GetGeometryType() ==
                            geometry::Geometry::GeometryType::PointCloud) {
                    utility::LogInfo("Voxel downsample with voxel size {:.4f}.",
                                     voxel_size_);
                    geometry::PointCloud &pcd =
                            (geometry::PointCloud &)*editing_geometry_ptr_;
                    pcd = *pcd.VoxelDownSample(voxel_size_);
                    UpdateGeometry();
                } else {
                    utility::LogWarning(
                            "No voxel downsample performed due to illegal "
                            "voxel size.");
                }
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
            }
            break;
        case GLFW_KEY_C:
            if (view_control.IsLocked() && selection_polygon_ptr_) {
                if (editing_geometry_ptr_ &&
                    editing_geometry_ptr_->GetGeometryType() ==
                            geometry::Geometry::GeometryType::PointCloud) {
                    glfwMakeContextCurrent(window_);
                    geometry::PointCloud &pcd =
                            (geometry::PointCloud &)*editing_geometry_ptr_;
                    pcd = *selection_polygon_ptr_->CropPointCloud(pcd,
                                                                  view_control);
                    editing_geometry_renderer_ptr_->UpdateGeometry();
                    const char *filename;
                    const char *pattern[1] = {"*.ply"};
                    std::string default_filename =
                            default_directory_ + "cropped_" +
                            std::to_string(crop_action_count_ + 1) + ".ply";
                    if (use_dialog_) {
                        filename = tinyfd_saveFileDialog(
                                "geometry::PointCloud file",
                                default_filename.c_str(), 1, pattern,
                                "Polygon File Format (*.ply)");
                    } else {
                        filename = default_filename.c_str();
                    }
                    if (filename == NULL) {
                        utility::LogWarning(
                                "No filename is given. Abort saving.");
                    } else {
                        SaveCroppingResult(filename);
                        crop_action_count_++;
                    }
                    view_control.ToggleLocking();
                    InvalidateSelectionPolygon();
                    InvalidatePicking();
                } else if (editing_geometry_ptr_ &&
                           editing_geometry_ptr_->GetGeometryType() ==
                                   geometry::Geometry::GeometryType::
                                           TriangleMesh) {
                    glfwMakeContextCurrent(window_);
                    geometry::TriangleMesh &mesh =
                            (geometry::TriangleMesh &)*editing_geometry_ptr_;
                    mesh = *selection_polygon_ptr_->CropTriangleMesh(
                            mesh, view_control);
                    editing_geometry_renderer_ptr_->UpdateGeometry();
                    const char *filename;
                    const char *pattern[1] = {"*.ply"};
                    std::string default_filename =
                            default_directory_ + "cropped_" +
                            std::to_string(crop_action_count_ + 1) + ".ply";
                    if (use_dialog_) {
                        filename = tinyfd_saveFileDialog(
                                "Mesh file", default_filename.c_str(), 1,
                                pattern, "Polygon File Format (*.ply)");
                    } else {
                        filename = default_filename.c_str();
                    }
                    if (filename == NULL) {
                        utility::LogWarning(
                                "No filename is given. Abort saving.");
                    } else {
                        SaveCroppingResult(filename);
                        crop_action_count_++;
                    }
                    view_control.ToggleLocking();
                    InvalidateSelectionPolygon();
                    InvalidatePicking();
                }
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
            }
            break;
        case GLFW_KEY_MINUS:
            if (mods & GLFW_MOD_SHIFT) {
                option.DecreaseSphereSize();
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
            }
            break;
        case GLFW_KEY_EQUAL:
            if (mods & GLFW_MOD_SHIFT) {
                option.IncreaseSphereSize();
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
            }
            break;
        default:
            Visualizer::KeyPressCallback(window, key, scancode, action, mods);
            break;
    }
    is_redraw_required_ = true;
    UpdateWindowTitle();
}

void VisualizerWithEditing::WindowResizeCallback(GLFWwindow *window,
                                                 int w,
                                                 int h) {
    InvalidateSelectionPolygon();
    Visualizer::WindowResizeCallback(window, w, h);
}

void VisualizerWithEditing::MouseMoveCallback(GLFWwindow *window,
                                              double x,
                                              double y) {
    auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
    if (view_control.IsLocked()) {
#ifdef __APPLE__
        x /= pixel_to_screen_coordinate_;
        y /= pixel_to_screen_coordinate_;
#endif
        double y_inv = view_control.GetWindowHeight() - y;
        if (selection_mode_ == SelectionMode::None) {
        } else if (selection_mode_ == SelectionMode::Rectangle) {
            selection_polygon_ptr_->polygon_[1](0) = x;
            selection_polygon_ptr_->polygon_[2](0) = x;
            selection_polygon_ptr_->polygon_[2](1) = y_inv;
            selection_polygon_ptr_->polygon_[3](1) = y_inv;
            selection_polygon_renderer_ptr_->UpdateGeometry();
            is_redraw_required_ = true;
        } else if (selection_mode_ == SelectionMode::Polygon) {
            selection_polygon_ptr_->polygon_.back() = Eigen::Vector2d(x, y_inv);
            selection_polygon_renderer_ptr_->UpdateGeometry();
            is_redraw_required_ = true;
        }
    } else {
        Visualizer::MouseMoveCallback(window, x, y);
    }
}

void VisualizerWithEditing::MouseScrollCallback(GLFWwindow *window,
                                                double x,
                                                double y) {
    auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
    if (view_control.IsLocked()) {
    } else {
        Visualizer::MouseScrollCallback(window, x, y);
    }
}

void VisualizerWithEditing::MouseButtonCallback(GLFWwindow *window,
                                                int button,
                                                int action,
                                                int mods) {
    auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
    if (view_control.IsLocked() && selection_polygon_ptr_ &&
        selection_polygon_renderer_ptr_) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            double x, y;
            glfwGetCursorPos(window, &x, &y);
#ifdef __APPLE__
            x /= pixel_to_screen_coordinate_;
            y /= pixel_to_screen_coordinate_;
#endif
            if (action == GLFW_PRESS) {
                double y_inv = view_control.GetWindowHeight() - y;
                if (selection_mode_ == SelectionMode::None) {
                    InvalidateSelectionPolygon();
                    if (mods & GLFW_MOD_CONTROL) {
                        selection_mode_ = SelectionMode::Polygon;
                        selection_polygon_ptr_->polygon_.push_back(
                                Eigen::Vector2d(x, y_inv));
                        selection_polygon_ptr_->polygon_.push_back(
                                Eigen::Vector2d(x, y_inv));
                    } else {
                        selection_mode_ = SelectionMode::Rectangle;
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
                } else if (selection_mode_ == SelectionMode::Rectangle) {
                } else if (selection_mode_ == SelectionMode::Polygon) {
                    if (mods & GLFW_MOD_CONTROL) {
                        selection_polygon_ptr_->polygon_.back() =
                                Eigen::Vector2d(x, y_inv);
                        selection_polygon_ptr_->polygon_.push_back(
                                Eigen::Vector2d(x, y_inv));
                        selection_polygon_renderer_ptr_->UpdateGeometry();
                    }
                }
            } else if (action == GLFW_RELEASE) {
                if (selection_mode_ == SelectionMode::None) {
                } else if (selection_mode_ == SelectionMode::Rectangle) {
                    selection_mode_ = SelectionMode::None;
                    selection_polygon_ptr_->FillPolygon(
                            view_control.GetWindowWidth(),
                            view_control.GetWindowHeight());
                    selection_polygon_ptr_->polygon_type_ =
                            SelectionPolygon::SectionPolygonType::Rectangle;
                    selection_polygon_renderer_ptr_->UpdateGeometry();
                } else if (selection_mode_ == SelectionMode::Polygon) {
                }
            }
            is_redraw_required_ = true;
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS &&
                selection_mode_ == SelectionMode::Polygon &&
                (mods & GLFW_MOD_CONTROL)) {
                if (selection_polygon_ptr_->polygon_.size() > 2) {
                    selection_polygon_ptr_
                            ->polygon_[selection_polygon_ptr_->polygon_.size() -
                                       2] =
                            selection_polygon_ptr_->polygon_
                                    [selection_polygon_ptr_->polygon_.size() -
                                     1];
                    selection_polygon_ptr_->polygon_.pop_back();
                    selection_polygon_renderer_ptr_->UpdateGeometry();
                    is_redraw_required_ = true;
                }
            }
        }
    } else {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE &&
            (mods & GLFW_MOD_SHIFT)) {
            double x, y;
            glfwGetCursorPos(window, &x, &y);
#ifdef __APPLE__
            x /= pixel_to_screen_coordinate_;
            y /= pixel_to_screen_coordinate_;
#endif
            int index = PickPoint(x, y);
            if (index == -1) {
                utility::LogInfo("No point has been picked.");
            } else {
                const auto &point =
                        ((const geometry::PointCloud &)(*editing_geometry_ptr_))
                                .points_[index];
                utility::LogInfo(
                        "Picked point #{:d} ({:.2}, {:.2}, {:.2}) to add in "
                        "queue.",
                        index, point(0), point(1), point(2));
                pointcloud_picker_ptr_->picked_indices_.push_back(
                        (size_t)index);
                is_redraw_required_ = true;
            }
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT &&
                   action == GLFW_RELEASE && (mods & GLFW_MOD_SHIFT)) {
            if (pointcloud_picker_ptr_->picked_indices_.empty() == false) {
                utility::LogInfo(
                        "Remove picked point #{} from pick queue.",
                        pointcloud_picker_ptr_->picked_indices_.back());
                pointcloud_picker_ptr_->picked_indices_.pop_back();
                is_redraw_required_ = true;
            }
        }
        Visualizer::MouseButtonCallback(window, button, action, mods);
    }
}

void VisualizerWithEditing::InvalidateSelectionPolygon() {
    if (selection_polygon_ptr_) selection_polygon_ptr_->Clear();
    if (selection_polygon_renderer_ptr_) {
        selection_polygon_renderer_ptr_->UpdateGeometry();
    }
    selection_mode_ = SelectionMode::None;
}

void VisualizerWithEditing::InvalidatePicking() {
    if (pointcloud_picker_ptr_) pointcloud_picker_ptr_->Clear();
    if (pointcloud_picker_renderer_ptr_) {
        pointcloud_picker_renderer_ptr_->UpdateGeometry();
    }
}

void VisualizerWithEditing::SaveCroppingResult(
        const std::string &filename /* = ""*/) {
    std::string ply_filename = filename;
    if (ply_filename.empty()) {
        ply_filename = "CroppedGeometry.ply";
    }
    std::string volume_filename =
            utility::filesystem::GetFileNameWithoutExtension(filename) +
            ".json";
    if (editing_geometry_ptr_->GetGeometryType() ==
        geometry::Geometry::GeometryType::PointCloud)
        io::WritePointCloud(
                ply_filename,
                (const geometry::PointCloud &)(*editing_geometry_ptr_));
    else if (editing_geometry_ptr_->GetGeometryType() ==
                     geometry::Geometry::GeometryType::TriangleMesh ||
             editing_geometry_ptr_->GetGeometryType() ==
                     geometry::Geometry::GeometryType::HalfEdgeTriangleMesh)
        io::WriteTriangleMesh(
                ply_filename,
                (const geometry::TriangleMesh &)(*editing_geometry_ptr_));
    io::WriteIJsonConvertible(
            volume_filename,
            *selection_polygon_ptr_->CreateSelectionPolygonVolume(
                    GetViewControl()));
}

}  // namespace visualization
}  // namespace open3d
