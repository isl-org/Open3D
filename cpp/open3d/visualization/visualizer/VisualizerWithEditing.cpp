// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/visualization/visualizer/VisualizerWithEditing.h"

#include <tinyfiledialogs/tinyfiledialogs.h>

#include "open3d/geometry/Image.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/GLHelper.h"
#include "open3d/visualization/utility/PointCloudPicker.h"
#include "open3d/visualization/utility/SelectionPolygon.h"
#include "open3d/visualization/utility/SelectionPolygonVolume.h"
#include "open3d/visualization/visualizer/RenderOptionWithEditing.h"
#include "open3d/visualization/visualizer/ViewControlWithEditing.h"
#include <map>

namespace open3d {
namespace visualization {

std::shared_ptr<geometry::Geometry>
VisualizerWithEditing::GetEditingGeometry() {
    return editing_geometry_ptr_;
}
std::shared_ptr<geometry::Geometry>
VisualizerWithEditing::GetDiscardedGeometry() {
    if (discarded_geometries_.empty()) {
        return nullptr;
    }

    auto geo = discarded_geometries_[0];
    if (geo->GetGeometryType() ==
        geometry::Geometry::GeometryType::PointCloud) {
        auto pcd = std::make_shared<geometry::PointCloud>();
        for (auto &other : discarded_geometries_) {
            *pcd += (geometry::PointCloud &)*other;
        }
        return pcd;
    } else if (geo->GetGeometryType() ==
               geometry::Geometry::GeometryType::TriangleMesh) {
        auto mesh = std::make_shared<geometry::TriangleMesh>();
        for (auto &other : discarded_geometries_) {
            *mesh += (geometry::TriangleMesh &)*other;
        }
        return mesh;
    }
    return nullptr;
}
void VisualizerWithEditing::Undo() {
    if (!editing_geometry_ptr_ || discarded_geometries_.empty()) {
        return;
    }
    auto last = discarded_geometries_.back();
    discarded_geometries_.pop_back();

    if (last->GetGeometryType() ==
        geometry::Geometry::GeometryType::PointCloud) {
        auto &pcd = (geometry::PointCloud &)*editing_geometry_ptr_;
        pcd += (geometry::PointCloud &)*last;
        editing_geometry_renderer_ptr_->UpdateGeometry();
    } else if (last->GetGeometryType() ==
               geometry::Geometry::GeometryType::TriangleMesh) {
        auto &mesh = (geometry::TriangleMesh &)*editing_geometry_ptr_;
        mesh += (geometry::TriangleMesh &)*last;
        editing_geometry_renderer_ptr_->UpdateGeometry();
    }
}

bool VisualizerWithEditing::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr,
        bool reset_bounding_box) {
    if (!is_initialized_ || !geometry_ptrs_.empty()) {
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
        if (!editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::LineSet) {
        auto ptr = std::make_shared<geometry::LineSet>();
        *ptr = (const geometry::LineSet &)*original_geometry_ptr_;
        editing_geometry_ptr_ = ptr;
        editing_geometry_renderer_ptr_ =
                std::make_shared<glsl::LineSetRenderer>();
        if (!editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_)) {
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
        if (!editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Image) {
        auto ptr = std::make_shared<geometry::Image>();
        *ptr = (const geometry::Image &)*original_geometry_ptr_;
        editing_geometry_ptr_ = ptr;
        editing_geometry_renderer_ptr_ =
                std::make_shared<glsl::ImageRenderer>();
        if (!editing_geometry_renderer_ptr_->AddGeometry(
                    editing_geometry_ptr_)) {
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
    utility::LogInfo("    K/E          : Lock / unlock camera.");
    utility::LogInfo("    G            : Enter / Exit selection mode");
    utility::LogInfo("    Ctrl + D     : Downsample point cloud with a voxel grid.");
    utility::LogInfo("    Ctrl + R     : Reset geometry to its initial state.");
    utility::LogInfo("    Ctrl + Z     : Undo latest editing.");
    utility::LogInfo("    Ctrl + S     : Save editing geometry to file.");
    utility::LogInfo("    Ctrl + F     : Print point cloud points");
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
    utility::LogInfo("    ESC                         : Unlock camera.");
    utility::LogInfo("    -- When entering selection mode --");
    utility::LogInfo("    F            : Select a plane, requires at least 3 picked points");
    utility::LogInfo("    -- When camera is locked or entered selection mode --");
    utility::LogInfo("    C            : Crop the geometry with selection region.");
    utility::LogInfo("    X            : Remove points within selection region.");
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
    if (!selection_polygon_renderer_ptr_->AddGeometry(selection_polygon_ptr_)) {
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
        !pointcloud_picker_ptr_->SetPointCloud(editing_geometry_ptr_)) {
        success = false;
    }
    pointcloud_picker_renderer_ptr_ =
            std::make_shared<glsl::PointCloudPickerRenderer>();
    if (!pointcloud_picker_renderer_ptr_->AddGeometry(pointcloud_picker_ptr_)) {
        success = false;
    }
    if (success) {
        utility_ptrs_.push_back(pointcloud_picker_ptr_);
        utility_renderer_ptrs_.push_back(pointcloud_picker_renderer_ptr_);
    }
}

int VisualizerWithEditing::PickPoint(double x, double y) {
    auto renderer_ptr = std::make_shared<glsl::PointCloudPickingRenderer>();
    if (!renderer_ptr->AddGeometry(editing_geometry_ptr_)) {
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
    int index = gl_util::ColorCodeToPickIndex(
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

void VisualizerWithEditing::UpdateBackground() {
    auto &bg = GetRenderOption().background_color_;
    auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
    if (select_editing_) {
        // 96, 109, 114
        bg[0] = 96.0/255.0;
        bg[1] = 109.0/255.0;
        bg[2] = 114.0/255.0;
    }
    else if (view_control.IsLocked()) {
        bg[0] = 0;
        bg[1] = 0.5;
        bg[2] = 0.5;
    } else {
        bg[0] = 1;
        bg[1] = 1;
        bg[2] = 1;
    }
    UpdateRender();
}
bool VisualizerWithEditing::InitRenderOption() {
    render_option_ptr_ = std::unique_ptr<RenderOptionWithEditing>(
            new RenderOptionWithEditing);
    return true;
}
void VisualizerWithEditing::Save() {
    if (!editing_geometry_ptr_) {
        return;
    }

    auto default_filename = save_file_path_;
    const char *filename = default_filename.c_str();
    if (default_filename.length() < 5 ||
        default_filename.substr(default_filename.length() - 4) != ".ply") {
        if (default_filename.empty()) {
            default_filename = utility::filesystem::GetWorkingDirectory();
        }
        default_filename += "/cropped.ply";
        const char *pattern[1] = {"*.ply"};
        switch (editing_geometry_ptr_->GetGeometryType()) {
            case geometry::Geometry::GeometryType::PointCloud:
                filename = tinyfd_saveFileDialog(
                        "geometry::PointCloud file", default_filename.c_str(),
                        1, pattern, "Polygon File Format (*.ply)");
                break;
            case geometry::Geometry::GeometryType::TriangleMesh:
                filename = tinyfd_saveFileDialog(
                        "Mesh file", default_filename.c_str(), 1, pattern,
                        "Polygon File Format (*.ply)");
                break;
            default:
                return;
        }
    }

    if (filename == nullptr) {
        utility::LogWarning("No filename is given. Abort saving.");
    } else {
        SaveCroppingResult(filename);
    }
}
std::shared_ptr<geometry::Geometry> VisualizerWithEditing::Crop(std::vector<size_t> &indexes,
                                 bool del /* del = true indicates indexes should be deleted */
                                 ) {
    if (editing_geometry_ptr_ &&
        editing_geometry_ptr_->GetGeometryType() ==
        geometry::Geometry::GeometryType::PointCloud) {
        auto &pcd = (geometry::PointCloud &)*editing_geometry_ptr_;
        auto keep = pcd.SelectByIndex(indexes, del);
        auto left = pcd.SelectByIndex(indexes, !del);
        pcd = *keep;
        editing_geometry_renderer_ptr_->UpdateGeometry();
        return left;
    } else if (editing_geometry_ptr_ &&
               editing_geometry_ptr_->GetGeometryType() ==
               geometry::Geometry::GeometryType::TriangleMesh) {
        auto &mesh = (geometry::TriangleMesh &)*editing_geometry_ptr_;
        auto keep = mesh.SelectByIndex(indexes, del);
        auto left = mesh.SelectByIndex(indexes, !del);
        mesh = *keep;
        editing_geometry_renderer_ptr_->UpdateGeometry();
        return left;
    }
    return nullptr;
}

void VisualizerWithEditing::Crop(bool del) {
    glfwMakeContextCurrent(window_);
    std::vector<size_t> indexes;
    auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
    if (editing_geometry_ptr_ &&
        editing_geometry_ptr_->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
        auto &pcd = (geometry::PointCloud &)*editing_geometry_ptr_;
        indexes = selection_polygon_ptr_->CropPointCloudIndex(pcd, view_control);
    } else if (editing_geometry_ptr_ &&
               editing_geometry_ptr_->GetGeometryType() ==
                       geometry::Geometry::GeometryType::TriangleMesh) {
        auto &mesh = (geometry::TriangleMesh &)*editing_geometry_ptr_;
        indexes = selection_polygon_ptr_->CropTriangleMeshIndex(mesh, view_control);
    }
    if (!indexes.empty()) {
        auto geo = Crop(indexes, del);
        if (geo) {
            discarded_geometries_.push_back(geo);
        }
    }
    InvalidateSelectionPolygon();
    InvalidatePicking();
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
        case GLFW_KEY_G:
            if (!view_control.IsLocked()) {
                if (!select_editing_) {
                    utility::LogInfo("Enter select-editing, you may select some points before editing");
                    PointSelectionHint();
                    select_editing_ = true;
                    Backup();
                    UpdateBackground();
                } else {
                    ExitSelectEdit();
                }
            }
            break;
        case GLFW_KEY_ENTER:
            Visualizer::KeyPressCallback(window, key, scancode, action, mods);
            break;

        case GLFW_KEY_F:
            if (mods & GLFW_MOD_CONTROL) {
                if (editing_geometry_ptr_ &&
                        editing_geometry_ptr_->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
                    auto &pcd = (geometry::PointCloud &)*editing_geometry_ptr_;
                    utility::LogInfo("current editing point cloud has {} points", pcd.points_.size());
                }
                if (original_geometry_ptr_ &&
                    original_geometry_ptr_->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
                    auto &pcd = (geometry::PointCloud &)*original_geometry_ptr_;
                    utility::LogInfo("original editing point cloud has {} points", pcd.points_.size());
                }
                for (auto i = 0u; i < selected_geometries_.size(); i++) {
                    auto geo = selected_geometries_[i];
                    if (geo && geo->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
                        auto &pcd = (geometry::PointCloud &)*geo;
                        utility::LogInfo("selected point cloud {} has {} points", i, pcd.points_.size());
                    }
                }
                for (auto i = 0u; i < discarded_geometries_.size(); i++) {
                    auto geo = discarded_geometries_[i];
                    if (geo && geo->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
                        auto &pcd = (geometry::PointCloud &)*geo;
                        utility::LogInfo("history point cloud {} has {} points", i, pcd.points_.size());
                    }
                }
            } else if (select_editing_) {
                FitPlane();
            } else {
                view_control.SetEditingMode(ViewControlWithEditing::EditingMode::FreeMode);
                utility::LogDebug("[Visualizer] Enter freeview mode.");
            }
            break;
        case GLFW_KEY_S:
            if (mods & GLFW_MOD_CONTROL) {
                Save();
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
            }
            break;
        case GLFW_KEY_X:
            if (select_editing_) {
                CropSelected(true);
            } else if (view_control.IsLocked()) {
                if (selection_polygon_ptr_) {
                    Crop(true);
                } else {
                    utility::LogDebug("No polygon");
                }
            } else {
                view_control.ToggleEditingX();
                utility::LogDebug(
                        "[Visualizer] Enter orthogonal X editing mode.");
            }
            break;
        case GLFW_KEY_Y:
            view_control.ToggleEditingY();
            utility::LogDebug("[Visualizer] Enter orthogonal Y editing mode.");
            break;
        case GLFW_KEY_Z:
            if (mods & GLFW_MOD_CONTROL) {
                Undo();
            } else {
                view_control.ToggleEditingZ();
                utility::LogDebug(
                        "[Visualizer] Enter orthogonal Z editing mode.");
            }
            break;
        case GLFW_KEY_ESCAPE:
            if (select_editing_) {
                ExitSelectEdit();
            } else if (view_control.IsLocked()) {
                view_control.ToggleLocking();
                InvalidateSelectionPolygon();
                UpdateBackground();
            }
            break;
        case GLFW_KEY_E:
        case GLFW_KEY_K: {
            if (!select_editing_) {
                view_control.ToggleLocking();
                InvalidateSelectionPolygon();
                utility::LogDebug("[Visualizer] Camera %s.",
                                  view_control.IsLocked() ? "Lock" : "Unlock");
                UpdateBackground();
            }
            break;
        }
        case GLFW_KEY_R:
            if (mods & GLFW_MOD_CONTROL) {
                (geometry::PointCloud &)*editing_geometry_ptr_ =
                        (const geometry::PointCloud &)*original_geometry_ptr_;
                editing_geometry_renderer_ptr_->UpdateGeometry();
            } else {
                Visualizer::ResetViewPoint(true);
//                Visualizer::KeyPressCallback(window, key, scancode, action,
//                                             mods);
            }
            break;
        case GLFW_KEY_D:
            if (mods & GLFW_MOD_CONTROL) {
                if (voxel_size_ > 0.0 && editing_geometry_ptr_ &&
                    editing_geometry_ptr_->GetGeometryType() ==
                            geometry::Geometry::GeometryType::PointCloud) {
                    auto &pcd = (geometry::PointCloud &)*editing_geometry_ptr_;
                    utility::LogInfo("Voxel downsample with voxel size {:.4f}, point size {}.",
                                     voxel_size_, pcd.points_.size());
                    pcd = *pcd.VoxelDownSample(voxel_size_);
                    utility::LogInfo("After voxel downsample, point size {}.", pcd.points_.size());
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
            if (select_editing_) {
                CropSelected(false);
            } else if (view_control.IsLocked() && selection_polygon_ptr_ &&
                !selection_polygon_ptr_->IsEmpty()) {
                Crop(false);
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
            }
            break;
        case GLFW_KEY_N:
            if (select_editing_) {
                FindNeighborWithSimilarNormals();
            } else {
                Visualizer::KeyPressCallback(window, key, scancode, action, mods);
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
                PointSelectionHint();
            }
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT &&
                   action == GLFW_RELEASE && (mods & GLFW_MOD_SHIFT)) {
            if (!pointcloud_picker_ptr_->picked_indices_.empty()) {
                utility::LogInfo(
                        "Remove picked point #{} from pick queue.",
                        pointcloud_picker_ptr_->picked_indices_.back());
                pointcloud_picker_ptr_->picked_indices_.pop_back();
                is_redraw_required_ = true;
                PointSelectionHint();
            }
        }
        Visualizer::MouseButtonCallback(window, button, action, mods);
    }
}

void VisualizerWithEditing::PointSelectionHint() {
    if (!select_editing_) {
        return;
    }
    std::string hint;
    auto sz = GetPickedPoints().size();
    if (sz >= 3) {
        hint += "Press <F> to find a plane ";
    }
    if (sz > 0) {
        hint += "Press <N> to find neighbors with similar normals ";
    }
    if (hint.length() > 0) {
        utility::LogInfo(hint.c_str());
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
void VisualizerWithEditing::Backup() {
    if (editing_geometry_ptr_->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
        auto pcd = std::make_shared<geometry::PointCloud>();
        *pcd = (geometry::PointCloud &)*editing_geometry_ptr_;
    }
}
void VisualizerWithEditing::ExitSelectEdit() {
    if (!select_editing_) {
        return;
    }
    glfwMakeContextCurrent(window_);
    utility::LogInfo("Exit select-editing");
    select_editing_ = false;
    auto &pcd = (geometry::PointCloud&)*editing_geometry_ptr_;
    for (auto &geo : selected_original_geometries_) {
        pcd += (geometry::PointCloud&)*geo;
    }
    for (auto &geo : selected_geometries_) {
        RemoveGeometry(geo, false);
    }
    selected_geometries_.clear();
    selected_original_geometries_.clear();
    editing_geometry_renderer_ptr_->UpdateGeometry();

    InvalidateSelectionPolygon();
    InvalidatePicking();
    UpdateBackground();
}
void VisualizerWithEditing::FindNeighborWithSimilarNormals() {
    auto &picked = GetPickedPoints();
    if (picked.size() < 1) {
        utility::LogInfo("Magic wand requires at least 1 points selected");
        return;
    }
    if (editing_geometry_ptr_->GetGeometryType() != geometry::Geometry::GeometryType::PointCloud) {
        utility::LogInfo("Magic wand works only on point cloud");
        return;
    }
    auto pcd = std::dynamic_pointer_cast<geometry::PointCloud>(editing_geometry_ptr_);
    if (pcd->points_.size() > 100 * 10000) {
        utility::LogInfo("Current point cloud contains {} points, it will be too slow to fit a plane, please downsample it.", pcd->points_.size());
        return;
    }
    if (!pcd->HasNormals()) {
        utility::LogInfo("Estimate normals, this will take a long time");
        pcd->EstimateNormals(geometry::KDTreeSearchParamHybrid(10, 30));
        pcd->OrientNormalsConsistentTangentPlane(30);
    }
    geometry::KDTreeFlann tree(*pcd);
    double maxRadius = 50; // mm

    std::atomic_int cnt;  // used to record fail count

    // indices we want for each picked point
    std::vector<std::vector<int>> indices;
    indices.resize(picked.size());

    utility::LogInfo("Search radius");
    // idx_pts_nei = [p for i in ids_pts_picked
    //                     for p in tree_pcd.search_radius_vector_3d(pcd_std.points[i], radius=max_radius)[1]
    //                ]
#pragma omp parallel for schedule(static) shared(indices)
    for (auto i = 0u; i < picked.size(); i++) {
        std::vector<double> distance;
        auto k = tree.SearchRadius(pcd->points_[picked[i]], maxRadius, indices[i], distance);
        if (k < 0) {
            cnt++; // fail
        }
    }

    if (cnt.load() > 0) {
        utility::LogWarning("Search radius fails");
        is_redraw_required_ = true;
        InvalidatePicking();
        return;
    }

    // merge distances to distance[0]
    for (auto i = 1u; i < indices.size(); i++) {
        indices[0].insert(indices[0].end(), indices[i].begin(), indices[i].end());
    }
#if 1
    std::vector<size_t> selected_indices(indices[0].size());
    for (auto i = 0u; i < indices[0].size(); i++) {
        selected_indices[i] = indices[0][i];
    }
#else

    // nrm_nei = np.asarray(pcd_std.normals)[idx_pts_nei,:]
    auto &idx_pts_nei = indices[0];
    std::vector<Eigen::Vector3d> nrm_nei;
    nrm_nei.resize(idx_pts_nei.size());
    for (auto i = 0u; i < idx_pts_nei.size(); i++) {
        nrm_nei[i] = pcd->normals_[idx_pts_nei[i]];
    }

    // nrm = np.asarray(pcd_sel.normals)[0]
    auto sel = pcd->SelectByIndex(picked);
    auto nrm = sel->normals_[0];

    //cs = np.inner(nrm,nrm_nei)
    // cs = np.clip(cs,-1,1)
    // ang = np.arccos(cs) / np.pi * 180
    std::vector<double> cs(nrm_nei.size());
    std::vector<double> ang(nrm_nei.size());
    const double pi = 3.141592653589793;
    const double g = 180/pi;

#pragma omp parallel for schedule(static)
    for (auto i = 0u; i < nrm_nei.size(); i++) {
        auto &v = nrm_nei[i];
        auto r = nrm[0] * v[0] + nrm[1] * v[1] + nrm[2] * v[2];
        if (r < -1.0) {
            r = -1.0;
        } else if (r > 1.0) {
            r = 1.0;
        }
        cs[i] = r;
        ang[i] = std::acos(r) * g;
    }
    std::vector<size_t> res;
    auto th_ang = 20; // degrees
    // similar
    for (auto i = 0u; i < ang.size(); i++) {
        if (fabs(ang[i]) < th_ang) {
            res.push_back(idx_pts_nei[i]);
        }
    }
#endif
    // crop from current editing geometry, the cropped part will be saved to selection histroy
    auto geo = std::dynamic_pointer_cast<geometry::PointCloud>(Crop(selected_indices, true));
    if (geo && geo->HasPoints()) {
        // copy and save cropped geo with original color
        auto orig = std::make_shared<geometry::PointCloud>();
        *orig = *geo;
        selected_original_geometries_.push_back(orig);

        // paint selected cloud with green for rendering
        geo->PaintUniformColor({0, 1, 0});
        Visualizer::AddGeometry(geo, false);
        selected_geometries_.push_back(geo);
    }
    is_redraw_required_ = true;
    InvalidatePicking();
}
//def dist2plane(x,plane_model):
//    # dist2plane = lambda x: plane_model[0]*x[0] + plane_model[1]*x[1] + plane_model[2]*x[2] + plane_model[3]
//    return (plane_model[0]*x[0] + plane_model[1]*x[1] + plane_model[2]*x[2] + plane_model[3])

// def nearest2plane(pts, plane_model, threshold=1, viz=False):
//    idx_inside = [abs(dist2plane(x, plane_model)) < threshold for x in pts]
//    idx_inside = np.flatnonzero(idx_inside)
//
// idx_inside = utils3d.nearest2plane(np.asarray(pcd_std.points),
//                                    plane_model, threshold=distance_threshold, viz=True)
void VisualizerWithEditing::FitPlane() {
    auto &picked = GetPickedPoints();
    if (picked.size() < 3) {
        utility::LogInfo("Plane segment requires at least 3 points selected");
        return;
    }
    if (editing_geometry_ptr_->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
        auto &pcd = (geometry::PointCloud &)*editing_geometry_ptr_;
        if (pcd.points_.size() > 30 * 10000) {
            utility::LogInfo("Current point cloud contains {} points, it will be too slow to fit a plane, please downsample it.", pcd.points_.size());
            return;
        }
        auto sel = pcd.SelectByIndex(picked);
        double distance = 3.0;
        int ransac_n = 3;
        int iteration = 500;
        utility::LogInfo("segment plane");
        auto tp = sel->SegmentPlane(distance, ransac_n, iteration);
        auto plane = std::get<0>(tp);

        utility::LogInfo("find plane");
        std::vector<size_t> inside; // indices on plane
        for (auto i = 0u; i < pcd.points_.size(); i++) {
            auto &p = pcd.points_[i];
            auto dist = p[0] * plane[0] + p[1] * plane[1] + p[2] * plane[2] + plane[3];
            if (fabs(dist) < distance) {
                inside.push_back(i);
            }
        }
        if (inside.empty()) {
            utility::LogInfo("find nothing");
            InvalidatePicking();
            return;
        }

        utility::LogInfo("dbscan");
        // filtered is all points in editing geometry that is close to the plane
        auto filtered = pcd.SelectByIndex(inside);
        // dbscan to label clusters, labels contains a vector with same size of filter.points_
        // and value >= 0 indicates it belongs to a point cluster whose sequence is value
        auto labels = filtered->ClusterDBSCAN(10, 30, true);
        auto max = *std::max_element(labels.begin(), labels.end());
        utility::LogInfo("dbscan max {}", max);
        // cluster_indices contains a group of cluster, which defines index in editing geometry(NOT filter)
        std::vector<std::vector<size_t>> cluster_indices;
        cluster_indices.resize(max+1);
        for (auto idx = 0u; idx < labels.size(); idx++) {
            auto label = labels[idx];
            if (label >= 0) {
                // turn index of filter to index of editing geometry
                cluster_indices[label].push_back(inside[idx]);
            }
        }

        utility::LogInfo("merge");
        std::vector<size_t> merged_indices;
        for (auto &v : cluster_indices) {
            if (v.empty()) {
                continue;
            }
            // check if picked indexes inside this one
            bool found = false;
            for (auto k : v) {
                auto &p = pcd.points_[k];
                for (auto &pick : sel->points_) {
                    if (pick == p) {
                        found = true;
                        break;
                    }
                }
            }
            if (found) {
                merged_indices.reserve(merged_indices.size() + v.size());
                merged_indices.insert(merged_indices.end(), v.begin(), v.end());
            }
        }

        utility::LogInfo("crop");
        // crop from current editing geometry, the cropped part will be saved to selection histroy
        auto geo = std::dynamic_pointer_cast<geometry::PointCloud>(Crop(merged_indices, true));
        if (geo && geo->HasPoints()) {
            // copy and save cropped geo with original color
            auto orig = std::make_shared<geometry::PointCloud>();
            *orig = *geo;
            selected_original_geometries_.push_back(orig);

            // paint selected cloud with green for rendering
            geo->PaintUniformColor({0, 1, 0});
            Visualizer::AddGeometry(geo, false);
            selected_geometries_.push_back(geo);
        }
    }
    is_redraw_required_ = true;
    InvalidatePicking();
}

void VisualizerWithEditing::CropSelected(bool del /* del = true indicates indexes should be deleted */ ) {
    if (selected_geometries_.empty()) {
        return;
    }

    glfwMakeContextCurrent(window_);
    if (editing_geometry_ptr_ &&
        editing_geometry_ptr_->GetGeometryType() == geometry::Geometry::GeometryType::PointCloud) {
        auto &pcd = (geometry::PointCloud &)*editing_geometry_ptr_;

        // merge selection
        auto selected = std::make_shared<geometry::PointCloud>(); // merged selected geometry
        for (auto &geo : selected_original_geometries_) {
            auto &cloud = (geometry::PointCloud&)*geo;
            *selected += cloud;
        }
        for (auto &geo : selected_geometries_) {
            RemoveGeometry(geo, false);
        }
        selected_geometries_.clear();
        selected_original_geometries_.clear();

        // collected deleted part and push to history
        auto deleted = std::make_shared<geometry::PointCloud>();
        if (!del) {
            // delete current editing, make selection being editing
            *deleted = pcd;
            pcd = *selected;
            editing_geometry_renderer_ptr_->UpdateGeometry();
        } else {
            // delete selected, keep editing unchanged
            deleted = selected;
        }
        if (deleted->HasPoints()) {
            discarded_geometries_.push_back(deleted);
        }
    }
    InvalidateSelectionPolygon();
    InvalidatePicking();
}

void VisualizerWithEditing::WindowCloseCallback(GLFWwindow *window) {
    OnExit();
}
void VisualizerWithEditing::OnExit() {
    utility::LogInfo("Exit, merge selections {}", select_editing_);
    if (select_editing_) {
        select_editing_ = false;
        auto &pcd = (geometry::PointCloud&)*editing_geometry_ptr_;
        for (auto &geo : selected_original_geometries_) {
            pcd += (geometry::PointCloud&)*geo;
        }
        selected_geometries_.clear();
        selected_original_geometries_.clear();
    }
}
}  // namespace visualization
}  // namespace open3d
