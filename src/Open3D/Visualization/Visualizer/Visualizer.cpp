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

#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d {

namespace {

class GLFWEnvironmentSingleton {
private:
    GLFWEnvironmentSingleton() { utility::LogDebug("GLFW init.\n"); }
    GLFWEnvironmentSingleton(const GLFWEnvironmentSingleton &) = delete;
    GLFWEnvironmentSingleton &operator=(const GLFWEnvironmentSingleton &) =
            delete;

public:
    ~GLFWEnvironmentSingleton() {
        glfwTerminate();
        utility::LogDebug("GLFW destruct.\n");
    }

public:
    static GLFWEnvironmentSingleton &GetInstance() {
        static GLFWEnvironmentSingleton singleton;
        return singleton;
    }

    static int InitGLFW() {
        GLFWEnvironmentSingleton::GetInstance();
        return glfwInit();
    }

    static void GLFWErrorCallback(int error, const char *description) {
        utility::LogError("GLFW Error: {}\n", description);
    }
};

}  // unnamed namespace

namespace visualization {

Visualizer::Visualizer() {}

Visualizer::~Visualizer() {
    glfwTerminate();  // to be safe
}

bool Visualizer::CreateVisualizerWindow(
        const std::string &window_name /* = "Open3D"*/,
        const int width /* = 640*/,
        const int height /* = 480*/,
        const int left /* = 50*/,
        const int top /* = 50*/,
        const bool visible /* = true*/) {
    window_name_ = window_name;
    if (window_) {  // window already created
        UpdateWindowTitle();
        glfwSetWindowPos(window_, left, top);
        glfwSetWindowSize(window_, width, height);
#ifdef __APPLE__
        glfwSetWindowSize(window_,
                          std::round(width * pixel_to_screen_coordinate_),
                          std::round(height * pixel_to_screen_coordinate_));
        glfwSetWindowPos(window_,
                         std::round(left * pixel_to_screen_coordinate_),
                         std::round(top * pixel_to_screen_coordinate_));
#endif  //__APPLE__
        return true;
    }

    glfwSetErrorCallback(GLFWEnvironmentSingleton::GLFWErrorCallback);
    if (!GLFWEnvironmentSingleton::InitGLFW()) {
        utility::LogError("Failed to initialize GLFW\n");
        return false;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, visible ? 1 : 0);

    window_ = glfwCreateWindow(width, height, window_name_.c_str(), NULL, NULL);
    if (!window_) {
        utility::LogError("Failed to create window\n");
        return false;
    }
    glfwSetWindowPos(window_, left, top);
    glfwSetWindowUserPointer(window_, this);

#ifdef __APPLE__
    // Some hacks to get pixel_to_screen_coordinate_
    glfwSetWindowSize(window_, 100, 100);
    glfwSetWindowPos(window_, 100, 100);
    int pixel_width_in_osx, pixel_height_in_osx;
    glfwGetFramebufferSize(window_, &pixel_width_in_osx, &pixel_height_in_osx);
    if (pixel_width_in_osx > 0) {
        pixel_to_screen_coordinate_ = 100.0 / (double)pixel_width_in_osx;
    } else {
        pixel_to_screen_coordinate_ = 1.0;
    }
    glfwSetWindowSize(window_, std::round(width * pixel_to_screen_coordinate_),
                      std::round(height * pixel_to_screen_coordinate_));
    glfwSetWindowPos(window_, std::round(left * pixel_to_screen_coordinate_),
                     std::round(top * pixel_to_screen_coordinate_));
#endif  //__APPLE__

    auto window_refresh_callback = [](GLFWwindow *window) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->WindowRefreshCallback(window);
    };
    glfwSetWindowRefreshCallback(window_, window_refresh_callback);

    auto window_resize_callback = [](GLFWwindow *window, int w, int h) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->WindowResizeCallback(window, w, h);
    };
    glfwSetFramebufferSizeCallback(window_, window_resize_callback);

    auto mouse_move_callback = [](GLFWwindow *window, double x, double y) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->MouseMoveCallback(window, x, y);
    };
    glfwSetCursorPosCallback(window_, mouse_move_callback);

    auto mouse_scroll_callback = [](GLFWwindow *window, double x, double y) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->MouseScrollCallback(window, x, y);
    };
    glfwSetScrollCallback(window_, mouse_scroll_callback);

    auto mouse_button_callback = [](GLFWwindow *window, int button, int action,
                                    int mods) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->MouseButtonCallback(window, button, action, mods);
    };
    glfwSetMouseButtonCallback(window_, mouse_button_callback);

    auto key_press_callback = [](GLFWwindow *window, int key, int scancode,
                                 int action, int mods) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->KeyPressCallback(window, key, scancode, action, mods);
    };
    glfwSetKeyCallback(window_, key_press_callback);

    auto window_close_callback = [](GLFWwindow *window) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->WindowCloseCallback(window);
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

    if (InitRenderOption() == false) {
        return false;
    }

    int window_width, window_height;
    glfwGetFramebufferSize(window_, &window_width, &window_height);
    WindowResizeCallback(window_, window_width, window_height);

    UpdateWindowTitle();

    is_initialized_ = true;
    return true;
}

void Visualizer::DestroyVisualizerWindow() {
    is_initialized_ = false;
    glDeleteVertexArrays(1, &vao_id_);
    glfwDestroyWindow(window_);
}

void Visualizer::RegisterAnimationCallback(
        std::function<bool(Visualizer *)> callback_func) {
    animation_callback_func_ = callback_func;
}

bool Visualizer::InitViewControl() {
    view_control_ptr_ = std::unique_ptr<ViewControl>(new ViewControl);
    ResetViewPoint();
    return true;
}

bool Visualizer::InitRenderOption() {
    render_option_ptr_ = std::unique_ptr<RenderOption>(new RenderOption);
    return true;
}

void Visualizer::UpdateWindowTitle() {
    if (window_ != NULL) {
        glfwSetWindowTitle(window_, window_name_.c_str());
    }
}

void Visualizer::BuildUtilities() {
    glfwMakeContextCurrent(window_);

    // 0. Build coordinate frame
    const auto boundingbox = GetViewControl().GetBoundingBox();
    coordinate_frame_mesh_ptr_ = geometry::TriangleMesh::CreateCoordinateFrame(
            boundingbox.GetMaxExtent() * 0.2, boundingbox.min_bound_);
    coordinate_frame_mesh_renderer_ptr_ =
            std::make_shared<glsl::CoordinateFrameRenderer>();
    if (coordinate_frame_mesh_renderer_ptr_->AddGeometry(
                coordinate_frame_mesh_ptr_) == false) {
        return;
    }
    utility_ptrs_.push_back(coordinate_frame_mesh_ptr_);
    utility_renderer_ptrs_.push_back(coordinate_frame_mesh_renderer_ptr_);
}

void Visualizer::Run() {
    BuildUtilities();
    UpdateWindowTitle();
    while (bool(animation_callback_func_) ? PollEvents() : WaitEvents()) {
        if (bool(animation_callback_func_in_loop_)) {
            if (animation_callback_func_in_loop_(this)) {
                UpdateGeometry();
            }
            // Set render flag as dirty anyways, because when we use callback
            // functions, we assume something has been changed in the callback
            // and the redraw event should be triggered.
            UpdateRender();
        }
    }
}

void Visualizer::Close() {
    glfwSetWindowShouldClose(window_, GL_TRUE);
    utility::LogDebug("[Visualizer] Window closing.\n");
}

bool Visualizer::WaitEvents() {
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

bool Visualizer::PollEvents() {
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

bool Visualizer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (is_initialized_ == false) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    std::shared_ptr<glsl::GeometryRenderer> renderer_ptr;
    if (geometry_ptr->GetGeometryType() ==
        geometry::Geometry::GeometryType::Unspecified) {
        return false;
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::PointCloud) {
        renderer_ptr = std::make_shared<glsl::PointCloudRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::VoxelGrid) {
        renderer_ptr = std::make_shared<glsl::VoxelGridRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Octree) {
        renderer_ptr = std::make_shared<glsl::OctreeRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::LineSet) {
        renderer_ptr = std::make_shared<glsl::LineSetRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
                       geometry::Geometry::GeometryType::TriangleMesh ||
               geometry_ptr->GetGeometryType() ==
                       geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        renderer_ptr = std::make_shared<glsl::TriangleMeshRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Image) {
        renderer_ptr = std::make_shared<glsl::ImageRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::RGBDImage) {
        renderer_ptr = std::make_shared<glsl::RGBDImageRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::TetraMesh) {
        renderer_ptr = std::make_shared<glsl::TetraMeshRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::OrientedBoundingBox) {
        renderer_ptr = std::make_shared<glsl::OrientedBoundingBoxRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        renderer_ptr = std::make_shared<glsl::AxisAlignedBoundingBoxRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else {
        return false;
    }
    geometry_renderer_ptrs_.insert(renderer_ptr);
    geometry_ptrs_.insert(geometry_ptr);
    view_control_ptr_->FitInGeometry(*geometry_ptr);
    ResetViewPoint();
    utility::LogDebug(
            "Add geometry and update bounding box to {}\n",
            view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
    return UpdateGeometry();
}

bool Visualizer::RemoveGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (is_initialized_ == false) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    std::shared_ptr<glsl::GeometryRenderer> geometry_renderer_delete = NULL;
    for (auto &geometry_renderer_ptr : geometry_renderer_ptrs_) {
        if (geometry_renderer_ptr->GetGeometry() == geometry_ptr)
            geometry_renderer_delete = geometry_renderer_ptr;
    }
    if (geometry_renderer_delete == NULL) return false;
    geometry_renderer_ptrs_.erase(geometry_renderer_delete);
    geometry_ptrs_.erase(geometry_ptr);
    ResetViewPoint(true);
    utility::LogDebug(
            "Remove geometry and update bounding box to {}\n",
            view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
    return UpdateGeometry();
}

bool Visualizer::UpdateGeometry() {
    glfwMakeContextCurrent(window_);
    bool success = true;
    for (const auto &renderer_ptr : geometry_renderer_ptrs_) {
        success = (success && renderer_ptr->UpdateGeometry());
    }
    UpdateRender();
    return success;
}

void Visualizer::UpdateRender() { is_redraw_required_ = true; }

bool Visualizer::HasGeometry() const { return !geometry_ptrs_.empty(); }

void Visualizer::PrintVisualizerHelp() {
    // clang-format off
    utility::LogInfo("  -- Mouse view control --\n");
    utility::LogInfo("    Left button + drag         : Rotate.\n");
    utility::LogInfo("    Ctrl + left button + drag  : Translate.\n");
    utility::LogInfo("    Wheel button + drag        : Translate.\n");
    utility::LogInfo("    Shift + left button + drag : Roll.\n");
    utility::LogInfo("    Wheel                      : Zoom in/out.\n");
    utility::LogInfo("\n");
    utility::LogInfo("  -- Keyboard view control --\n");
    utility::LogInfo("    [/]          : Increase/decrease field of view.\n");
    utility::LogInfo("    R            : Reset view point.\n");
    utility::LogInfo("    Ctrl/Cmd + C : Copy current view status into the clipboard.\n");
    utility::LogInfo("    Ctrl/Cmd + V : Paste view status from clipboard.\n");
    utility::LogInfo("\n");
    utility::LogInfo("  -- General control --\n");
    utility::LogInfo("    Q, Esc       : Exit window.\n");
    utility::LogInfo("    H            : Print help message.\n");
    utility::LogInfo("    P, PrtScn    : Take a screen capture.\n");
    utility::LogInfo("    D            : Take a depth capture.\n");
    utility::LogInfo("    O            : Take a capture of current rendering settings.\n");
    utility::LogInfo("\n");
    utility::LogInfo("  -- Render mode control --\n");
    utility::LogInfo("    L            : Turn on/off lighting.\n");
    utility::LogInfo("    +/-          : Increase/decrease point size.\n");
    utility::LogInfo("    Ctrl + +/-   : Increase/decrease width of geometry::LineSet.\n");
    utility::LogInfo("    N            : Turn on/off point cloud normal rendering.\n");
    utility::LogInfo("    S            : Toggle between mesh flat shading and smooth shading.\n");
    utility::LogInfo("    W            : Turn on/off mesh wireframe.\n");
    utility::LogInfo("    B            : Turn on/off back face rendering.\n");
    utility::LogInfo("    I            : Turn on/off image zoom in interpolation.\n");
    utility::LogInfo("    T            : Toggle among image render:\n");
    utility::LogInfo("                   no stretch / keep ratio / freely stretch.\n");
    utility::LogInfo("\n");
    utility::LogInfo("  -- Color control --\n");
    utility::LogInfo("    0..4,9       : Set point cloud color option.\n");
    utility::LogInfo("                   0 - Default behavior, render point color.\n");
    utility::LogInfo("                   1 - Render point color.\n");
    utility::LogInfo("                   2 - x coordinate as color.\n");
    utility::LogInfo("                   3 - y coordinate as color.\n");
    utility::LogInfo("                   4 - z coordinate as color.\n");
    utility::LogInfo("                   9 - normal as color.\n");
    utility::LogInfo("    Ctrl + 0..4,9: Set mesh color option.\n");
    utility::LogInfo("                   0 - Default behavior, render uniform gray color.\n");
    utility::LogInfo("                   1 - Render point color.\n");
    utility::LogInfo("                   2 - x coordinate as color.\n");
    utility::LogInfo("                   3 - y coordinate as color.\n");
    utility::LogInfo("                   4 - z coordinate as color.\n");
    utility::LogInfo("                   9 - normal as color.\n");
    utility::LogInfo("    Shift + 0..4 : Color map options.\n");
    utility::LogInfo("                   0 - Gray scale color.\n");
    utility::LogInfo("                   1 - JET color map.\n");
    utility::LogInfo("                   2 - SUMMER color map.\n");
    utility::LogInfo("                   3 - WINTER color map.\n");
    utility::LogInfo("                   4 - HOT color map.\n");
    utility::LogInfo("\n");
    // clang-format on
}
}  // namespace visualization
}  // namespace open3d
