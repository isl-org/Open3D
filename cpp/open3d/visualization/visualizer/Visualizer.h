// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Avoid warning caused by redefinition of APIENTRY macro
// defined also in glfw3.h
#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "open3d/geometry/Geometry.h"
#include "open3d/visualization/shader/GeometryRenderer.h"
#include "open3d/visualization/utility/ColorMap.h"
#include "open3d/visualization/visualizer/RenderOption.h"
#include "open3d/visualization/visualizer/ViewControl.h"

namespace open3d {

namespace geometry {
class TriangleMesh;
class Image;
}  // namespace geometry

namespace visualization {

class GLFWContext;

/// \class Visualizer
///
/// \brief The main Visualizer class.
class Visualizer {
public:
    struct MouseControl {
    public:
        bool is_mouse_left_button_down = false;
        bool is_mouse_middle_button_down = false;
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
    /// \brief Function to create a window and initialize GLFW.
    ///
    /// This function MUST be called from the main thread.
    ///
    /// \param window_name Window title name.
    /// \param width Width of the window.
    /// \param height Height of window.
    /// \param left Left margin of the window to the screen.
    /// \param top Top margin of the window to the screen.
    /// \param visible Whether the window is visible.
    bool CreateVisualizerWindow(const std::string &window_name = "Open3D",
                                const int width = 640,
                                const int height = 480,
                                const int left = 50,
                                const int top = 50,
                                const bool visible = true);

    /// \brief Function to destroy a window.
    ///
    /// This function MUST be called from the main thread.
    void DestroyVisualizerWindow();

    /// \brief Function to register a callback function for animation.
    ///
    /// The callback function returns if UpdateGeometry() needs to be run.
    ///
    /// \param callback_func The call back function.
    void RegisterAnimationCallback(
            std::function<bool(Visualizer *)> callback_func);

    /// \brief Function to activate the window.
    ///
    /// This function will block the current thread until the window is closed.
    void Run();

    /// Function to to notify the window to be closed
    void Close();

    /// \brief Function to process the event queue and return if the window is
    /// closed.
    ///
    /// Use this function if you want to manage the while loop yourself. This
    /// function will block the thread.
    bool WaitEvents();

    /// Function to process the event queue and return if the window is closed.
    ///
    /// Use this function if you want to manage the while loop yourself. This
    /// function will NOT block the thread. Thus it is suitable for computation
    /// heavy task behind the scene.
    bool PollEvents();

    /// \brief Function to add geometry to the scene and create corresponding
    /// shaders.
    ///
    /// 1. After calling this function, the Visualizer owns the geometry object.
    /// 2. This function MUST be called after CreateVisualizerWindow().
    /// 3. This function returns FALSE when the geometry is of an unsupported
    /// type.
    /// 4. If an added geometry is changed, the behavior of Visualizer is
    /// undefined. Programmers are responsible for calling Geometry() to
    /// notify the Visualizer that the geometry has been changed and the
    /// Visualizer should be updated accordingly.
    ///
    /// \param geometry_ptr The Geometry object.
    /// \param reset_bounding_box Reset viewpoint to view all geometries.
    virtual bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr,
            bool reset_bounding_box = true);

    /// \brief Function to remove geometry from the scene.
    ///
    /// 1. After calling this function, the Visualizer releases the pointer of
    /// the geometry object.
    /// 2. This function MUST be called after CreateVisualizerWindow().
    /// 3. This function returns FALSE if the geometry to be removed is not
    /// added by AddGeometry
    ///
    /// \param geometry_ptr The Geometry object.
    /// \param reset_bounding_box Reset viewpoint to view all geometries.
    virtual bool RemoveGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr,
            bool reset_bounding_box = true);

    /// Function to remove all geometries from the scene.
    /// After calling this function, the Visualizer releases the pointer of
    /// all geometry objects.
    virtual bool ClearGeometries();

    /// \brief Function to update geometry.
    ///
    /// This function must be called when geometry has been changed. Otherwise
    /// the behavior of Visualizer is undefined.
    /// If called without an argument, updates all geometries, otherwise only
    /// updates the geometry specified.
    virtual bool UpdateGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr = nullptr);
    virtual bool HasGeometry() const;

    /// Function to inform render needed to be updated.
    virtual void UpdateRender();

    /// Functions to change between fullscreen and windowed modes
    virtual void SetFullScreen(bool fullscreen);
    virtual void ToggleFullScreen();
    virtual bool IsFullScreen();

    virtual void PrintVisualizerHelp();
    virtual void UpdateWindowTitle();
    virtual void BuildUtilities();

    /// Function to retrieve the associated ViewControl
    ViewControl &GetViewControl() { return *view_control_ptr_; }
    /// Function to retrieve the associated RenderOption.
    RenderOption &GetRenderOption() { return *render_option_ptr_; }
    /// \brief Function to capture screen and store RGB in a float buffer.
    ///
    /// \param do_render Set to `true` to do render.
    std::shared_ptr<geometry::Image> CaptureScreenFloatBuffer(
            bool do_render = true);
    /// \brief Function to capture and save a screen image.
    ///
    /// \param filename Path to file.
    /// \param do_render Set to `true` to do render.
    void CaptureScreenImage(const std::string &filename = "",
                            bool do_render = true);
    /// Function to capture depth in a float buffer.
    ///
    /// \param do_render Set to `true` to do render.
    std::shared_ptr<geometry::Image> CaptureDepthFloatBuffer(
            bool do_render = true);
    /// Function to capture and save a depth image.
    ///
    /// \param filename Path to file.
    /// \param do_render Set to `true` to do render.
    /// \param depth_scale Scale depth value when capturing the depth image.
    void CaptureDepthImage(const std::string &filename = "",
                           bool do_render = true,
                           double depth_scale = 1000.0);
    /// \brief Function to capture and save local point cloud.
    ///
    /// \param filename Path to file.
    /// \param do_render Set to `true` to do render.
    /// \param convert_to_world_coordinate Set to `true` to convert to world
    /// coordinates.
    void CaptureDepthPointCloud(const std::string &filename = "",
                                bool do_render = true,
                                bool convert_to_world_coordinate = false);
    void CaptureRenderOption(const std::string &filename = "");

    /// Function to reset view point.
    void ResetViewPoint(bool reset_bounding_box = false);

    const std::string &GetWindowName() const { return window_name_; }

    /// Get the current view status as a json string of ViewTrajectory.
    std::string GetViewStatus();

    /// Set the current view status from a json string of ViewTrajectory.
    void SetViewStatus(const std::string &view_status_str);

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
    virtual void Render(bool render_screen = false);

    /// Copy the current view status to clipboard.
    void CopyViewStatusToClipboard();

    /// Apply the view point from clipboard.
    void CopyViewStatusFromClipboard();

    /// Callback functions
    virtual void WindowRefreshCallback(GLFWwindow *window);
    virtual void WindowResizeCallback(GLFWwindow *window, int w, int h);
    virtual void MouseMoveCallback(GLFWwindow *window, double x, double y);
    virtual void MouseScrollCallback(GLFWwindow *window, double x, double y);
    virtual void MouseButtonCallback(GLFWwindow *window,
                                     int button,
                                     int action,
                                     int mods);
    virtual void KeyPressCallback(
            GLFWwindow *window, int key, int scancode, int action, int mods);
    /// \brief Function to notify the window to be closed.
    virtual void WindowCloseCallback(GLFWwindow *window);

protected:
    // window
    GLFWwindow *window_ = NULL;
    std::string window_name_ = "Open3D";

    /// \brief Shared GLFW context.
    std::shared_ptr<GLFWContext> glfw_context_ = nullptr;

    Eigen::Vector2i saved_window_size_ = Eigen::Vector2i::Zero();
    Eigen::Vector2i saved_window_pos_ = Eigen::Vector2i::Zero();
    std::function<bool(Visualizer *)> animation_callback_func_ = nullptr;
    // Auxiliary internal backup of the callback function.
    // It copies animation_callback_func_ in each PollEvent() or WaitEvent()
    // so that even if user calls RegisterAnimationCallback() within the
    // callback function it is still safe.
    std::function<bool(Visualizer *)> animation_callback_func_in_loop_ =
            nullptr;

    // control
    MouseControl mouse_control_;
    bool is_redraw_required_ = true;
    bool is_initialized_ = false;
    GLuint vao_id_ = 0;

    // render targets for "capture_screen_float_buffer" and
    // "capture_screen_image" in offscreen render mode
    GLuint render_fbo_ = 0;
    GLuint render_rgb_tex_ = 0;
    GLuint render_depth_stencil_rbo_ = 0;

    // view control
    std::unique_ptr<ViewControl> view_control_ptr_;

    // rendering properties
    std::unique_ptr<RenderOption> render_option_ptr_;

    // geometry to be rendered
    std::unordered_set<std::shared_ptr<const geometry::Geometry>>
            geometry_ptrs_;

    // geometry renderers
    std::unordered_set<std::shared_ptr<glsl::GeometryRenderer>>
            geometry_renderer_ptrs_;

    // utilities owned by the Visualizer
    std::vector<std::shared_ptr<const geometry::Geometry>> utility_ptrs_;

    // utility renderers
    std::vector<std::shared_ptr<glsl::GeometryRenderer>> utility_renderer_ptrs_;
    // map's key is the renderer for which the RenderOption applies
    // (should be something in utility_renderer_ptrs_)
    std::unordered_map<std::shared_ptr<glsl::GeometryRenderer>, RenderOption>
            utility_renderer_opts_;

    // coordinate frame
    std::shared_ptr<geometry::TriangleMesh> coordinate_frame_mesh_ptr_;
    std::shared_ptr<glsl::CoordinateFrameRenderer>
            coordinate_frame_mesh_renderer_ptr_;

#ifdef __APPLE__
    // MacBook with Retina display does not have a 1:1 mapping from screen
    // coordinates to pixels. Thus we hack it back.
    // http://www.glfw.org/faq.html#why-is-my-output-in-the-lower-left-corner-of-the-window
    double pixel_to_screen_coordinate_ = 1.0;
#endif  //__APPLE__
};

}  // namespace visualization
}  // namespace open3d
