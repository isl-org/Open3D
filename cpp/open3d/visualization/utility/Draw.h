// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <optional>
#include <vector>

#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/visualizer/O3DVisualizer.h"

namespace open3d {
namespace visualization {

struct DrawObject {
    std::string name;
    std::shared_ptr<geometry::Geometry3D> geometry;
    std::shared_ptr<t::geometry::Geometry> tgeometry;
    std::shared_ptr<rendering::TriangleMeshModel> model;
    rendering::MaterialRecord material;
    std::string group;
    double time = 0.0;
    bool has_material = false;
    bool is_visible;

    DrawObject(const std::string &n,
               std::shared_ptr<geometry::Geometry3D> g,
               bool vis = true);
    DrawObject(const std::string &n,
               std::shared_ptr<t::geometry::Geometry> tg,
               bool vis = true);
    DrawObject(const std::string &n,
               std::shared_ptr<rendering::TriangleMeshModel> m,
               bool vis = true);
};

struct DrawAction {
    std::string name;
    std::function<void(visualizer::O3DVisualizer &)> callback;
};

/// Configuration for advanced Draw parameters.
///
/// DrawConfig provides optional parameters for customizing the visualization,
/// matching the draw.py API. The initial view may be specified either as a
/// combination of (lookat, eye, up, and field_of_view) or (intrinsic_matrix,
/// extrinsic_matrix) pair. A simple pinhole camera model is used.
///
/// Example:
/// \code{.cpp}
/// DrawConfig config;
/// config.eye = Eigen::Vector3d(0, 0, 5);
/// config.lookat = Eigen::Vector3d(0, 0, 0);
/// config.up = Eigen::Vector3d(0, 1, 0);
/// config.field_of_view = 60.0f;
/// config.bg_color = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);  // white
/// \endcode
struct DrawConfig {
    /// \section Camera Setup
    /// Camera principal axis direction (lookat point). Use with eye and up.
    std::optional<Eigen::Vector3f> lookat;
    /// Camera location. Use with lookat and up.
    std::optional<Eigen::Vector3f> eye;
    /// Camera up direction. Use with lookat and eye.
    std::optional<Eigen::Vector3f> up;
    /// Camera horizontal field of view in degrees. Default: 60.0.
    float field_of_view = 60.0f;
    /// Camera intrinsic matrix (3x3). Use with extrinsic_matrix instead of
    /// lookat/eye/up.
    std::optional<Eigen::Matrix3d> intrinsic_matrix;
    /// Camera extrinsic matrix (4x4) for world-to-camera transformation. Use
    /// with intrinsic_matrix instead of lookat/eye/up.
    std::optional<Eigen::Matrix4d> extrinsic_matrix;

    /// \section Background and Rendering
    /// Background color as RGBA float with range [0,1]. Default: white.
    std::optional<Eigen::Vector4f> bg_color;
    /// Background image. If specified, overrides bg_color.
    std::shared_ptr<geometry::Image> bg_image;

    /// \section IBL and Environment
    /// Path to environment map for image-based lighting (IBL).
    std::optional<std::string> ibl;
    /// IBL intensity multiplier. Default: 1.0.
    std::optional<float> ibl_intensity;
    /// Show skybox as scene background. Default: false.
    std::optional<bool> show_skybox;

    /// \section UI and Rendering Modes
    /// Show settings user interface (can be toggled from Actions menu).
    /// Default: false.
    std::optional<bool> show_ui;
    /// Show world-space axes at the scene origin. Default: false.
    std::optional<bool> show_axes;
    /// Use raw mode for simpler rendering of basic geometry. Default: false.
    std::optional<bool> raw_mode;

    /// \section Camera Clip Planes
    /// Near clip distance (world units). If set, overrides the value chosen by
    /// ResetCameraToDefault(). Must be positive. Default: auto.
    std::optional<float> near_plane;
    /// Far clip distance (world units). Must be greater than near_plane.
    /// Default: auto.
    std::optional<float> far_plane;
    /// 3D point size (pixel count). Default: 3.
    std::optional<int> point_size;
    /// 3D line width (pixel count). Default: 2.
    std::optional<int> line_width;

    /// \section Animation Parameters
    /// Duration in seconds for each animation frame. Default: 1.0.
    double animation_time_step = 1.0;
    /// Total animation duration in seconds. If not specified, animation runs
    /// indefinitely.
    std::optional<double> animation_duration;

    /// \section RPC Interface
    /// RPC interface address string (e.g., "tcp://localhost:51454").
    /// Empty string disables RPC interface. Default: empty.
    std::string rpc_interface;

    /// \section Callbacks
    /// Optional callback for extra initialization of the GUI window.
    /// Signature: void(O3DVisualizer&)
    std::function<void(visualizer::O3DVisualizer &)> on_init;
    /// Optional callback invoked for each animation frame update.
    /// Signature: void(O3DVisualizer&, double time)
    std::function<void(visualizer::O3DVisualizer &, double)> on_animation_frame;
    /// Optional callback invoked for each animation time step.
    /// Signature: TickResult(O3DVisualizer&, double tick_duration, double time)
    /// Return TickResult::REDRAW to trigger scene redraw, or
    /// TickResult::NO_CHANGE if redraw is not needed.
    std::function<visualizer::O3DVisualizer::TickResult(
            visualizer::O3DVisualizer &, double, double)>
            on_animation_tick;

    /// \section Blocking Mode
    /// If true, do not block waiting for window close. Instead, return the
    /// window ID. Useful for embedding the visualizer. Default: false.
    bool non_blocking_and_return_uid = false;
};

/// Draw 3D geometry and models with advanced configuration.
///
/// This is a high-level interface to O3DVisualizer that supports drawing
/// Open3D geometry types and TriangleMeshModels with optional metadata.
///
/// \param [in] objects List of DrawObject items to visualize. Each object
///     can optionally specify name, material, group, time stamp, and
///     visibility.
/// \param [in] window_name Title for the visualization window.
///     Default: "Open3D".
/// \param [in] width Viewport width in pixels. Default: 1024.
/// \param [in] height Viewport height in pixels. Default: 768.
/// \param [in] actions List of (name, callback) pairs for custom actions.
///     These are displayed as buttons in the settings panel. Each callback
///     receives the O3DVisualizer as an argument and can modify the
///     visualization.
/// \param [in] config Advanced drawing configuration (see DrawConfig).
///
/// \return Window UID if config.non_blocking_and_return_uid is true;
///     otherwise blocks until window closes and returns empty string.
///
/// Example:
/// \code{.cpp}
/// auto cloud = std::make_shared<geometry::PointCloud>();
/// // ... populate cloud ...
/// DrawObject obj("Point Cloud", cloud);
/// DrawConfig config;
/// config.eye = Eigen::Vector3d(0, 0, 5);
/// config.lookat = Eigen::Vector3d(0, 0, 0);
/// config.bg_color = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);
/// open3d::visualization::Draw({obj}, "My Visualization", 1024, 768, {},
/// config); \endcode
std::string Draw(const std::vector<DrawObject> &objects,
                 const std::string &window_name = "Open3D",
                 int width = 1024,
                 int height = 768,
                 const std::vector<DrawAction> &actions = {},
                 const DrawConfig &config = DrawConfig());

/// Simple wrapper for drawing legacy CPU-based geometries.
/// Wraps geometries in DrawObjects and calls the advanced Draw function.
///
/// \param [in] geometries List of legacy Geometry3D objects to visualize.
/// \param [in] window_name Window title. Default: "Open3D".
/// \param [in] width Viewport width. Default: 1024.
/// \param [in] height Viewport height. Default: 768.
/// \param [in] actions Custom action callbacks.
void Draw(const std::vector<std::shared_ptr<geometry::Geometry3D>> &geometries,
          const std::string &window_name = "Open3D",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

/// Simple wrapper for drawing tensor-based geometries (CPU/CUDA/SYCL).
/// Wraps geometries in DrawObjects and calls the advanced Draw function.
///
/// \param [in] tgeometries List of tensor-based Geometry objects to visualize.
/// \param [in] window_name Window title. Default: "Open3D".
/// \param [in] width Viewport width. Default: 1024.
/// \param [in] height Viewport height. Default: 768.
/// \param [in] actions Custom action callbacks.
void Draw(
        const std::vector<std::shared_ptr<t::geometry::Geometry>> &tgeometries,
        const std::string &window_name = "Open3D",
        int width = 1024,
        int height = 768,
        const std::vector<DrawAction> &actions = {});

/// Simple wrapper for drawing pre-loaded triangle mesh models.
/// Wraps models in DrawObjects and calls the advanced Draw function.
///
/// \param [in] models List of TriangleMeshModel objects to visualize.
/// \param [in] window_name Window title. Default: "Open3D".
/// \param [in] width Viewport width. Default: 1024.
/// \param [in] height Viewport height. Default: 768.
/// \param [in] actions Custom action callbacks.
void Draw(const std::vector<std::shared_ptr<rendering::TriangleMeshModel>>
                  &models,
          const std::string &window_name = "Open3D",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

}  // namespace visualization
}  // namespace open3d
