// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open3d/geometry/Geometry.h"

namespace open3d {
namespace visualization {

class Visualizer;

/// \brief Function to draw a list of geometry objects
///
/// The convenient function of drawing something
/// This function is a wrapper that calls the core functions of Visualizer.
/// This function MUST be called from the main thread. It blocks the main thread
/// until the window is closed.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
/// \param point_show_normal visualize point normals if set to true.
/// \param mesh_show_wireframe visualize mesh wireframe if set to true.
/// \param mesh_show_back_face visualize also the back face of the mesh
/// triangles.
/// \param lookat The lookat vector of the camera.
/// \param up The up vector of the camera.
/// \param front The front vector of the camera.
/// \param zoom The zoom of the camera.
bool DrawGeometries(const std::vector<std::shared_ptr<const geometry::Geometry>>
                            &geometry_ptrs,
                    const std::string &window_name = "Open3D",
                    int width = 640,
                    int height = 480,
                    int left = 50,
                    int top = 50,
                    bool point_show_normal = false,
                    bool mesh_show_wireframe = false,
                    bool mesh_show_back_face = false,
                    Eigen::Vector3d *lookat = nullptr,
                    Eigen::Vector3d *up = nullptr,
                    Eigen::Vector3d *front = nullptr,
                    double *zoom = nullptr);

/// \brief Function to draw a list of geometry objects with a GUI that
/// supports animation.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
/// \param json_filename Camera trajectory json file path
/// for custom animation.
bool DrawGeometriesWithCustomAnimation(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::string &window_name = "Open3D",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50,
        const std::string &json_filename = "");

/// \brief Function to draw a list of geometry objects with a
/// customized animation callback function.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param callback_func Call back function to be triggered at a key press
/// event.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
bool DrawGeometriesWithAnimationCallback(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        std::function<bool(Visualizer *)> callback_func,
        const std::string &window_name = "Open3D",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

/// \brief Function to draw a list of geometry.
///
/// Geometry objects with a customized key-callback mapping
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param key_to_callback Map of key to call back functions.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
bool DrawGeometriesWithKeyCallbacks(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::map<int, std::function<bool(Visualizer *)>> &key_to_callback,
        const std::string &window_name = "Open3D",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

/// \brief Function to draw a list of geometry.
///
/// Geometry providing user interaction.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
bool DrawGeometriesWithEditing(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::string &window_name = "Open3D",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

bool DrawGeometriesWithVertexSelection(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::string &window_name = "Open3D",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

}  // namespace visualization
}  // namespace open3d
