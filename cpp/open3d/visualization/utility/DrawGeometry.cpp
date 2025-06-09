// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/utility/DrawGeometry.h"

#include <Eigen/Core>

#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/visualizer/GuiVisualizer.h"
#include "open3d/visualization/visualizer/ViewControlWithCustomAnimation.h"
#include "open3d/visualization/visualizer/ViewControlWithEditing.h"
#include "open3d/visualization/visualizer/Visualizer.h"
#include "open3d/visualization/visualizer/VisualizerWithCustomAnimation.h"
#include "open3d/visualization/visualizer/VisualizerWithEditing.h"
#include "open3d/visualization/visualizer/VisualizerWithKeyCallback.h"
#include "open3d/visualization/visualizer/VisualizerWithVertexSelection.h"

namespace open3d {
namespace visualization {

bool DrawGeometries(const std::vector<std::shared_ptr<const geometry::Geometry>>
                            &geometry_ptrs,
                    const std::string &window_name /* = "Open3D"*/,
                    int width /* = 640*/,
                    int height /* = 480*/,
                    int left /* = 50*/,
                    int top /* = 50*/,
                    bool point_show_normal /* = false */,
                    bool mesh_show_wireframe /* = false */,
                    bool mesh_show_back_face /* = false */,
                    Eigen::Vector3d *lookat /* = nullptr */,
                    Eigen::Vector3d *up /* = nullptr */,
                    Eigen::Vector3d *front /* = nullptr */,
                    double *zoom /* = nullptr */) {
    Visualizer visualizer;
    if (!visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                           top)) {
        utility::LogWarning(
                "[DrawGeometries] Failed creating OpenGL "
                "window.");
        return false;
    }
    visualizer.GetRenderOption().point_show_normal_ = point_show_normal;
    visualizer.GetRenderOption().mesh_show_wireframe_ = mesh_show_wireframe;
    visualizer.GetRenderOption().mesh_show_back_face_ = mesh_show_back_face;
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (!visualizer.AddGeometry(geometry_ptr)) {
            utility::LogWarning("[DrawGeometries] Failed adding geometry.");
            utility::LogWarning(
                    "[DrawGeometries] Possibly due to bad geometry or wrong"
                    " geometry type.");
            return false;
        }
    }

    ViewControl &view_control = visualizer.GetViewControl();
    if (lookat != nullptr) {
        view_control.SetLookat(*lookat);
    }
    if (up != nullptr) {
        view_control.SetUp(*up);
    }
    if (front != nullptr) {
        view_control.SetFront(*front);
    }
    if (zoom != nullptr) {
        view_control.SetZoom(*zoom);
    }

    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}

bool DrawGeometriesWithCustomAnimation(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::string &window_name /* = "Open3D"*/,
        int width /* = 640*/,
        int height /* = 480*/,
        int left /* = 50*/,
        int top /* = 50*/,
        const std::string &json_filename /* = ""*/) {
    VisualizerWithCustomAnimation visualizer;
    if (!visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                           top)) {
        utility::LogWarning(
                "[DrawGeometriesWithCustomAnimation] Failed creating OpenGL "
                "window.");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (!visualizer.AddGeometry(geometry_ptr)) {
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Failed adding "
                    "geometry.");
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Possibly due to bad "
                    "geometry or wrong geometry type.");
            return false;
        }
    }
    auto &view_control =
            (ViewControlWithCustomAnimation &)visualizer.GetViewControl();
    if (!json_filename.empty()) {
        if (!view_control.LoadTrajectoryFromJsonFile(json_filename)) {
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Failed loading json "
                    "file.");
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Possibly due to bad "
                    "file or file does not contain trajectory.");
            return false;
        }
        visualizer.UpdateWindowTitle();
    }
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}

bool DrawGeometriesWithAnimationCallback(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        std::function<bool(Visualizer *)> callback_func,
        const std::string &window_name /* = "Open3D"*/,
        int width /* = 640*/,
        int height /* = 480*/,
        int left /* = 50*/,
        int top /* = 50*/) {
    Visualizer visualizer;
    if (!visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                           top)) {
        utility::LogWarning(
                "[DrawGeometriesWithAnimationCallback] Failed creating OpenGL "
                "window.");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (!visualizer.AddGeometry(geometry_ptr)) {
            utility::LogWarning(
                    "[DrawGeometriesWithAnimationCallback] Failed adding "
                    "geometry.");
            utility::LogWarning(
                    "[DrawGeometriesWithAnimationCallback] Possibly due to bad "
                    "geometry or wrong geometry type.");
            return false;
        }
    }
    visualizer.RegisterAnimationCallback(callback_func);
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}

bool DrawGeometriesWithKeyCallbacks(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::map<int, std::function<bool(Visualizer *)>> &key_to_callback,
        const std::string &window_name /* = "Open3D"*/,
        int width /* = 640*/,
        int height /* = 480*/,
        int left /* = 50*/,
        int top /* = 50*/) {
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                           top)) {
        utility::LogWarning(
                "[DrawGeometriesWithKeyCallbacks] Failed creating OpenGL "
                "window.");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (!visualizer.AddGeometry(geometry_ptr)) {
            utility::LogWarning(
                    "[DrawGeometriesWithKeyCallbacks] Failed adding "
                    "geometry.");
            utility::LogWarning(
                    "[DrawGeometriesWithKeyCallbacks] Possibly due to bad "
                    "geometry or wrong geometry type.");
            return false;
        }
    }
    for (auto key_func_pair : key_to_callback) {
        visualizer.RegisterKeyCallback(key_func_pair.first,
                                       key_func_pair.second);
    }
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}

bool DrawGeometriesWithEditing(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::string &window_name /* = "Open3D"*/,
        int width /* = 640*/,
        int height /* = 480*/,
        int left /* = 50*/,
        int top /* = 50*/) {
    VisualizerWithEditing visualizer;
    if (!visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                           top)) {
        utility::LogWarning(
                "[DrawGeometriesWithEditing] Failed creating OpenGL window.");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (!visualizer.AddGeometry(geometry_ptr)) {
            utility::LogWarning(
                    "[DrawGeometriesWithEditing] Failed adding geometry.");
            utility::LogWarning(
                    "[DrawGeometriesWithEditing] Possibly due to bad geometry "
                    "or wrong geometry type.");
            return false;
        }
    }
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}

bool DrawGeometriesWithVertexSelection(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometry_ptrs,
        const std::string &window_name /* = "Open3D"*/,
        int width /* = 640*/,
        int height /* = 480*/,
        int left /* = 50*/,
        int top /* = 50*/) {
    VisualizerWithVertexSelection visualizer;
    if (!visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                           top)) {
        utility::LogWarning(
                "[DrawGeometriesWithVertexSelection] Failed creating OpenGL "
                "window.");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (!visualizer.AddGeometry(geometry_ptr)) {
            utility::LogWarning(
                    "[DrawGeometriesWithVertexSelection] Failed adding "
                    "geometry.");
            utility::LogWarning(
                    "[DrawGeometriesWithVertexSelection] Possibly due to bad "
                    "geometry or wrong geometry type.");
            return false;
        }
    }
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}

}  // namespace visualization
}  // namespace open3d
