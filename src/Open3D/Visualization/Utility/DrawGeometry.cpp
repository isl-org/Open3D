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

#include "Open3D/Visualization/Utility/DrawGeometry.h"

#include "Open3D/Visualization/Visualizer/ViewControlWithCustomAnimation.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithEditing.h"
#include "Open3D/Visualization/Visualizer/Visualizer.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithCustomAnimation.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithEditing.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithKeyCallback.h"

namespace open3d {
namespace visualization {

bool DrawGeometries(const std::vector<std::shared_ptr<const geometry::Geometry>>
                            &geometry_ptrs,
                    const std::string &window_name /* = "Open3D"*/,
                    int width /* = 640*/,
                    int height /* = 480*/,
                    int left /* = 50*/,
                    int top /* = 50*/) {
    Visualizer visualizer;
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                          top) == false) {
        utility::LogError("[DrawGeometries] Failed creating OpenGL window.\n");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (visualizer.AddGeometry(geometry_ptr) == false) {
            utility::LogWarning("[DrawGeometries] Failed adding geometry.\n");
            utility::LogWarning(
                    "[DrawGeometries] Possibly due to bad geometry or wrong "
                    "geometry type.\n");
            return false;
        }
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
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                          top) == false) {
        utility::LogError(
                "[DrawGeometriesWithCustomAnimation] Failed creating OpenGL "
                "window.\n");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (visualizer.AddGeometry(geometry_ptr) == false) {
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Failed adding "
                    "geometry.\n");
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Possibly due to bad "
                    "geometry or wrong geometry type.\n");
            return false;
        }
    }
    auto &view_control =
            (ViewControlWithCustomAnimation &)visualizer.GetViewControl();
    if (json_filename.empty() == false) {
        if (view_control.LoadTrajectoryFromJsonFile(json_filename) == false) {
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Failed loading json "
                    "file.\n");
            utility::LogWarning(
                    "[DrawGeometriesWithCustomAnimation] Possibly due to bad "
                    "file or file does not contain trajectory.\n");
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
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                          top) == false) {
        utility::LogError(
                "[DrawGeometriesWithAnimationCallback] Failed creating OpenGL "
                "window.\n");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (visualizer.AddGeometry(geometry_ptr) == false) {
            utility::LogWarning(
                    "[DrawGeometriesWithAnimationCallback] Failed adding "
                    "geometry.\n");
            utility::LogWarning(
                    "[DrawGeometriesWithAnimationCallback] Possibly due to bad "
                    "geometry or wrong geometry type.\n");
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
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                          top) == false) {
        utility::LogError(
                "[DrawGeometriesWithKeyCallbacks] Failed creating OpenGL "
                "window.\n");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (visualizer.AddGeometry(geometry_ptr) == false) {
            utility::LogWarning(
                    "[DrawGeometriesWithKeyCallbacks] Failed adding "
                    "geometry.\n");
            utility::LogWarning(
                    "[DrawGeometriesWithKeyCallbacks] Possibly due to bad "
                    "geometry or wrong geometry type.\n");
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
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                          top) == false) {
        utility::LogError(
                "[DrawGeometriesWithEditing] Failed creating OpenGL window.\n");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (visualizer.AddGeometry(geometry_ptr) == false) {
            utility::LogWarning(
                    "[DrawGeometriesWithEditing] Failed adding geometry.\n");
            utility::LogWarning(
                    "[DrawGeometriesWithEditing] Possibly due to bad geometry "
                    "or wrong geometry type.\n");
            return false;
        }
    }
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}

}  // namespace visualization
}  // namespace open3d
