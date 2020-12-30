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

#pragma once

#include <map>

#include "open3d/visualization/visualizer/Visualizer.h"

namespace open3d {
namespace visualization {

class SelectionPolygon;
class PointCloudPicker;

/// \class VisualizerWithKeyCallback
///
/// \brief Visualizer with custom key callack capabilities.

class VisualizerWithEditAndKeyCallback : public Visualizer {
public:
    typedef std::pair<int, std::function<bool(Visualizer *)>> KeyCallbackPair;

public:
    VisualizerWithEditAndKeyCallback(double voxel_size = -1.0,
                          bool use_dialog = true,
                          const std::string &directory = "")
        : voxel_size_(voxel_size),
          use_dialog_(use_dialog),
          default_directory_(directory) {}
    ~VisualizerWithEditAndKeyCallback() override {}
    VisualizerWithEditAndKeyCallback(const VisualizerWithEditAndKeyCallback &) = delete;
    VisualizerWithEditAndKeyCallback &operator=(const VisualizerWithEditAndKeyCallback &) = delete;

// public:
//     /// \brief Default Constructor.
//     VisualizerWithEditAndKeyCallback();
//     ~VisualizerWithEditAndKeyCallback() override;
//     VisualizerWithEditAndKeyCallback(const VisualizerWithEditAndKeyCallback &) = delete;
//     VisualizerWithEditAndKeyCallback &operator=(const VisualizerWithKeyCallback &) =
//             delete;

public:
    void PrintVisualizerHelp() override;
    void RegisterKeyCallback(int key,
                             std::function<bool(Visualizer *)> callback);
    /// Register callback function with access to GLFW key actions.
    ///
    /// \param key GLFW key value, see [GLFW key
    /// values](https://www.glfw.org/docs/latest/group__keys.html).
    ///
    /// \param callback The callback function. The callback function takes
    /// `Visualizer *`, `action` and `mods` as input and returns a boolean
    /// indicating UpdateGeometry() needs to be run. The `action` can be one of
    /// GLFW_RELEASE (0), GLFW_PRESS (1) or GLFW_REPEAT (2), see [GLFW input
    /// interface](https://www.glfw.org/docs/latest/group__input.html). The
    /// `mods` specifies the modifier key, see [GLFW modifier
    /// key](https://www.glfw.org/docs/latest/group__mods.html).
    void RegisterKeyActionCallback(
            int key, std::function<bool(Visualizer *, int, int)> callback);

public:
    enum class SelectionMode {
        None = 0,
        Rectangle = 1,
        Polygon = 2,
    };

public:
    /// Function to add geometry to the scene and create corresponding shaders.
    ///
    /// \param geometry_ptr The Geometry object.
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr,
                     bool reset_bounding_box = true) override;
    //void PrintVisualizerHelp() override;
    void UpdateWindowTitle() override;
    void BuildUtilities() override;
    int PickPoint(double x, double y);
    std::vector<size_t> &GetPickedPoints();

protected:
    bool InitViewControl() override;
    bool InitRenderOption() override;
    void WindowResizeCallback(GLFWwindow *window, int w, int h) override;
    void MouseMoveCallback(GLFWwindow *window, double x, double y) override;
    void MouseScrollCallback(GLFWwindow *window, double x, double y) override;
    void MouseButtonCallback(GLFWwindow *window,
                             int button,
                             int action,
                             int mods) override;
    void KeyPressCallback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods) override;
    void InvalidateSelectionPolygon();
    void InvalidatePicking();
    void SaveCroppingResult(const std::string &filename = "");

protected:
    std::shared_ptr<SelectionPolygon> selection_polygon_ptr_;
    std::shared_ptr<glsl::SelectionPolygonRenderer>
            selection_polygon_renderer_ptr_;
    SelectionMode selection_mode_ = SelectionMode::None;

    std::shared_ptr<PointCloudPicker> pointcloud_picker_ptr_;
    std::shared_ptr<glsl::PointCloudPickerRenderer>
            pointcloud_picker_renderer_ptr_;

    std::shared_ptr<const geometry::Geometry> original_geometry_ptr_;
    std::shared_ptr<geometry::Geometry> editing_geometry_ptr_;
    std::shared_ptr<glsl::GeometryRenderer> editing_geometry_renderer_ptr_;

    double voxel_size_ = -1.0;
    bool use_dialog_ = true;
    std::string default_directory_;
    unsigned int crop_action_count_ = 0;

protected:
    /**void KeyPressCallback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods) override;**/
    std::string PrintKeyToString(int key);

protected:
    std::map<int, std::function<bool(Visualizer *)>> key_to_callback_;
    std::map<int, std::function<bool(Visualizer *, int, int)>>
            key_action_to_callback_;
};

}  // namespace visualization
}  // namespace open3d
