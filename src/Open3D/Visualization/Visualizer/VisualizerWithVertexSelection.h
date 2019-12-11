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

#include "Open3D/Visualization/Visualizer/Visualizer.h"

#include <unordered_map>

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace visualization {
class SelectionPolygon;
class PointCloudPicker;

class VisualizerWithVertexSelection : public Visualizer {
public:
    enum class SelectionMode {
        None = 0,
        Point = 1,
        Rectangle = 2,
    };

public:
    VisualizerWithVertexSelection() {}
    ~VisualizerWithVertexSelection() override {}
    VisualizerWithVertexSelection(const VisualizerWithVertexSelection &) =
            delete;
    VisualizerWithVertexSelection &operator=(
            const VisualizerWithVertexSelection &) = delete;

public:
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr,
                     bool reset_bounding_box = true) override;
    bool UpdateGeometry() override;
    void PrintVisualizerHelp() override;
    void UpdateWindowTitle() override;
    void BuildUtilities() override;
    std::vector<int> PickPoints(double x, double y, double w, double h);
    std::vector<int> GetPickedPoints() const;
    void ClearPickedPoints();
    void SetPointSize(double size);

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
    void AddPickedPoints(const std::vector<int> indices);
    void RemovePickedPoints(const std::vector<int> indices);

protected:
    std::shared_ptr<SelectionPolygon> selection_polygon_ptr_;
    std::shared_ptr<glsl::SelectionPolygonRenderer>
            selection_polygon_renderer_ptr_;
    SelectionMode selection_mode_ = SelectionMode::None;
    Eigen::Vector2d mouse_down_pos_;
    std::vector<int> points_in_rect_;

    std::shared_ptr<PointCloudPicker> pointcloud_picker_ptr_;
    std::shared_ptr<glsl::PointCloudPickerRenderer>
            pointcloud_picker_renderer_ptr_;

    std::shared_ptr<const geometry::Geometry> geometry_ptr_;
    std::shared_ptr<glsl::GeometryRenderer> geometry_renderer_ptr_;

    RenderOption pick_point_opts_;

    std::shared_ptr<geometry::PointCloud> ui_points_geometry_ptr_;
    std::shared_ptr<glsl::GeometryRenderer> ui_points_renderer_ptr_;

    std::unordered_map<int, Eigen::Vector3d> selected_points_;
    std::shared_ptr<geometry::PointCloud> ui_selected_points_geometry_ptr_;
    std::shared_ptr<glsl::GeometryRenderer> ui_selected_points_renderer_ptr_;
};

}  // namespace visualization
}  // namespace open3d
