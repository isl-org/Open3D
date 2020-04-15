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

#include <unordered_map>

#include "Open3D/Visualization/Visualizer/Visualizer.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace visualization {
class SelectionPolygon;
class PointCloudPicker;

class VisualizerWithVertexSelection : public Visualizer {
public:
    enum class SelectionMode { None = 0, Point = 1, Rectangle = 2, Moving = 3 };

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
    bool UpdateGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr =
                                nullptr) override;
    void PrintVisualizerHelp() override;
    void UpdateWindowTitle() override;
    void BuildUtilities() override;
    void SetPointSize(double size);
    std::vector<int> PickPoints(double x, double y, double w, double h);

    struct PickedPoint {
        int index;
        Eigen::Vector3d coord;
    };
    std::vector<PickedPoint> GetPickedPoints() const;
    void ClearPickedPoints();

    void RegisterSelectionChangedCallback(std::function<void()> f);
    /// Do not change the number of vertices in geometry, but can change the
    /// vertex values and call UpdateGeometry().
    void RegisterSelectionMovingCallback(std::function<void()> f);
    void RegisterSelectionMovedCallback(std::function<void()> f);

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
    float GetDepth(int winX, int winY);
    Eigen::Vector3d CalcDragDelta(double winX, double winY);
    enum DragType { DRAG_MOVING, DRAG_END };
    void DragSelectedPoints(const Eigen::Vector3d &delta, DragType type);
    const std::vector<Eigen::Vector3d> *GetGeometryPoints(
            std::shared_ptr<const geometry::Geometry> geometry);

protected:
    std::shared_ptr<SelectionPolygon> selection_polygon_ptr_;
    std::shared_ptr<glsl::SelectionPolygonRenderer>
            selection_polygon_renderer_ptr_;
    SelectionMode selection_mode_ = SelectionMode::None;
    Eigen::Vector2d mouse_down_pos_;
    std::vector<int> points_in_rect_;
    float drag_depth_;

    std::shared_ptr<PointCloudPicker> pointcloud_picker_ptr_;
    std::shared_ptr<glsl::PointCloudPickerRenderer>
            pointcloud_picker_renderer_ptr_;

    std::shared_ptr<const geometry::Geometry> geometry_ptr_;
    std::shared_ptr<glsl::GeometryRenderer> geometry_renderer_ptr_;

    RenderOption pick_point_opts_;

    std::shared_ptr<geometry::PointCloud> ui_points_geometry_ptr_;
    std::shared_ptr<glsl::GeometryRenderer> ui_points_renderer_ptr_;

    std::unordered_map<int, Eigen::Vector3d> selected_points_;
    std::unordered_map<int, Eigen::Vector3d> selected_points_before_drag_;
    std::shared_ptr<geometry::PointCloud> ui_selected_points_geometry_ptr_;
    std::shared_ptr<glsl::GeometryRenderer> ui_selected_points_renderer_ptr_;

    std::function<void()> on_selection_changed_;
    std::function<void()> on_selection_moving_;
    std::function<void()> on_selection_moved_;
};

}  // namespace visualization
}  // namespace open3d
