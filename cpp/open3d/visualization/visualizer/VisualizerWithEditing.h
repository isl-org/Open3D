// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/visualizer/Visualizer.h"

namespace open3d {

namespace visualization {
class SelectionPolygon;
class PointCloudPicker;

/// \class VisualizerWithEditing
///
/// \brief Visualizer with editing capabilities.
class VisualizerWithEditing : public Visualizer {
public:
    enum class SelectionMode {
        None = 0,
        Rectangle = 1,
        Polygon = 2,
    };

public:
    VisualizerWithEditing(double voxel_size = -1.0,
                          bool use_dialog = true,
                          const std::string &directory = "")
        : voxel_size_(voxel_size),
          use_dialog_(use_dialog),
          default_directory_(directory) {}
    ~VisualizerWithEditing() override {}
    VisualizerWithEditing(const VisualizerWithEditing &) = delete;
    VisualizerWithEditing &operator=(const VisualizerWithEditing &) = delete;

public:
    /// Function to add geometry to the scene and create corresponding shaders.
    ///
    /// \param geometry_ptr The Geometry object.
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr,
                     bool reset_bounding_box = true) override;
    void PrintVisualizerHelp() override;
    void UpdateWindowTitle() override;
    void BuildUtilities() override;
    int PickPoint(double x, double y);
    std::vector<size_t> &GetPickedPoints();
    std::shared_ptr<geometry::Geometry> GetCroppedGeometry() const;

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
};

}  // namespace visualization
}  // namespace open3d
