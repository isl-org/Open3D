// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <vector>

#include "open3d/visualization/gui/Widget.h"

namespace open3d {

namespace geometry {
class Geometry3D;
class Image;
}

namespace visualization {
namespace rendering {
class Open3DScene;
class Camera;
}

namespace gui {
/// Geometry editor hooked to SceneWidget to collect points from Geometry3D
/// by drawing 2D shapes. Currently only PointCloud and TriangleMesh are
/// supported.
/// This works only in ROTATE_CAMERA view mode.
/// How to draw 2D shape:
///   - Alt + LeftMouseDrag for a rectangle
///   - Shift + LeftMouseDraw for circle from center to edge
///   - Ctrl + Left Clicks for a polygon, specially, Ctrl + LeftMouseDrag
///     to change vertex position dynamically, and Right click to cancel
///     the latest vertex
///   - Middle click to cancel all points
class GeometryEditor {
public:
    enum class SelectionType { None, Rectangle, Polygon, Circle};
    using Target = std::shared_ptr<const geometry::Geometry3D>;

    explicit GeometryEditor(rendering::Open3DScene* scene);
    ~GeometryEditor() = default;

    /// Start geometry editing.
    /// @param target Geometry to be edited. This target is supposed to be
    ///        added into scene before this call.
    /// @param selectionCallback callback function to notify client the
    ///        selection area is ready and indices could be collected
    ///        by CollectSelectedIndices
    bool Start(Target target, std::function<void(bool)> selectionCallback);
    /// Stop geometry editing.
    void Stop();
    /// Collect selected indices from geometry. For PointCloud the indices of
    /// points is collected and for TriangleMesh the indices of vertices is
    /// collected.
    std::vector<size_t> CollectSelectedIndices();

    /// hooked to SceneWidget Mouse
    Widget::EventResult Mouse(const MouseEvent& e);
    /// hooked to SceneWidget Draw
    Widget::DrawResult Draw(const DrawContext& context, const Rect &frame);

private:
    const std::vector<Eigen::Vector3d> & GetPoints();
    bool SetSelection(SelectionType type);
    void CheckEditable();
    bool AllowEdit();
    bool Started();
    void UpdatePolygonPoint(int x, int y);
    bool CheckPolygonPoint(int x, int y);
    void AddPoint(int x, int y);
    Eigen::Vector2f PointAt(int i);
    Eigen::Vector2f PointAt(int i, int x, int y);
    std::vector<size_t> CropPolygon();
    std::vector<size_t> CropRectangle();
    std::vector<size_t> CropCircle();
private:
    rendering::Open3DScene* scene_;
    rendering::Camera* camera_;
    Target target_;
    bool editable_ = false; // selection area ready flag
    std::function<void(bool)> selection_callback_;
    std::vector<Eigen::Vector2i> selection_;
    SelectionType type_ = SelectionType::None;
};
}  // namespace gui
}  // namespace visualization
}  // namespace open3d
