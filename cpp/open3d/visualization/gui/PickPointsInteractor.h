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

#include <queue>

#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/rendering/MatrixInteractorLogic.h"

namespace open3d {

namespace geometry {
class Geometry3D;
class Image;
}  // namespace geometry

namespace visualization {

namespace rendering {
class Camera;
struct MaterialRecord;
class MatrixInteractorLogic;
class Open3DScene;
}  // namespace rendering

namespace gui {

class SelectionIndexLookup;

// This is an internal class used by SceneWidget
class PickPointsInteractor : public SceneWidget::MouseInteractor {
public:
    PickPointsInteractor(rendering::Open3DScene* scene,
                         rendering::Camera* camera);
    virtual ~PickPointsInteractor();

    void SetPointSize(int px);

    /// Sets the points that can be picked. Limited to 16 million or less
    /// points/vertices total. Geometry pointers will not be cached.
    void SetPickableGeometry(
            const std::vector<SceneWidget::PickableGeometry>& geometry);

    /// Indicates that the selection scene must be redrawn and the picking
    /// pixels retrieved again before picking.
    void SetNeedsRedraw();

    /// Calls the provided function when points are picked:
    ///    f(indices, key_modifiers)
    void SetOnPointsPicked(
            std::function<void(
                    const std::map<
                            std::string,
                            std::vector<std::pair<size_t, Eigen::Vector3d>>>&,
                    int)> f);

    /// Calls the provided function when a "UI" needs to be drawn. This is
    /// used for polygon picking to draw the polygon in progress.
    /// will be passed
    void SetOnUIChanged(
            std::function<void(const std::vector<Eigen::Vector2i>&)>);

    /// Calls the provided function when polygon picking is initiated
    void SetOnStartedPolygonPicking(std::function<void()> on_poly_pick);

    void DoPick();
    void ClearPick();

    rendering::MatrixInteractorLogic& GetMatrixInteractor() override;
    void Mouse(const MouseEvent& e) override;
    void Key(const KeyEvent& e) override;

protected:
    void OnPickImageDone(std::shared_ptr<geometry::Image> img);

    rendering::MaterialRecord MakeMaterial();

private:
    rendering::Open3DScene* scene_;
    rendering::Camera* camera_;

    std::function<void(
            const std::map<std::string,
                           std::vector<std::pair<size_t, Eigen::Vector3d>>>&,
            int)>
            on_picked_;
    std::function<void(const std::vector<Eigen::Vector2i>&)> on_ui_changed_;
    std::function<void()> on_started_poly_pick_;
    int point_size_ = 3;
    rendering::MatrixInteractorLogic matrix_logic_;
    std::shared_ptr<rendering::Open3DScene> picking_scene_;
    std::vector<Eigen::Vector3d> points_;
    // This is a pointer rather than unique_ptr so that we don't have
    // to define this (internal) class in the header file.
    SelectionIndexLookup* lookup_ = nullptr;
    std::shared_ptr<geometry::Image> pick_image_;
    bool dirty_ = true;
    struct PickInfo {
        std::vector<gui::Point> polygon;  // or point, if only one item
        int keymods;
    };
    std::queue<PickInfo> pending_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
