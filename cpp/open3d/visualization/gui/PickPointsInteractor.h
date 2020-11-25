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

#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/rendering/MatrixInteractorLogic.h"

#include <queue>

namespace open3d {

namespace geometry {
class Image;
}  // namespace geometry

namespace visualization {

namespace rendering {
class Camera;
struct Material;
class MatrixInteractorLogic;
class Open3DScene;
}  // namespace rendering

namespace gui {

// This is an internal class used by SceneWidget
class PickPointsInteractor : public SceneWidget::MouseInteractor {
public:
    PickPointsInteractor(rendering::Open3DScene *scene,
                         rendering::Camera *camera);
    virtual ~PickPointsInteractor();

    void SetPointSize(int px);

    /// Sets the points that can be picked. Limited to 16 million or less
    void SetPickablePoints(const std::vector<Eigen::Vector3d>& points);

    /// Indicates that the selection scene must be redrawn and the picking
    /// pixels retrieved again before picking.
    void SetNeedsRedraw();

    /// Calls the provided function when points are picked:
    ///    f(indices, key_modifiers)
    void SetOnPointsPicked(std::function<void(const std::vector<size_t>&, int)> f);

    rendering::MatrixInteractorLogic& GetMatrixInteractor() override;
    void Mouse(const MouseEvent& e) override;
    void Key(const KeyEvent& e) override;

protected:
    void OnPickImageDone(std::shared_ptr<geometry::Image> img);

    rendering::Material MakeMaterial();

private:
    rendering::Open3DScene *scene_;
    rendering::Camera *camera_;

    std::function<void(const std::vector<size_t>&, int)> on_picked_;
    int point_size_ = 3;
    rendering::MatrixInteractorLogic matrix_logic_;
    std::shared_ptr<rendering::Open3DScene> picking_scene_;
    std::vector<Eigen::Vector3d> pickable_points_;
    std::shared_ptr<geometry::Image> pick_image_;
    bool dirty_ = true;
    struct PickInfo {
        gui::Rect rect;
        int keymods;
    };
    std::queue<PickInfo> pending_;

};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
