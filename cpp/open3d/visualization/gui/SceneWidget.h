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

#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/View.h"

namespace open3d {

namespace geometry {
class AxisAlignedBoundingBox;
}  // namespace geometry

namespace visualization {
namespace rendering {
class Camera;
class CameraManipulator;
class Open3DScene;
class View;
}  // namespace rendering
}  // namespace visualization

namespace visualization {
namespace gui {

class Color;

class SceneWidget : public Widget {
    using Super = Widget;

public:
    explicit SceneWidget();
    ~SceneWidget() override;

    void SetFrame(const Rect& f) override;

    enum Controls { ROTATE_CAMERA, FLY, ROTATE_SUN, ROTATE_IBL, ROTATE_MODEL };
    void SetViewControls(Controls mode);

    void SetupCamera(float verticalFoV,
                     const geometry::AxisAlignedBoundingBox& geometry_bounds,
                     const Eigen::Vector3f& center_of_rotation);
    void SetOnCameraChanged(
            std::function<void(visualization::rendering::Camera*)>
                    on_cam_changed);

    /// Enables changing the directional light with the mouse.
    /// SceneWidget will update the light's direction, so onDirChanged is
    /// only needed if other things need to be updated (like a UI).
    void SetOnSunDirectionChanged(
            std::function<void(const Eigen::Vector3f&)> on_dir_changed);
    /// Enables showing the skybox while in skybox ROTATE_IBL mode.
    void ShowSkybox(bool is_on);

    void SetScene(std::shared_ptr<rendering::Open3DScene> scene);
    std::shared_ptr<rendering::Open3DScene> GetScene() const;

    rendering::View* GetRenderView() const;  // is nullptr if no scene

    /// Enable (or disable) caching of scene to improve UI responsiveness when
    /// dealing with large scenes (especially point clouds)
    void EnableSceneCaching(bool enable);

    /// Forces the scene to redraw regardless of Renderer caching
    /// settings.
    void ForceRedraw();
    enum class Quality { FAST, BEST };
    void SetRenderQuality(Quality level);
    Quality GetRenderQuality() const;

    enum class CameraPreset {
        PLUS_X,  // at (X, 0, 0), looking (-1, 0, 0)
        PLUS_Y,  // at (0, Y, 0), looking (0, -1, 0)
        PLUS_Z   // at (0, 0, Z), looking (0, 0, 1) [default OpenGL camera]
    };
    void GoToCameraPreset(CameraPreset preset);

    void Layout(const Theme& theme) override;
    Widget::DrawResult Draw(const DrawContext& context) override;

    Widget::EventResult Mouse(const MouseEvent& e) override;
    Widget::EventResult Key(const KeyEvent& e) override;
    Widget::DrawResult Tick(const TickEvent& e) override;

private:
    visualization::rendering::Camera* GetCamera() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
