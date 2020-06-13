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

#include "Open3D/GUI/Widget.h"
#include "Open3D/Visualization/Rendering/RendererHandle.h"
#include "Open3D/Visualization/Rendering/View.h"

namespace open3d {

namespace geometry {
class AxisAlignedBoundingBox;
}  // namespace geometry

namespace visualization {
class Camera;
class CameraManipulator;
class Scene;
class View;
}  // namespace visualization

namespace gui {

class Color;

class SceneWidget : public Widget {
    using Super = Widget;

public:
    explicit SceneWidget(visualization::Scene& scene);
    ~SceneWidget() override;

    void SetFrame(const Rect& f) override;

    void SetBackgroundColor(const Color& color);
    void SetDiscardBuffers(const visualization::View::TargetBuffers& buffers);

    enum Controls { ROTATE_OBJ, FLY, ROTATE_SUN, ROTATE_IBL, ROTATE_MODEL };
    void SetViewControls(Controls mode);

    void SetupCamera(float verticalFoV,
                     const geometry::AxisAlignedBoundingBox& geometry_bounds,
                     const Eigen::Vector3f& center_of_rotation);
    void SetCameraChangedCallback(
            std::function<void(visualization::Camera*)> on_cam_changed);

    /// Enables changing the directional light with the mouse.
    /// SceneWidget will update the light's direction, so onDirChanged is
    /// only needed if other things need to be updated (like a UI).
    void SelectDirectionalLight(
            visualization::LightHandle dirLight,
            std::function<void(const Eigen::Vector3f&)> on_dir_changed);
    /// Enables showing the skybox while in skybox ROTATE_IBL mode.
    void SetSkyboxHandle(visualization::SkyboxHandle skybox, bool is_on);

    struct ModelDescription {
        visualization::GeometryHandle axes;
        std::vector<visualization::GeometryHandle> point_clouds;
        std::vector<visualization::GeometryHandle> meshes;
        // Optional point clouds drawn instead of 'pointClouds' when rotating.
        // These should have substantially fewer points than the originals
        // so that rotations are faster.
        std::vector<visualization::GeometryHandle> fast_point_clouds;
    };
    void SetModel(const ModelDescription& desc);

    enum class Quality { FAST, BEST };
    void SetRenderQuality(Quality level);
    Quality GetRenderQuality() const;

    enum class CameraPreset {
        PLUS_X,  // at (X, 0, 0), looking (-1, 0, 0)
        PLUS_Y,  // at (0, Y, 0), looking (0, -1, 0)
        PLUS_Z   // at (0, 0, Z), looking (0, 0, 1) [default OpenGL camera]
    };
    void GoToCameraPreset(CameraPreset preset);

    visualization::View* GetView() const;
    visualization::Scene* GetScene() const;

    Widget::DrawResult Draw(const DrawContext& context) override;

    Widget::EventResult Mouse(const MouseEvent& e) override;
    Widget::EventResult Key(const KeyEvent& e) override;
    Widget::DrawResult Tick(const TickEvent& e) override;

private:
    visualization::Camera* GetCamera() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace open3d
