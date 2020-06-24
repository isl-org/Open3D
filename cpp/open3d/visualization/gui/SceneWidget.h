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
class Scene;
class View;
}  // namespace rendering
}  // namespace visualization

namespace visualization {
namespace gui {

class Color;

class SceneWidget : public Widget {
    using Super = Widget;

public:
    explicit SceneWidget(visualization::rendering::Scene& scene);
    ~SceneWidget() override;

    void SetFrame(const Rect& f) override;

    void SetBackgroundColor(const Color& color);
    void SetDiscardBuffers(
            const visualization::rendering::View::TargetBuffers& buffers);

    enum Controls { ROTATE_OBJ, FLY, ROTATE_SUN, ROTATE_IBL, ROTATE_MODEL };
    void SetViewControls(Controls mode);

    void SetupCamera(float verticalFoV,
                     const geometry::AxisAlignedBoundingBox& geometry_bounds,
                     const Eigen::Vector3f& center_of_rotation);
    void SetCameraChangedCallback(
            std::function<void(visualization::rendering::Camera*)>
                    on_cam_changed);

    /// Enables changing the directional light with the mouse.
    /// SceneWidget will update the light's direction, so onDirChanged is
    /// only needed if other things need to be updated (like a UI).
    void SelectDirectionalLight(
            visualization::rendering::LightHandle dirLight,
            std::function<void(const Eigen::Vector3f&)> on_dir_changed);
    /// Enables showing the skybox while in skybox ROTATE_IBL mode.
    void SetSkyboxHandle(visualization::rendering::SkyboxHandle skybox,
                         bool is_on);

    struct ModelDescription {
        visualization::rendering::GeometryHandle axes;
        std::vector<visualization::rendering::GeometryHandle> point_clouds;
        std::vector<visualization::rendering::GeometryHandle> meshes;
        // Optional point clouds drawn instead of 'pointClouds' when rotating.
        // These should have substantially fewer points than the originals
        // so that rotations are faster.
        std::vector<visualization::rendering::GeometryHandle> fast_point_clouds;
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

    visualization::rendering::View* GetView() const;
    visualization::rendering::Scene* GetScene() const;

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
