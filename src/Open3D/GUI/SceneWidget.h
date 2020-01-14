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

#include "Open3D/Visualization/Rendering/View.h"
#include "Open3D/Visualization/Rendering/RendererHandle.h"

#include "Widget.h"

namespace open3d {

namespace visualization
{
    class Scene;
    class Camera;
    class CameraManipulator;
}

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

    visualization::Scene* GetScene() const;
    visualization::CameraManipulator* GetCameraManipulator() const;

    // switchCamera flag make center of geometry become camera's POI;
    void SetSelectedGeometry(const visualization::GeometryHandle& geometry, bool switchCamera);
    void SetCameraPOI(const Eigen::Vector3f& location);

    Widget::DrawResult Draw(const DrawContext& context) override;

    void Mouse(const MouseEvent& e) override;
    void Key(const KeyEvent& e) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    struct CameraControlsState {
        Eigen::Vector3f poi;
        float orbitHeight = 0.f;
        float rotationSpeed = M_PI;

        bool orbiting = false;

        // use |mousePos - frameDelta| to get
        // mouse position at start of frame
        float lastMouseX = 0.f;
        float lastMouseY = 0.f;

        float frameDx = 0.f;
        float frameDy = 0.f;
        float frameWheelDelta = 0.f;

        void Reset();
    } cameraControlsState_;
};

} // gui
} // open3d
