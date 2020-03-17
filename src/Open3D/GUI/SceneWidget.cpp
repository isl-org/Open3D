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

#include "SceneWidget.h"

#include "Color.h"
#include "Events.h"

#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Visualization/Rendering/Camera.h"
#include "Open3D/Visualization/Rendering/Scene.h"
#include "Open3D/Visualization/Rendering/View.h"

#include "Open3D/Visualization/Rendering/CameraInteractor.h"
#include "Open3D/Visualization/Rendering/LightDirectionInteractor.h"

#include <Eigen/Geometry>

#include <set>

#define ENABLE_PAN 1

namespace open3d {
namespace gui {

static const double NEAR_PLANE = 0.1;
static const double MIN_FAR_PLANE = 1.0;

// ----------------------------------------------------------------------------
class MouseInteractor {
public:
    virtual ~MouseInteractor() = default;

    virtual visualization::MatrixInteractor& GetMatrixInteractor() = 0;
    virtual void Mouse(const MouseEvent& e) = 0;
    virtual void Key(const KeyEvent& e) = 0;
    virtual Widget::DrawResult Tick(const TickEvent& e) = 0;
};

class RotateSunInteractor : public MouseInteractor {
public:
    RotateSunInteractor(visualization::Scene* scene,
                        visualization::Camera* camera)
        : lightDir_(std::make_unique<visualization::LightDirectionInteractor>(
                  scene, camera)) {}

    visualization::MatrixInteractor& GetMatrixInteractor() override {
        return *lightDir_.get();
    }

    void SetDirectionalLight(
            visualization::LightHandle dirLight,
            std::function<void(const Eigen::Vector3f&)> onChanged) {
        lightDir_->SetDirectionalLight(dirLight);
        onLightDirChanged_ = onChanged;
    }

    void Mouse(const MouseEvent& e) override {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
                mouseDownX_ = e.x;
                mouseDownY_ = e.y;
                lightDir_->StartMouseDrag();
                break;
            case MouseEvent::DRAG: {
                int dx = e.x - mouseDownX_;
                int dy = e.y - mouseDownY_;
                lightDir_->Rotate(dx, dy);
                if (onLightDirChanged_) {
                    onLightDirChanged_(lightDir_->GetCurrentDirection());
                }
                break;
            }
            case MouseEvent::WHEEL: {
                break;
            }
            case MouseEvent::BUTTON_UP:
                lightDir_->EndMouseDrag();
                break;
            default:
                break;
        }
    }

    void Key(const KeyEvent& e) override {}

    Widget::DrawResult Tick(const TickEvent& e) override {
        return Widget::DrawResult::NONE;
    }

private:
    std::unique_ptr<visualization::LightDirectionInteractor> lightDir_;
    int mouseDownX_ = 0;
    int mouseDownY_ = 0;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged_;
};

class FPSInteractor : public MouseInteractor {
public:
    FPSInteractor(visualization::Camera* camera)
        : camera_(camera),
          cameraControls_(std::make_unique<visualization::CameraInteractor>(
                  camera, MIN_FAR_PLANE)) {}

    visualization::MatrixInteractor& GetMatrixInteractor() override {
        return *cameraControls_.get();
    }

    void Mouse(const MouseEvent& e) override {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
                mouseDownX_ = e.x;
                mouseDownY_ = e.y;
                cameraControls_->SetCenterOfRotation(camera_->GetPosition());
                cameraControls_->StartMouseDrag();
                break;
            case MouseEvent::DRAG: {
                int dx = e.x - mouseDownX_;
                int dy = e.y - mouseDownY_;
                cameraControls_->Rotate(-dx, -dy);
                break;
            }
            case MouseEvent::WHEEL: {
                break;
            }
            case MouseEvent::BUTTON_UP:
                cameraControls_->EndMouseDrag();
                break;
            default:
                break;
        }
    }

    void Key(const KeyEvent& e) override {
        if (e.type == KeyEvent::Type::DOWN) {
            keysDown_.insert(e.key);
        } else if (e.type == KeyEvent::Type::UP) {
            keysDown_.erase(e.key);
        }
    }

    Widget::DrawResult Tick(const TickEvent& e) override {
        const float dist = 0.1f;
        const float angleRad = 0.01f;

        bool redraw = false;

        if (keysDown_.find('a') != keysDown_.end()) {
            cameraControls_->MoveLocal({-dist, 0, 0});
            redraw = true;
        }
        if (keysDown_.find('d') != keysDown_.end()) {
            cameraControls_->MoveLocal({dist, 0, 0});
            redraw = true;
        }
        if (keysDown_.find('w') != keysDown_.end()) {
            cameraControls_->MoveLocal({0, 0, -dist});
            redraw = true;
        }
        if (keysDown_.find('s') != keysDown_.end()) {
            cameraControls_->MoveLocal({0, 0, dist});
            redraw = true;
        }
        if (keysDown_.find('q') != keysDown_.end()) {
            cameraControls_->MoveLocal({0, dist, 0});
            redraw = true;
        }
        if (keysDown_.find('z') != keysDown_.end()) {
            cameraControls_->MoveLocal({0, -dist, 0});
            redraw = true;
        }
        if (keysDown_.find(KEY_UP) != keysDown_.end()) {
            cameraControls_->RotateLocal(angleRad, {1, 0, 0});
            redraw = true;
        }
        if (keysDown_.find(KEY_DOWN) != keysDown_.end()) {
            cameraControls_->RotateLocal(-angleRad, {1, 0, 0});
            redraw = true;
        }
        if (keysDown_.find(KEY_LEFT) != keysDown_.end()) {
            cameraControls_->RotateLocal(angleRad, {0, 1, 0});
            redraw = true;
        }
        if (keysDown_.find(KEY_RIGHT) != keysDown_.end()) {
            cameraControls_->RotateLocal(-angleRad, {0, 1, 0});
            redraw = true;
        }

        if (redraw) {
            return Widget::DrawResult::REDRAW;
        }
        return Widget::DrawResult::NONE;
    }

private:
    visualization::Camera* camera_;
    std::unique_ptr<visualization::CameraInteractor> cameraControls_;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged_;
    int mouseDownX_ = 0;
    int mouseDownY_ = 0;
    std::set<uint32_t> keysDown_;
};

class RotateObjectInteractor : public MouseInteractor {
public:
    RotateObjectInteractor(visualization::Camera* camera)
        : cameraControls_(std::make_unique<visualization::CameraInteractor>(
                  camera, MIN_FAR_PLANE)) {}

    visualization::MatrixInteractor& GetMatrixInteractor() override {
        return *cameraControls_.get();
    }

    void SetCenterOfRotation(const Eigen::Vector3f& center) {
        cameraControls_->SetCenterOfRotation(center);
    }

    void Mouse(const MouseEvent& e) override {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
                mouseDownX_ = e.x;
                mouseDownY_ = e.y;
                if (e.button.button == MouseButton::LEFT) {
                    if (e.modifiers & int(KeyModifier::SHIFT)) {
                        state_ = State::DOLLY;
#if ENABLE_PAN
                    } else if (e.modifiers & int(KeyModifier::CTRL)) {
                        state_ = State::PAN;
#endif  // ENABLE_PAN
                    } else if (e.modifiers & int(KeyModifier::META)) {
                        state_ = State::ROTATE_Z;
                    } else {
                        state_ = State::ROTATE_XY;
                    }
#if ENABLE_PAN
                } else if (e.button.button == MouseButton::RIGHT) {
                    state_ = State::PAN;
#endif  // ENABLE_PAN
                }
                cameraControls_->StartMouseDrag();
                break;
            case MouseEvent::DRAG: {
                int dx = e.x - mouseDownX_;
                int dy = e.y - mouseDownY_;
                switch (state_) {
                    case State::NONE:
                        break;
                    case State::PAN:
                        cameraControls_->Pan(dx, dy);
                        break;
                    case State::DOLLY:
                        cameraControls_->Dolly(dy,
                                               visualization::MatrixInteractor::
                                                       DragType::MOUSE);
                        break;
                    case State::ZOOM:
                        cameraControls_->Zoom(dy,
                                              visualization::MatrixInteractor::
                                                      DragType::MOUSE);
                        break;
                    case State::ROTATE_XY:
                        cameraControls_->Rotate(dx, dy);
                        break;
                    case State::ROTATE_Z:
                        cameraControls_->RotateZ(dx, dy);
                        break;
                }
                break;
            }
            case MouseEvent::WHEEL: {
                if (e.modifiers & int(KeyModifier::SHIFT)) {
                    cameraControls_->Zoom(
                            e.wheel.dy,
                            e.wheel.isTrackpad
                                    ? visualization::MatrixInteractor::
                                              DragType::TWO_FINGER
                                    : visualization::MatrixInteractor::
                                              DragType::WHEEL);
                } else {
                    cameraControls_->Dolly(
                            e.wheel.dy,
                            e.wheel.isTrackpad
                                    ? visualization::MatrixInteractor::
                                              DragType::TWO_FINGER
                                    : visualization::MatrixInteractor::
                                              DragType::WHEEL);
                }
                break;
            }
            case MouseEvent::BUTTON_UP:
                cameraControls_->EndMouseDrag();
                state_ = State::NONE;
                break;
            default:
                break;
        }
    }

    void Key(const KeyEvent& e) override {}

    Widget::DrawResult Tick(const TickEvent& e) override {
        return Widget::DrawResult::NONE;
    }

private:
    std::unique_ptr<visualization::CameraInteractor> cameraControls_;
    int mouseDownX_ = 0;
    int mouseDownY_ = 0;

    enum class State { NONE, PAN, DOLLY, ZOOM, ROTATE_XY, ROTATE_Z };
    State state_ = State::NONE;
};

// ----------------------------------------------------------------------------
class Interactors {
public:
    Interactors(visualization::Scene* scene, visualization::Camera* camera) {
        rotate_ = std::make_unique<RotateObjectInteractor>(camera);
        fps_ = std::make_unique<FPSInteractor>(camera);
        lightDir_ = std::make_unique<RotateSunInteractor>(scene, camera);

        current_ = rotate_.get();
    }

    void SetViewSize(const Size& size) {
        rotate_->GetMatrixInteractor().SetViewSize(size.width, size.height);
        fps_->GetMatrixInteractor().SetViewSize(size.width, size.height);
        lightDir_->GetMatrixInteractor().SetViewSize(size.width, size.height);
    }

    void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds) {
        rotate_->GetMatrixInteractor().SetBoundingBox(bounds);
        fps_->GetMatrixInteractor().SetBoundingBox(bounds);
        lightDir_->GetMatrixInteractor().SetBoundingBox(bounds);
    }

    void SetCenterOfRotation(const Eigen::Vector3f& center) {
        rotate_->SetCenterOfRotation(center);
    }

    void SetDirectionalLight(
            visualization::LightHandle dirLight,
            std::function<void(const Eigen::Vector3f&)> onChanged) {
        lightDir_->SetDirectionalLight(dirLight, onChanged);
    }

    SceneWidget::Controls GetControls() const {
        if (current_ == fps_.get()) {
            return SceneWidget::Controls::FPS;
        } else if (current_ == lightDir_.get()) {
            return SceneWidget::Controls::ROTATE_SUN;
        } else {
            return SceneWidget::Controls::ROTATE_OBJ;
        }
    }

    void SetControls(SceneWidget::Controls mode) {
        switch (mode) {
            case SceneWidget::Controls::ROTATE_OBJ:
                current_ = rotate_.get();
                break;
            case SceneWidget::Controls::FPS:
                current_ = fps_.get();
                break;
            case SceneWidget::Controls::ROTATE_SUN:
                current_ = lightDir_.get();
                break;
            case SceneWidget::Controls::ROTATE_IBL:
                break;
        }
    }

    void Mouse(const MouseEvent& e) {
        if (current_ == rotate_.get()) {
            if (e.type == MouseEvent::Type::BUTTON_DOWN &&
                (e.button.button == MouseButton::MIDDLE ||
                 e.modifiers & int(KeyModifier::ALT))) {
                override_ = lightDir_.get();
            }
        }

        if (override_) {
            override_->Mouse(e);
        } else if (current_) {
            current_->Mouse(e);
        }

        if (override_ && e.type == MouseEvent::Type::BUTTON_UP) {
            override_ = nullptr;
        }
    }

    void Key(const KeyEvent& e) {
        if (current_) {
            current_->Key(e);
        }
    }

    Widget::DrawResult Tick(const TickEvent& e) {
        if (current_) {
            return current_->Tick(e);
        }
        return Widget::DrawResult::NONE;
    }

private:
    std::unique_ptr<RotateObjectInteractor> rotate_;
    std::unique_ptr<FPSInteractor> fps_;
    std::unique_ptr<RotateSunInteractor> lightDir_;

    MouseInteractor* current_ = nullptr;
    MouseInteractor* override_ = nullptr;
};

// ----------------------------------------------------------------------------
struct SceneWidget::Impl {
    visualization::Scene& scene;
    visualization::ViewHandle viewId;
    visualization::Camera* camera;
    geometry::AxisAlignedBoundingBox bounds;
    std::shared_ptr<Interactors> controls;
    bool frameChanged = false;
    visualization::LightHandle dirLight;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged;

    explicit Impl(visualization::Scene& aScene) : scene(aScene) {}
};

SceneWidget::SceneWidget(visualization::Scene& scene) : impl_(new Impl(scene)) {
    impl_->viewId = scene.AddView(0, 0, 1, 1);

    auto view = impl_->scene.GetView(impl_->viewId);
    impl_->camera = view->GetCamera();
    impl_->controls = std::make_shared<Interactors>(&scene, view->GetCamera());
}

SceneWidget::~SceneWidget() { impl_->scene.RemoveView(impl_->viewId); }

void SceneWidget::SetFrame(const Rect& f) {
    Super::SetFrame(f);

    impl_->controls->SetViewSize(Size(f.width, f.height));

    // We need to update the viewport and camera, but we can't do it here
    // because we need to know the window height to convert the frame
    // to OpenGL coordinates. We will actually do the updating in Draw().
    impl_->frameChanged = true;
}

void SceneWidget::SetBackgroundColor(const Color& color) {
    auto view = impl_->scene.GetView(impl_->viewId);
    view->SetClearColor({color.GetRed(), color.GetGreen(), color.GetBlue()});
}

void SceneWidget::SetDiscardBuffers(
        const visualization::View::TargetBuffers& buffers) {
    auto view = impl_->scene.GetView(impl_->viewId);
    view->SetDiscardBuffers(buffers);
}

void SceneWidget::SetupCamera(
        float verticalFoV,
        const geometry::AxisAlignedBoundingBox& geometryBounds,
        const Eigen::Vector3f& centerOfRotation) {
    impl_->bounds = geometryBounds;
    impl_->controls->SetBoundingBox(geometryBounds);

    auto f = GetFrame();
    float aspect = 1.0f;
    if (f.height > 0) {
        aspect = float(f.width) / float(f.height);
    }
    auto far = std::max(MIN_FAR_PLANE, 2.0 * geometryBounds.GetExtent().norm());
    GetCamera()->SetProjection(verticalFoV, aspect, NEAR_PLANE, far,
                               visualization::Camera::FovType::Vertical);

    GoToCameraPreset(CameraPreset::PLUS_Z);  // default OpenGL view
}

void SceneWidget::SelectDirectionalLight(
        visualization::LightHandle dirLight,
        std::function<void(const Eigen::Vector3f&)> onDirChanged) {
    impl_->dirLight = dirLight;
    impl_->onLightDirChanged = onDirChanged;
    impl_->controls->SetDirectionalLight(
            dirLight, [this, dirLight](const Eigen::Vector3f& dir) {
                impl_->scene.SetLightDirection(dirLight, dir);
                if (impl_->onLightDirChanged) {
                    impl_->onLightDirChanged(dir);
                }
            });
}

void SceneWidget::SetViewControls(Controls mode) {
    if (mode == Controls::ROTATE_OBJ &&
        impl_->controls->GetControls() == Controls::FPS) {
        // If we're going from FPS to standard rotate obj, reset the
        // camera
        impl_->controls->SetControls(mode);
        GoToCameraPreset(CameraPreset::PLUS_Z);
    } else {
        impl_->controls->SetControls(mode);
    }
}

void SceneWidget::GoToCameraPreset(CameraPreset preset) {
    auto boundsMax = impl_->bounds.GetMaxBound();
    auto maxDim =
            std::max(boundsMax.x(), std::max(boundsMax.y(), boundsMax.z()));
    maxDim = 1.5f * maxDim;
    auto center = impl_->bounds.GetCenter().cast<float>();
    Eigen::Vector3f eye, up;
    switch (preset) {
        case CameraPreset::PLUS_X: {
            eye = Eigen::Vector3f(maxDim, center.y(), center.z());
            up = Eigen::Vector3f(0, 1, 0);
            break;
        }
        case CameraPreset::PLUS_Y: {
            eye = Eigen::Vector3f(center.x(), maxDim, center.z());
            up = Eigen::Vector3f(1, 0, 0);
            break;
        }
        case CameraPreset::PLUS_Z: {
            eye = Eigen::Vector3f(center.x(), center.y(), maxDim);
            up = Eigen::Vector3f(0, 1, 0);
            break;
        }
    }
    impl_->camera->LookAt(center, eye, up);
    impl_->controls->SetCenterOfRotation(center);
}

visualization::View* SceneWidget::GetView() const {
    return impl_->scene.GetView(impl_->viewId);
}

visualization::Scene* SceneWidget::GetScene() const { return &impl_->scene; }

visualization::Camera* SceneWidget::GetCamera() const {
    auto view = impl_->scene.GetView(impl_->viewId);
    return view->GetCamera();
}

Widget::DrawResult SceneWidget::Draw(const DrawContext& context) {
    // If the widget has changed size we need to update the viewport and the
    // camera. We can't do it in SetFrame() because we need to know the height
    // of the window to convert to OpenGL coordinates for the viewport.
    if (impl_->frameChanged) {
        impl_->frameChanged = false;

        auto f = GetFrame();
        impl_->controls->SetViewSize(Size(f.width, f.height));
        // GUI has origin of Y axis at top, but renderer has it at bottom
        // so we need to convert coordinates.
        int y = context.screenHeight - (f.height + f.y);

        auto view = impl_->scene.GetView(impl_->viewId);
        view->SetViewport(f.x, y, f.width, f.height);

        auto* camera = GetCamera();
        float aspect = 1.0f;
        if (f.height > 0) {
            aspect = float(f.width) / float(f.height);
        }
        GetCamera()->SetProjection(camera->GetFieldOfView(), aspect,
                                   camera->GetNear(), camera->GetFar(),
                                   camera->GetFieldOfViewType());
    }

    // The actual drawing is done later, at the end of drawing in
    // Window::OnDraw(), in FilamentRenderer::Draw(). We can always
    // return NONE because any changes this frame will automatically
    // be rendered (unlike the ImGUI parts).
    return Widget::DrawResult::NONE;
}

void SceneWidget::Mouse(const MouseEvent& e) { impl_->controls->Mouse(e); }

void SceneWidget::Key(const KeyEvent& e) { impl_->controls->Key(e); }

Widget::DrawResult SceneWidget::Tick(const TickEvent& e) {
    return impl_->controls->Tick(e);
}

}  // namespace gui
}  // namespace open3d
