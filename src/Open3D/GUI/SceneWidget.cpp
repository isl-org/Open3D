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

#include "Application.h"
#include "Color.h"
#include "Events.h"

#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Visualization/Rendering/Camera.h"
#include "Open3D/Visualization/Rendering/Scene.h"
#include "Open3D/Visualization/Rendering/View.h"

#include "Open3D/Visualization/Rendering/CameraInteractor.h"
#include "Open3D/Visualization/Rendering/IBLRotationInteractor.h"
#include "Open3D/Visualization/Rendering/LightDirectionInteractor.h"
#include "Open3D/Visualization/Rendering/ModelInteractor.h"

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

private:
    std::unique_ptr<visualization::LightDirectionInteractor> lightDir_;
    int mouseDownX_ = 0;
    int mouseDownY_ = 0;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged_;
};

class RotateIBLInteractor : public MouseInteractor {
public:
    RotateIBLInteractor(visualization::Scene* scene,
                        visualization::Camera* camera)
        : ibl_(std::make_unique<visualization::IBLRotationInteractor>(
                  scene, camera)) {}

    visualization::MatrixInteractor& GetMatrixInteractor() override {
        return *ibl_.get();
    }

    void SetSkyboxHandle(visualization::SkyboxHandle skybox, bool isOn) {
        ibl_->SetSkyboxHandle(skybox, isOn);
    }

    void SetOnChanged(
            std::function<void(const visualization::Camera::Transform&)>
                    onChanged) {
        onRotationChanged_ = onChanged;
    }

    void Mouse(const MouseEvent& e) override {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
                mouseDownX_ = e.x;
                mouseDownY_ = e.y;
                ibl_->StartMouseDrag();
                break;
            case MouseEvent::DRAG: {
                int dx = e.x - mouseDownX_;
                int dy = e.y - mouseDownY_;
                if (e.modifiers & int(KeyModifier::META)) {
                    ibl_->RotateZ(dx, dy);
                } else {
                    ibl_->Rotate(dx, dy);
                }
                if (onRotationChanged_) {
                    onRotationChanged_(ibl_->GetCurrentRotation());
                }
                break;
            }
            case MouseEvent::WHEEL: {
                break;
            }
            case MouseEvent::BUTTON_UP:
                ibl_->EndMouseDrag();
                break;
            default:
                break;
        }
    }

    void Key(const KeyEvent& e) override {}

private:
    std::unique_ptr<visualization::IBLRotationInteractor> ibl_;
    int mouseDownX_ = 0;
    int mouseDownY_ = 0;
    std::function<void(const visualization::Camera::Transform&)>
            onRotationChanged_;
};

class FPSInteractor : public MouseInteractor {
public:
    explicit FPSInteractor(visualization::Camera* camera)
        : cameraControls_(std::make_unique<visualization::CameraInteractor>(
                  camera, MIN_FAR_PLANE)) {}

    visualization::MatrixInteractor& GetMatrixInteractor() override {
        return *cameraControls_.get();
    }

    void Mouse(const MouseEvent& e) override {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
                lastMouseX_ = e.x;
                lastMouseY_ = e.y;
                cameraControls_->StartMouseDrag();
                break;
            case MouseEvent::DRAG: {
                // Use relative movement because user may be moving
                // with keys at the same time.
                int dx = e.x - lastMouseX_;
                int dy = e.y - lastMouseY_;
                if (e.modifiers & int(KeyModifier::META)) {
                    // RotateZ() was not intended to be used for relative
                    // movement, so reset the mouse-down matrix first.
                    cameraControls_->StartMouseDrag();
                    cameraControls_->RotateZ(dx, dy);
                } else {
                    cameraControls_->RotateFPS(-dx, -dy);
                }
                lastMouseX_ = e.x;
                lastMouseY_ = e.y;
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
        if (e.type != KeyEvent::Type::UP) {
            auto& bounds = cameraControls_->GetBoundingBox();
            const float dist = 0.02f * bounds.GetExtent().norm();
            const float angleRad = 0.0075f;

            auto hasKey = [&e](uint32_t key) -> bool { return (e.key == key); };

            if (hasKey('a')) {
                cameraControls_->MoveLocal({-dist, 0, 0});
            }
            if (hasKey('d')) {
                cameraControls_->MoveLocal({dist, 0, 0});
            }
            if (hasKey('w')) {
                cameraControls_->MoveLocal({0, 0, -dist});
            }
            if (hasKey('s')) {
                cameraControls_->MoveLocal({0, 0, dist});
            }
            if (hasKey('q')) {
                cameraControls_->MoveLocal({0, dist, 0});
            }
            if (hasKey('z')) {
                cameraControls_->MoveLocal({0, -dist, 0});
            }
            if (hasKey('e')) {
                cameraControls_->StartMouseDrag();
                cameraControls_->RotateZ(0, -2);
            }
            if (hasKey('r')) {
                cameraControls_->StartMouseDrag();
                cameraControls_->RotateZ(0, 2);
            }
            if (hasKey(KEY_UP)) {
                cameraControls_->RotateLocal(angleRad, {1, 0, 0});
            }
            if (hasKey(KEY_DOWN)) {
                cameraControls_->RotateLocal(-angleRad, {1, 0, 0});
            }
            if (hasKey(KEY_LEFT)) {
                cameraControls_->RotateLocal(angleRad, {0, 1, 0});
            }
            if (hasKey(KEY_RIGHT)) {
                cameraControls_->RotateLocal(-angleRad, {0, 1, 0});
            }
        }
    }

private:
    std::unique_ptr<visualization::CameraInteractor> cameraControls_;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged_;
    int lastMouseX_ = 0;
    int lastMouseY_ = 0;
    visualization::Camera::Transform _mouseDownRotation;
};

class RotationInteractor : public MouseInteractor {
public:
    void SetInteractor(visualization::RotationInteractor* r) {
        interactor_ = r;
    }

    visualization::MatrixInteractor& GetMatrixInteractor() override {
        return *interactor_;
    }

    void SetCenterOfRotation(const Eigen::Vector3f& center) {
        interactor_->SetCenterOfRotation(center);
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
                interactor_->StartMouseDrag();
                break;
            case MouseEvent::DRAG: {
                int dx = e.x - mouseDownX_;
                int dy = e.y - mouseDownY_;
                switch (state_) {
                    case State::NONE:
                        break;
                    case State::PAN:
                        interactor_->Pan(dx, dy);
                        break;
                    case State::DOLLY:
                        interactor_->Dolly(dy, visualization::MatrixInteractor::
                                                       DragType::MOUSE);
                        break;
                    case State::ROTATE_XY:
                        interactor_->Rotate(dx, dy);
                        break;
                    case State::ROTATE_Z:
                        interactor_->RotateZ(dx, dy);
                        break;
                }
                interactor_->UpdateMouseDragUI();
                break;
            }
            case MouseEvent::WHEEL: {
                interactor_->Dolly(2.0 * e.wheel.dy,
                                   e.wheel.isTrackpad
                                           ? visualization::MatrixInteractor::
                                                     DragType::TWO_FINGER
                                           : visualization::MatrixInteractor::
                                                     DragType::WHEEL);
                break;
            }
            case MouseEvent::BUTTON_UP:
                interactor_->EndMouseDrag();
                state_ = State::NONE;
                break;
            default:
                break;
        }
    }

    void Key(const KeyEvent& e) override {}

protected:
    visualization::RotationInteractor* interactor_ = nullptr;
    int mouseDownX_ = 0;
    int mouseDownY_ = 0;

    enum class State { NONE, PAN, DOLLY, ROTATE_XY, ROTATE_Z };
    State state_ = State::NONE;
};

class RotateModelInteractor : public RotationInteractor {
public:
    explicit RotateModelInteractor(visualization::Scene* scene,
                                   visualization::Camera* camera)
        : RotationInteractor(),
          rotation_(new visualization::ModelInteractor(
                  scene, camera, MIN_FAR_PLANE)) {
        SetInteractor(rotation_.get());
    }

    void SetModel(visualization::GeometryHandle axes,
                  const std::vector<visualization::GeometryHandle>& objects) {
        rotation_->SetModel(axes, objects);
    }

private:
    std::unique_ptr<visualization::ModelInteractor> rotation_;
    visualization::GeometryHandle axes_;
};

class RotateCameraInteractor : public RotationInteractor {
    using Super = RotationInteractor;

public:
    explicit RotateCameraInteractor(visualization::Camera* camera)
        : cameraControls_(std::make_unique<visualization::CameraInteractor>(
                  camera, MIN_FAR_PLANE)) {
        SetInteractor(cameraControls_.get());
    }

    void Mouse(const MouseEvent& e) override {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
            case MouseEvent::DRAG:
            case MouseEvent::BUTTON_UP:
            default:
                Super::Mouse(e);
                break;
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
                    Super::Mouse(e);
                }
                break;
            }
        }
    }

private:
    std::unique_ptr<visualization::CameraInteractor> cameraControls_;
};

// ----------------------------------------------------------------------------
class Interactors {
public:
    Interactors(visualization::Scene* scene, visualization::Camera* camera)
        : rotate_(std::make_unique<RotateCameraInteractor>(camera)),
          fps_(std::make_unique<FPSInteractor>(camera)),
          lightDir_(std::make_unique<RotateSunInteractor>(scene, camera)),
          ibl_(std::make_unique<RotateIBLInteractor>(scene, camera)),
          model_(std::make_unique<RotateModelInteractor>(scene, camera)) {
        current_ = rotate_.get();
    }

    void SetViewSize(const Size& size) {
        rotate_->GetMatrixInteractor().SetViewSize(size.width, size.height);
        fps_->GetMatrixInteractor().SetViewSize(size.width, size.height);
        lightDir_->GetMatrixInteractor().SetViewSize(size.width, size.height);
        ibl_->GetMatrixInteractor().SetViewSize(size.width, size.height);
        model_->GetMatrixInteractor().SetViewSize(size.width, size.height);
    }

    void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds) {
        rotate_->GetMatrixInteractor().SetBoundingBox(bounds);
        fps_->GetMatrixInteractor().SetBoundingBox(bounds);
        lightDir_->GetMatrixInteractor().SetBoundingBox(bounds);
        ibl_->GetMatrixInteractor().SetBoundingBox(bounds);
        model_->GetMatrixInteractor().SetBoundingBox(bounds);
    }

    void SetCenterOfRotation(const Eigen::Vector3f& center) {
        rotate_->SetCenterOfRotation(center);
    }

    void SetDirectionalLight(
            visualization::LightHandle dirLight,
            std::function<void(const Eigen::Vector3f&)> onChanged) {
        lightDir_->SetDirectionalLight(dirLight, onChanged);
    }

    void SetSkyboxHandle(visualization::SkyboxHandle skybox, bool isOn) {
        ibl_->SetSkyboxHandle(skybox, isOn);
    }

    void SetModel(visualization::GeometryHandle axes,
                  const std::vector<visualization::GeometryHandle>& objects) {
        model_->SetModel(axes, objects);
    }

    SceneWidget::Controls GetControls() const {
        if (current_ == fps_.get()) {
            return SceneWidget::Controls::FPS;
        } else if (current_ == lightDir_.get()) {
            return SceneWidget::Controls::ROTATE_SUN;
        } else if (current_ == ibl_.get()) {
            return SceneWidget::Controls::ROTATE_IBL;
        } else if (current_ == model_.get()) {
            return SceneWidget::Controls::ROTATE_MODEL;
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
                current_ = ibl_.get();
                break;
            case SceneWidget::Controls::ROTATE_MODEL:
                current_ = model_.get();
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
        return Widget::DrawResult::NONE;
    }

private:
    std::unique_ptr<RotateCameraInteractor> rotate_;
    std::unique_ptr<FPSInteractor> fps_;
    std::unique_ptr<RotateSunInteractor> lightDir_;
    std::unique_ptr<RotateIBLInteractor> ibl_;
    std::unique_ptr<RotateModelInteractor> model_;

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

void SceneWidget::SetSkyboxHandle(visualization::SkyboxHandle skybox,
                                  bool isOn) {
    impl_->controls->SetSkyboxHandle(skybox, isOn);
}

void SceneWidget::SetModel(
        visualization::GeometryHandle axes,
        const std::vector<visualization::GeometryHandle>& objects) {
    impl_->controls->SetModel(axes, objects);
}

void SceneWidget::SetViewControls(Controls mode) {
    if (mode == Controls::ROTATE_OBJ &&
        impl_->controls->GetControls() == Controls::FPS) {
        impl_->controls->SetControls(mode);
        // If we're going from FPS to standard rotate obj, we need to
        // adjust the center of rotation or it will jump to a different
        // matrix rather abruptly. The center of rotation is used for the
        // panning distance so that the cursor stays in roughly the same
        // position as the user moves the mouse. Use the distance to the
        // center of the model, which should be reasonable.
        Eigen::Vector3f toCenter = impl_->bounds.GetCenter().cast<float>() -
                                   impl_->camera->GetPosition();
        Eigen::Vector3f forward = impl_->camera->GetForwardVector();
        Eigen::Vector3f center =
                impl_->camera->GetPosition() + toCenter.norm() * forward;
        impl_->controls->SetCenterOfRotation(center);
    } else {
        impl_->controls->SetControls(mode);
    }
}

void SceneWidget::GoToCameraPreset(CameraPreset preset) {
    auto boundsMax = impl_->bounds.GetMaxBound();
    auto maxDim =
            std::max(boundsMax.x(), std::max(boundsMax.y(), boundsMax.z()));
    maxDim = 1.5f * maxDim;
    Eigen::Vector3f center = impl_->bounds.GetCenter().cast<float>();
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

Widget::EventResult SceneWidget::Mouse(const MouseEvent& e) {
    impl_->controls->Mouse(e);
    return Widget::EventResult::CONSUMED;
}

Widget::EventResult SceneWidget::Key(const KeyEvent& e) {
    impl_->controls->Key(e);
    return Widget::EventResult::CONSUMED;
}

Widget::DrawResult SceneWidget::Tick(const TickEvent& e) {
    return impl_->controls->Tick(e);
}

}  // namespace gui
}  // namespace open3d
