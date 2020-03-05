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
#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Visualization/Rendering/Camera.h"
#include "Open3D/Visualization/Rendering/Scene.h"
#include "Open3D/Visualization/Rendering/View.h"

#include <Eigen/Geometry>

#define ENABLE_PAN 1

namespace open3d {
namespace gui {

static const double NEAR_PLANE = 0.1;
static const double MIN_FAR_PLANE = 100.0;

// ----------------------------------------------------------------------------
class MatrixControl {
public:
    virtual ~MatrixControl() {}

    void SetViewSize(const Size& size) { viewSize_ = size; }

    virtual void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds) {
        modelSize_ = (bounds.GetMaxBound() - bounds.GetMinBound()).norm();
        modelBounds_ = bounds;
    }

    void SetMouseDownInfo(const visualization::Camera::Transform& matrix,
                          const Eigen::Vector3f& centerOfRotation) {
        matrix_ = matrix;
        centerOfRotation_ = centerOfRotation;

        matrixAtMouseDown_ = matrix;
        centerOfRotationAtMouseDown_ = centerOfRotation;
    }

    const visualization::Camera::Transform& GetMatrix() const {
        return matrix_;
    }

    virtual void Rotate(int dx, int dy) {
        auto matrix = matrixAtMouseDown_;  // copy
        Eigen::AngleAxisf rotMatrix(0, Eigen::Vector3f(1, 0, 0));

        // We want to rotate as if we were rotating an imaginary trackball
        // centered at the point of rotation. To do this we need an axis
        // of rotation and an angle about the axis. To find the axis, we
        // imagine that the viewing plane has been translated into the screen
        // so that it intersects the center of rotation. The axis we want
        // to rotate around is perpendicular to the vector defined by (dx, dy)
        // (assuming +x is right and +y is up). (Imagine the situation if the
        // mouse movement is (100, 0) or (0, 100).) Now it is easy to find
        // the perpendicular in 2D. Conveniently, (axis.x, axis.y, 0) is the
        // correct axis in camera-local coordinates. We can multiply by the
        // camera's rotation matrix to get the correct world vector.
        dy = -dy;  // up is negative, but the calculations are easiest to
                   // imagine up is positive.
        Eigen::Vector3f axis(-dy, dx, 0);  // rotate by 90 deg in 2D
        float theta =
                0.5 * M_PI * axis.norm() / (0.5f * float(viewSize_.height));
        axis = axis.normalized();

        axis = matrix.rotation() * axis;  // convert axis to world coords
        rotMatrix = rotMatrix * Eigen::AngleAxisf(-theta, axis);

        auto pos = matrix * Eigen::Vector3f(0, 0, 0);
        auto dist = (centerOfRotation_ - pos).norm();
        visualization::Camera::Transform m;
        m.fromPositionOrientationScale(centerOfRotation_,
                                       rotMatrix * matrix.rotation(),
                                       Eigen::Vector3f(1, 1, 1));
        m.translate(Eigen::Vector3f(0, 0, dist));

        matrix_ = m;
    }

    virtual void RotateWorld(int dx, int dy,
                             const Eigen::Vector3f& xAxis,
                             const Eigen::Vector3f& yAxis) {
        auto matrix = matrixAtMouseDown_;  // copy

        dy = -dy;  // up is negative, but the calculations are easiest to
                   // imagine up is positive.
        Eigen::Vector3f axis = dx * xAxis + dy * yAxis;
        float theta =
                0.5 * M_PI * axis.norm() / (0.5f * float(viewSize_.height));
        axis = axis.normalized();

        axis = matrix.rotation() * axis;  // convert axis to world coords
        auto rotMatrix = visualization::Camera::Transform::Identity() * Eigen::AngleAxisf(-theta, axis);

        auto pos = matrix * Eigen::Vector3f(0, 0, 0);
        auto dist = (centerOfRotation_ - pos).norm();
        visualization::Camera::Transform m;
        m.fromPositionOrientationScale(centerOfRotation_,
                                       rotMatrix * matrix.rotation(),
                                       Eigen::Vector3f(1, 1, 1));
        m.translate(Eigen::Vector3f(0, 0, dist));

        matrix_ = m;
    }

    virtual void RotateZ(int dx, int dy) {
        // RotateZ rotates around the axis normal to the screen. Since we
        // will be rotating using camera coordinates, we want to rotate
        // about (0, 0, 1).
        Eigen::Vector3f axis(0, 0, 1);
        // Moving half the height should rotate 360 deg (= 2 * PI).
        // This makes it easy to rotate enough without rotating too much.
        auto rad = 4.0 * M_PI * dy / viewSize_.height;

        auto matrix = matrixAtMouseDown_;  // copy
        matrix.rotate(Eigen::AngleAxisf(rad, axis));
        matrix_ = matrix;
    }

    enum class DragType { MOUSE, WHEEL, TWO_FINGER };

    virtual void Dolly(int dy, DragType dragType) {
        float dist = 0.0f;  // initialize to make GCC happy
        switch (dragType) {
            case DragType::MOUSE:
                // Zoom out is "push away" or up, is a negative value for
                // mousing
                dist = float(dy) * 0.0025f * modelSize_;
                break;
            case DragType::TWO_FINGER:
                // Zoom out is "push away" or up, is a positive value for
                // two-finger scrolling, so we need to invert dy.
                dist = float(-dy) * 0.005f * modelSize_;
                break;
            case DragType::WHEEL:  // actual mouse wheel, same as two-fingers
                dist = float(-dy) * 0.1f * modelSize_;
                break;
        }

        if (dragType == DragType::MOUSE) {
            Dolly(dist, matrixAtMouseDown_);  // copies the matrix
        } else {
            Dolly(dist, matrix_);
        }
    }

    // Note: we pass `matrix` by value because we want to copy it,
    //       as translate() will be modifying it.
    virtual void Dolly(float zDist, visualization::Camera::Transform matrix) {
        // Dolly is just moving the camera forward. Filament uses right as +x,
        // up as +y, and forward as -z (standard OpenGL coordinates). So to
        // move forward all we need to do is translate the camera matrix by
        // dist * (0, 0, -1). Note that translating by camera_->GetForwardVector
        // would be incorrect, since GetForwardVector() returns the forward
        // vector in world space, but the translation happens in camera space.)
        // Since we want trackpad down (negative) to go forward ("pulling" the
        // model toward the viewer) we need to negate dy.
        auto forward = Eigen::Vector3f(0, 0, -zDist);  // zDist * (0, 0, -1)
        matrix.translate(forward);
        matrix_ = matrix;
    }

private:
    visualization::Camera::Transform matrix_;

    visualization::Camera::Transform matrixAtMouseDown_;
    Eigen::Vector3f centerOfRotationAtMouseDown_;

protected:
    Size viewSize_;
    double modelSize_ = 100.0;
    geometry::AxisAlignedBoundingBox modelBounds_;
    Eigen::Vector3f centerOfRotation_;
};

class LightDirControl : public MatrixControl {
    using Super = MatrixControl;
public:
    LightDirControl(visualization::Camera *camera)
    : camera_(camera) {
    }

    void SetDirectionalLight(visualization::Scene *scene,
                             visualization::LightHandle dirLight) {
        scene_ = scene;
        dirLight_ = dirLight;
    }

    void Rotate(int dx, int dy) override {
        Eigen::Vector3f up = camera_->GetUpVector();
        Eigen::Vector3f right = -camera_->GetLeftVector();
        RotateWorld(-dx, -dy, up, right);
        UpdateMouseDragUI();
    }

    void StartMouseDrag() {
        lightDirAtMouseDown_ = scene_->GetLightDirection(dirLight_);
        auto identity = visualization::Camera::Transform::Identity();
        Super::SetMouseDownInfo(identity, {0.0f, 0.0f, 0.0f});

        for (auto &o : uiObjs_) {
            scene_->RemoveGeometry(o.handle);
        }

        auto dir = scene_->GetLightDirection(dirLight_);

        // TODO: it seems that SetEntityTransform moves an object to be
        //       centered about the origin before applying the transform.
        //       This appears to be a Filament behavior, which we work around
        //       here. This code needs to be changed once we fix that
        //       behavior.

        double sphereSize = 0.5 * modelSize_; // modelSize_ is a diameter
        auto sphereTris = geometry::TriangleMesh::CreateSphere(sphereSize, 20);
        auto sphere = geometry::LineSet::CreateFromTriangleMesh(*sphereTris);
        sphere->PaintUniformColor({0.0f, 0.0f, 1.0f});
        auto t0 = visualization::Camera::Transform::Identity();
        uiObjs_.push_back({scene_->AddGeometry(*sphere), t0});
        scene_->SetEntityTransform(uiObjs_[0].handle, t0);

        auto sunRadius = 0.05 * modelSize_;
        auto sun = geometry::TriangleMesh::CreateSphere(sunRadius, 20);
        sun->PaintUniformColor({1.0f, 0.5f, 0.0f});
        auto t1 = visualization::Camera::Transform::Identity();
        t1.translate(-sphereSize * dir);
        uiObjs_.push_back({scene_->AddGeometry(*sun), t1});
        scene_->SetEntityTransform(uiObjs_[1].handle, t1);

        const double arrowRadius = 0.075 * sunRadius;
        const double arrowLength = 0.333 * modelSize_;
        auto sunDir = CreateArrow(dir.cast<double>(), arrowRadius,
                                  arrowLength, 0.1 * arrowLength, 20);
        sunDir->PaintUniformColor({1.0f, 0.5f, 0.0f});
        auto t2 = visualization::Camera::Transform::Identity();
        t2.translate(-sphereSize * dir);
        t2.translate(0.5 * arrowLength * dir);
        uiObjs_.push_back({scene_->AddGeometry(*sunDir), t2});
        scene_->SetEntityTransform(uiObjs_[2].handle, t2);

        UpdateMouseDragUI();
    }

    void UpdateMouseDragUI() {
        // TODO: uncomment the two lines here when we setting a transform
        //       no longer moves an object to be centered about the origin.
        //auto modelCenter = modelBounds_.GetCenter().cast<float>();
        for (auto &o : uiObjs_) {
            visualization::Camera::Transform t = GetMatrix() * o.transform;
            //t.pretranslate(modelCenter);
            scene_->SetEntityTransform(o.handle, t);
        }
    }

    void EndMouseDrag() {
        for (auto &o : uiObjs_) {
            scene_->RemoveGeometry(o.handle);
        }
    }

    Eigen::Vector3f GetCurrentDirection() const {
        return GetMatrix() * lightDirAtMouseDown_;
    }

private:
    visualization::Scene *scene_;
    visualization::Camera *camera_;
    visualization::LightHandle dirLight_;
    Eigen::Vector3f lightDirAtMouseDown_;

    struct UIObj {
        visualization::GeometryHandle handle;
        visualization::Camera::Transform transform;
    };
    std::vector<UIObj> uiObjs_;

    std::shared_ptr<geometry::TriangleMesh> CreateArrow(const Eigen::Vector3d& dir,
                                                        double radius,
                                                        double length,
                                                        double headLength,
                                                        int nSegs = 20) {
        Eigen::Vector3d tmp(dir.y(), dir.z(), dir.x());
        Eigen::Vector3d u = dir.cross(tmp).normalized();
        Eigen::Vector3d v = dir.cross(u);

        Eigen::Vector3d start(0, 0, 0);
        Eigen::Vector3d headStart((length - headLength) * dir.x(),
                                  (length - headLength) * dir.y(),
                                  (length - headLength) * dir.z());
        Eigen::Vector3d end(length * dir.x(), length * dir.y(), length * dir.z());
        auto arrow = std::make_shared<geometry::TriangleMesh>();
        // Cylinder
        CreateCircle(start, u, v, radius, nSegs,
                     arrow->vertices_, arrow->vertex_normals_);
        int nVertsInCircle = nSegs + 1;
        CreateCircle(headStart, u, v, radius, nSegs,
                     arrow->vertices_, arrow->vertex_normals_);
        for (int i = 0;  i < nSegs;  ++i) {
            arrow->triangles_.push_back({i, i + 1, nVertsInCircle + i + 1});
            arrow->triangles_.push_back({nVertsInCircle + i + 1, nVertsInCircle + i, i});
        }

        // End of cone
        int startIdx = int(arrow->vertices_.size());
        CreateCircle(headStart, u, v, 2.0 * radius, nSegs,
                     arrow->vertices_, arrow->vertex_normals_);
        for (int i = startIdx;  i < int(arrow->vertices_.size());  ++i) {
            arrow->vertex_normals_.push_back(-dir);
        }
        int centerIdx = int(arrow->vertices_.size());
        arrow->vertices_.push_back(headStart);
        arrow->vertex_normals_.push_back(-dir);
        for (int i = 0;  i < nSegs;  ++i) {
            arrow->triangles_.push_back({startIdx + i,
                                         startIdx + i + 1,
                                         centerIdx});
        }

        // Cone
        startIdx = int(arrow->vertices_.size());
        CreateCircle(headStart, u, v, 2.0 * radius, nSegs,
                     arrow->vertices_, arrow->vertex_normals_);
        for (int i = 0;  i < nSegs;  ++i) {
            int pointIdx = int(arrow->vertices_.size());
            arrow->vertices_.push_back(end);
            arrow->vertex_normals_.push_back(arrow->vertex_normals_[startIdx + i]);
            arrow->triangles_.push_back({startIdx + i,
                                         startIdx + i + 1,
                                         pointIdx});
        }

        return arrow;
    }

    void CreateCircle(const Eigen::Vector3d& center,
                      const Eigen::Vector3d& u, const Eigen::Vector3d& v,
                      double radius, int nSegs,
                      std::vector<Eigen::Vector3d>& points,
                      std::vector<Eigen::Vector3d>& normals) {
        for (int i = 0;  i <= nSegs;  ++i) {
            double theta = 2.0 * M_PI * double(i) / double(nSegs);
            double cosT = std::cos(theta);
            double sinT = std::sin(theta);
            Eigen::Vector3d p = center + radius * cosT * u + radius * sinT * v;
            Eigen::Vector3d n = (cosT * u + sinT * v).normalized();
            points.push_back(p);
            normals.push_back(n);
        }
    }
};

class CameraControls : public MatrixControl {
    using Super = MatrixControl;
public:
    CameraControls(visualization::Camera* c) : camera_(c) {}

    void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds) override{
        Super::SetBoundingBox(bounds);
        // Initialize parent's matrix_ (in case we do a mouse wheel, which
        // doesn't involve a mouse down) and the center of rotation.
        SetMouseDownInfo(camera_->GetModelMatrix(),
                         bounds.GetCenter().cast<float>());
    }

    void Rotate(int dx, int dy) override {
        Super::Rotate(dx, dy);
        camera_->SetModelMatrix(GetMatrix());
    }

    void RotateZ(int dx, int dy) override {
        Super::RotateZ(dx, dy);
        camera_->SetModelMatrix(GetMatrix());
    }

    void Pan(int dx, int dy) {
        // Calculate the depth to the pixel we clicked on, so that we
        // can compensate for perspective and have the mouse stays on
        // that location. Unfortunately, we don't really have access to
        // the depth buffer with Filament, so we'll fake it by finding
        // the depth of the center of rotation.
        auto pos = camera_->GetPosition();
        auto forward = camera_->GetForwardVector();
        float near = camera_->GetNear();
        float dist = forward.dot(centerOfRotationAtMouseDown_ - pos);
        dist = std::max(near, dist);

        // How far is one pixel?
        auto modelMatrix = matrixAtMouseDown_;  // copy
        float halfFoV = camera_->GetFieldOfView() / 2.0;
        float halfFoVRadians = halfFoV * M_PI / 180.0;
        float unitsAtDist = 2.0f * std::tan(halfFoVRadians) * (near + dist);
        float unitsPerPx = unitsAtDist / float(viewSize_.height);

        // Move camera and center of rotation. Adjust values from the
        // original positions at mousedown to avoid hysteresis problems.
        auto localMove = Eigen::Vector3f(-dx * unitsPerPx, dy * unitsPerPx, 0);
        auto worldMove = modelMatrix.rotation() * localMove;
        centerOfRotation_ = centerOfRotationAtMouseDown_ + worldMove;
        modelMatrix.translate(localMove);
        camera_->SetModelMatrix(modelMatrix);
    }


    void Dolly(int dy, DragType type) override {
        // Parent's matrix_ may not have been set yet
        if (type != DragType::MOUSE) {
            SetMouseDownInfo(camera_->GetModelMatrix(), centerOfRotation_);
        }
        Super::Dolly(dy, type);
    }

    void Dolly(float zDist, visualization::Camera::Transform matrixIn) override {
        Super::Dolly(zDist, matrixIn);
        auto matrix = GetMatrix();
        camera_->SetModelMatrix(matrix);

        // Update the far plane so that we don't get clipped by it as we dolly
        // out or lose precision as we dolly in.
        auto pos = matrix.translation().cast<double>();
        auto far1 = (modelBounds_.GetMinBound() - pos).norm();
        auto far2 = (modelBounds_.GetMaxBound() - pos).norm();
        auto modelSize = 2.0 * modelBounds_.GetExtent().norm();
        auto far = std::max(MIN_FAR_PLANE, std::max(far1, far2) + modelSize);
        float aspect = 1.0f;
        if (viewSize_.height > 0) {
            aspect = float(viewSize_.width) / float(viewSize_.height);
        }
        camera_->SetProjection(camera_->GetFieldOfView(), aspect,
                               camera_->GetNear(), far,
                               camera_->GetFieldOfViewType());
    }

    void Zoom(int dy, DragType dragType) {
        float dFOV = 0.0f;  // initialize to make GCC happy
        switch (dragType) {
            case DragType::MOUSE:
                dFOV = float(-dy) * 0.1;  // deg
                break;
            case DragType::TWO_FINGER:
                dFOV = float(dy) * 0.2f;  // deg
                break;
            case DragType::WHEEL:  // actual mouse wheel, same as two-fingers
                dFOV = float(dy) * 2.0f;  // deg
                break;
        }
        float oldFOV = 0.0;
        if (dragType == DragType::MOUSE) {
            oldFOV = fovAtMouseDown_;
        } else {
            oldFOV = camera_->GetFieldOfView();
        }
        float newFOV = oldFOV + dFOV;
        newFOV = std::max(5.0f, newFOV);
        newFOV = std::min(90.0f, newFOV);

        float toRadians = M_PI / 180.0;
        float near = camera_->GetNear();
        Eigen::Vector3f cameraPos, COR;
        if (dragType == DragType::MOUSE) {
            cameraPos = matrixAtMouseDown_.translation();
            COR = centerOfRotationAtMouseDown_;
        } else {
            cameraPos = camera_->GetPosition();
            COR = centerOfRotation_;
        }
        Eigen::Vector3f toCOR = COR - cameraPos;
        float oldDistFromPlaneToCOR = toCOR.norm() - near;
        float newDistFromPlaneToCOR =
                (near + oldDistFromPlaneToCOR) *
                        std::tan(oldFOV / 2.0 * toRadians) /
                        std::tan(newFOV / 2.0 * toRadians) -
                near;
        if (dragType == DragType::MOUSE) {
            Dolly(-(newDistFromPlaneToCOR - oldDistFromPlaneToCOR),
                  matrixAtMouseDown_);
        } else {
            Dolly(-(newDistFromPlaneToCOR - oldDistFromPlaneToCOR),
                  camera_->GetModelMatrix());
        }

        float aspect = 1.0f;
        if (viewSize_.height > 0) {
            aspect = float(viewSize_.width) / float(viewSize_.height);
        }
        camera_->SetProjection(newFOV, aspect, camera_->GetNear(),
                               camera_->GetFar(),
                               camera_->GetFieldOfViewType());
    }

    void GoToPreset(SceneWidget::CameraPreset preset) {
        auto boundsMax = modelBounds_.GetMaxBound();
        auto maxDim =
                std::max(boundsMax.x(), std::max(boundsMax.y(), boundsMax.z()));
        maxDim = 1.5f * maxDim;
        auto center = centerOfRotation_;
        Eigen::Vector3f eye, up;
        switch (preset) {
            case SceneWidget::CameraPreset::PLUS_X: {
                eye = Eigen::Vector3f(maxDim, center.y(), center.z());
                up = Eigen::Vector3f(0, 1, 0);
                break;
            }
            case SceneWidget::CameraPreset::PLUS_Y: {
                eye = Eigen::Vector3f(center.x(), maxDim, center.z());
                up = Eigen::Vector3f(1, 0, 0);
                break;
            }
            case SceneWidget::CameraPreset::PLUS_Z: {
                eye = Eigen::Vector3f(center.x(), center.y(), maxDim);
                up = Eigen::Vector3f(0, 1, 0);
                break;
            }
        }
        camera_->LookAt(center, eye, up);
    }

    void StartMouseDrag() {
        matrixAtMouseDown_ = camera_->GetModelMatrix();
        centerOfRotationAtMouseDown_ = centerOfRotation_;
        fovAtMouseDown_ = camera_->GetFieldOfView();

        Super::SetMouseDownInfo(matrixAtMouseDown_, centerOfRotation_);
    }

    void UpdateMouseDragUI() {}

    void EndMouseDrag() {}

private:
    visualization::Camera* camera_;

    visualization::Camera::Transform matrixAtMouseDown_;
    Eigen::Vector3f centerOfRotationAtMouseDown_;
    double fovAtMouseDown_;
};

class MouseInteractor {
public:
    MouseInteractor(visualization::Camera* camera)
    : cameraControls_(std::make_unique<CameraControls>(camera))
    , lightDir_(std::make_unique<LightDirControl>(camera)) {
    }

    void SetViewSize(const Size& size) {
        cameraControls_->SetViewSize(size);
        lightDir_->SetViewSize(size);
    }

    void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds) {
        cameraControls_->SetBoundingBox(bounds);
        lightDir_->SetBoundingBox(bounds);
    }

    void SetDirectionalLight(
                    visualization::Scene *scene,
                    visualization::LightHandle dirLight,
                    std::function<void(const Eigen::Vector3f&)> onChanged) {
        lightDir_->SetDirectionalLight(scene, dirLight);
        onLightDirChanged_ = onChanged;
    }

    void GoToCameraPreset(SceneWidget::CameraPreset preset) {
        cameraControls_->GoToPreset(preset);
    }

    void Mouse(const MouseEvent& e) {
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
                    } else if (e.modifiers & int(KeyModifier::ALT)) {
                        state_ = State::ROTATE_LIGHT;
                    } else {
                        state_ = State::ROTATE_XY;
                    }
#if ENABLE_PAN
                } else if (e.button.button == MouseButton::RIGHT) {
                    state_ = State::PAN;
#endif  // ENABLE_PAN
                }
                if (state_ == State::ROTATE_LIGHT) {
                    lightDir_->StartMouseDrag();
                } else if (state_ != State::NONE) {
                    cameraControls_->StartMouseDrag();
                }
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
                        cameraControls_->Dolly(dy, MatrixControl::DragType::MOUSE);
                        break;
                    case State::ZOOM:
                        cameraControls_->Zoom(dy, MatrixControl::DragType::MOUSE);
                        break;
                    case State::ROTATE_XY:
                        cameraControls_->Rotate(dx, dy);
                        break;
                    case State::ROTATE_Z:
                        cameraControls_->RotateZ(dx, dy);
                        break;
                    case State::ROTATE_LIGHT:
                        lightDir_->Rotate(dx, dy);
                        if (onLightDirChanged_) {
                            onLightDirChanged_(lightDir_->GetCurrentDirection());
                        }
                        break;
                }
                break;
            }
            case MouseEvent::WHEEL: {
                if (e.modifiers & int(KeyModifier::SHIFT)) {
                    cameraControls_->Zoom(e.wheel.dy,
                                          e.wheel.isTrackpad ? MatrixControl::DragType::TWO_FINGER
                                                             : MatrixControl::DragType::WHEEL);
                } else {
                    cameraControls_->Dolly(e.wheel.dy,
                                           e.wheel.isTrackpad ? MatrixControl::DragType::TWO_FINGER
                                                              : MatrixControl::DragType::WHEEL);
                }
                break;
            }
            case MouseEvent::BUTTON_UP:
                cameraControls_->EndMouseDrag();
                lightDir_->EndMouseDrag();
                state_ = State::NONE;
                break;
            default:
                break;
        }
    }

private:
    std::unique_ptr<CameraControls> cameraControls_;
    std::unique_ptr<LightDirControl> lightDir_;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged_;

    int mouseDownX_;
    int mouseDownY_;

    enum class State { NONE, PAN, DOLLY, ZOOM, ROTATE_XY, ROTATE_Z, ROTATE_LIGHT };
    State state_ = State::NONE;
};
// ----------------------------------------------------------------------------
struct SceneWidget::Impl {
    visualization::Scene& scene;
    visualization::ViewHandle viewId;
//    std::shared_ptr<CameraControls> controls;
    std::shared_ptr<MouseInteractor> controls;
    bool frameChanged = false;
    visualization::LightHandle dirLight;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged;

    explicit Impl(visualization::Scene& aScene) : scene(aScene) {}
};

SceneWidget::SceneWidget(visualization::Scene& scene) : impl_(new Impl(scene)) {
    impl_->viewId = scene.AddView(0, 0, 1, 1);

    auto view = impl_->scene.GetView(impl_->viewId);
    impl_->controls = std::make_shared<MouseInteractor>(view->GetCamera());
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

void SceneWidget::SetDirectionalLight(
                visualization::Scene *scene,
                visualization::LightHandle dirLight,
                std::function<void(const Eigen::Vector3f&)> onDirChanged) {
    impl_->dirLight = dirLight;
    impl_->onLightDirChanged = onDirChanged;
    impl_->controls->SetDirectionalLight(scene, dirLight,
                        [this, scene, dirLight](const Eigen::Vector3f& dir) {
        scene->SetLightDirection(dirLight, dir);
        if (impl_->onLightDirChanged) {
            impl_->onLightDirChanged(dir);
        }
    });
}

void SceneWidget::GoToCameraPreset(CameraPreset preset) {
    impl_->controls->GoToCameraPreset(preset);
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

}  // namespace gui
}  // namespace open3d
