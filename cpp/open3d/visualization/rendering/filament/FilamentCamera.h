// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <utils/Entity.h>

#include "open3d/visualization/rendering/Camera.h"

/// @cond
namespace filament {
class Camera;
class Engine;
}  // namespace filament
/// @endcond

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentCamera : public Camera {
public:
    explicit FilamentCamera(filament::Engine& engine);
    ~FilamentCamera();

    void SetProjection(double fov,
                       double aspect,
                       double near,
                       double far,
                       FovType fov_type) override;

    void SetProjection(Projection projection,
                       double left,
                       double right,
                       double bottom,
                       double top,
                       double near,
                       double far) override;

    void SetProjection(const Eigen::Matrix3d& intrinsics,
                       double near,
                       double far,
                       double width,
                       double height) override;

    void LookAt(const Eigen::Vector3f& center,
                const Eigen::Vector3f& eye,
                const Eigen::Vector3f& up) override;

    void SetModelMatrix(const Transform& view) override;
    void SetModelMatrix(const Eigen::Vector3f& forward,
                        const Eigen::Vector3f& left,
                        const Eigen::Vector3f& up) override;

    double GetNear() const override;
    double GetFar() const override;
    /// only valid if fov was passed to SetProjection()
    double GetFieldOfView() const override;
    /// only valid if fov was passed to SetProjection()
    FovType GetFieldOfViewType() const override;

    Eigen::Vector3f GetPosition() const override;
    Eigen::Vector3f GetForwardVector() const override;
    Eigen::Vector3f GetLeftVector() const override;
    Eigen::Vector3f GetUpVector() const override;
    Transform GetModelMatrix() const override;
    Transform GetViewMatrix() const override;
    ProjectionMatrix GetProjectionMatrix() const override;
    Transform GetCullingProjectionMatrix() const override;
    const ProjectionInfo& GetProjection() const override;

    Eigen::Vector3f Unproject(float x,
                              float y,
                              float z,
                              float view_width,
                              float view_height) const override;

    Eigen::Vector2f GetNDC(const Eigen::Vector3f& pt) const override;
    double GetViewZ(float z_buffer) const override;

    void CopyFrom(const Camera* camera) override;

    filament::Camera* GetNativeCamera() const { return camera_; }

private:
    filament::Camera* camera_ = nullptr;
    utils::Entity camera_entity_;
    filament::Engine& engine_;
    Camera::ProjectionInfo projection_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
