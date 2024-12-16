// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/Camera.h"

#include "open3d/geometry/BoundingVolume.h"

namespace open3d {
namespace visualization {
namespace rendering {

static const double NEAR_PLANE = 0.1;
static const double MIN_FAR_PLANE = 1.0;

void Camera::FromExtrinsics(const Eigen::Matrix4d& extrinsic) {
    // The intrinsic * extrinsic matrix models projection from the world through
    // a pinhole onto the projection plane. The intrinsic matrix is the
    // projection matrix, and extrinsic.inverse() is the camera pose. But the
    // OpenGL camera has the projection plane in front of the camera, which
    // essentially inverts all the axes of the projection. (Pinhole camera
    // images are flipped horizontally and vertically and the camera is the
    // other direction.) But the extrinsic matrix is left-handed, so we also
    // need to convert to OpenGL's right-handed matrices.
    Eigen::Matrix4d toGLCamera;
    toGLCamera << 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix4f m = (extrinsic.inverse() * toGLCamera).cast<float>();
    SetModelMatrix(Camera::Transform(m));
}

void Camera::SetupCameraAsPinholeCamera(
        rendering::Camera& camera,
        const Eigen::Matrix3d& intrinsic,
        const Eigen::Matrix4d& extrinsic,
        int intrinsic_width_px,
        int intrinsic_height_px,
        const geometry::AxisAlignedBoundingBox& scene_bounds) {
    camera.FromExtrinsics(extrinsic);
    camera.SetProjection(intrinsic, NEAR_PLANE,
                         CalcFarPlane(camera, scene_bounds), intrinsic_width_px,
                         intrinsic_height_px);
}

float Camera::CalcNearPlane() { return NEAR_PLANE; }

float Camera::CalcFarPlane(
        const rendering::Camera& camera,
        const geometry::AxisAlignedBoundingBox& scene_bounds) {
    // The far plane needs to be the max absolute distance, not just the
    // max extent, so that axes are visible if requested.
    // See also RotationInteractorLogic::UpdateCameraFarPlane().
    Eigen::Vector3d cam_xlate =
            camera.GetModelMatrix().translation().cast<double>();
    auto far1 = (scene_bounds.GetCenter() - cam_xlate).norm() * 2.5;
    auto far2 = cam_xlate.norm();
    auto model_size = 4.0 * scene_bounds.GetExtent().norm();
    auto far = std::max(MIN_FAR_PLANE, std::max(far1, far2) + model_size);
    return far;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
