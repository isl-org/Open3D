// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
    auto cam_xlate = camera.GetModelMatrix().translation().cast<double>();
    auto far1 = (scene_bounds.GetCenter() - cam_xlate).norm() * 2.5;
    auto far2 = cam_xlate.norm();
    auto model_size = 4.0 * scene_bounds.GetExtent().norm();
    auto far = std::max(MIN_FAR_PLANE, std::max(far1, far2) + model_size);
    return far;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
