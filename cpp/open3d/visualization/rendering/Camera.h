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

#pragma once

#include <Eigen/Geometry>

namespace open3d {

namespace geometry {
class AxisAlignedBoundingBox;
}  // namespace geometry

namespace visualization {
namespace rendering {

class Camera {
public:
    enum class FovType { Vertical, Horizontal };
    enum class Projection { Perspective, Ortho };
    using Transform = Eigen::Transform<float, 3, Eigen::Affine>;
    using ProjectionMatrix = Eigen::Transform<float, 3, Eigen::Projective>;

    virtual ~Camera() = default;

    virtual void SetProjection(double fov,
                               double aspect,
                               double near,
                               double far,
                               FovType fov_type) = 0;

    /** Sets the projection matrix from a frustum defined by six planes.
     *
     * @param projection    type of #Projection to use.
     *
     * @param left      distance in world units from the camera to the left
     * plane, at the near plane.
     *
     * @param right     distance in world units from the camera to the right
     * plane, at the near plane.
     *
     * @param bottom    distance in world units from the camera to the bottom
     * plane, at the near plane.
     *
     * @param top       distance in world units from the camera to the top
     * plane, at the near plane.
     *
     * @param near      distance in world units from the camera to the near
     * plane. The near plane's
     *
     * @param far       distance in world units from the camera to the far
     * plane. The far plane's
     */
    virtual void SetProjection(Projection projection,
                               double left,
                               double right,
                               double bottom,
                               double top,
                               double near,
                               double far) = 0;

    virtual void SetProjection(const Eigen::Matrix3d& intrinsics,
                               double near,
                               double far,
                               double width,
                               double height) = 0;

    virtual void LookAt(const Eigen::Vector3f& center,
                        const Eigen::Vector3f& eye,
                        const Eigen::Vector3f& up) = 0;
    virtual void FromExtrinsics(const Eigen::Matrix4d& extrinsics);

    virtual void SetModelMatrix(const Transform& view) = 0;
    virtual void SetModelMatrix(const Eigen::Vector3f& forward,
                                const Eigen::Vector3f& left,
                                const Eigen::Vector3f& up) = 0;

    virtual double GetNear() const = 0;
    virtual double GetFar() const = 0;
    /// only valid if fov was passed to SetProjection()
    virtual double GetFieldOfView() const = 0;
    /// only valid if fov was passed to SetProjection()
    virtual FovType GetFieldOfViewType() const = 0;

    virtual Eigen::Vector3f GetPosition() const = 0;
    virtual Eigen::Vector3f GetForwardVector() const = 0;
    virtual Eigen::Vector3f GetLeftVector() const = 0;
    virtual Eigen::Vector3f GetUpVector() const = 0;
    virtual Transform GetModelMatrix() const = 0;
    virtual Transform GetViewMatrix() const = 0;
    virtual ProjectionMatrix GetProjectionMatrix() const = 0;
    virtual Transform GetCullingProjectionMatrix() const = 0;

    /// Returns world space coordinates given an x,y position in screen
    /// coordinates relative to upper left, the screen dimensions, and z is the
    /// depth value (0.0 - 1.0)
    virtual Eigen::Vector3f Unproject(float x,
                                      float y,
                                      float z,
                                      float view_width,
                                      float view_height) const = 0;

    // Returns the normalized device coordinates (NDC) of the specified point
    // given the view and projection matrices of the camera. The returned point
    // is in the range [-1, 1] if the point is in view, or outside the range if
    // the point is out of view.
    virtual Eigen::Vector2f GetNDC(const Eigen::Vector3f& pt) const = 0;

    /// Returns the view space depth (i.e., distance from camera) for the given
    /// Z-buffer value
    virtual double GetViewZ(float z_buffer) const = 0;

    struct ProjectionInfo {
        bool is_ortho;
        bool is_intrinsic;
        union {
            struct {
                Projection projection;
                double left;
                double right;
                double bottom;
                double top;
                double near_plane;  // Windows #defines "near"
                double far_plane;   // Windows #defines "far"
            } ortho;
            struct {
                FovType fov_type;
                double fov;
                double aspect;
                double near_plane;
                double far_plane;
            } perspective;
            struct {
                double fx;
                double fy;
                double cx;
                double cy;
                double near_plane;
                double far_plane;
                double width;
                double height;
            } intrinsics;
        } proj;
    };
    virtual const ProjectionInfo& GetProjection() const = 0;

    virtual void CopyFrom(const Camera* camera) = 0;

    /// Convenience function for configuring a camera as a pinhole camera.
    /// Configures the projection using the intrinsics and bounds,
    /// and the model matrix using the extrinsic matrix. Equivalent to calling
    /// SetProjection() and FromExtrinsics().
    static void SetupCameraAsPinholeCamera(
            rendering::Camera& camera,
            const Eigen::Matrix3d& intrinsic,
            const Eigen::Matrix4d& extrinsic,
            int intrinsic_width_px,
            int intrinsic_height_px,
            const geometry::AxisAlignedBoundingBox& scene_bounds);

    /// Returns a good value for the near plane.
    static float CalcNearPlane();

    /// Returns a value for the far plane that ensures that the entire bounds
    /// provided will not be clipped.
    static float CalcFarPlane(
            const rendering::Camera& camera,
            const geometry::AxisAlignedBoundingBox& scene_bounds);
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
