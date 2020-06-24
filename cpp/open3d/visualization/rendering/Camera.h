// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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
namespace visualization {
namespace rendering {

class Camera {
public:
    enum class FovType { Vertical, Horizontal };
    enum class Projection { Perspective, Ortho };
    using Transform = Eigen::Transform<float, 3, Eigen::Affine>;

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

    virtual double GetNear() const = 0;
    virtual double GetFar() const = 0;
    /// only valid if fov was passed to SetProjection()
    virtual double GetFieldOfView() const = 0;
    /// only valid if fov was passed to SetProjection()
    virtual FovType GetFieldOfViewType() const = 0;

    virtual void SetModelMatrix(const Transform& view) = 0;
    virtual void SetModelMatrix(const Eigen::Vector3f& forward,
                                const Eigen::Vector3f& left,
                                const Eigen::Vector3f& up) = 0;

    virtual void LookAt(const Eigen::Vector3f& center,
                        const Eigen::Vector3f& eye,
                        const Eigen::Vector3f& up) = 0;

    virtual Eigen::Vector3f GetPosition() const = 0;
    virtual Eigen::Vector3f GetForwardVector() const = 0;
    virtual Eigen::Vector3f GetLeftVector() const = 0;
    virtual Eigen::Vector3f GetUpVector() const = 0;
    virtual Transform GetModelMatrix() const = 0;
    virtual Transform GetViewMatrix() const = 0;
    virtual Transform GetProjectionMatrix() const = 0;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
