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

    double GetNear() const override;
    double GetFar() const override;
    /// only valid if fov was passed to SetProjection()
    double GetFieldOfView() const override;
    /// only valid if fov was passed to SetProjection()
    FovType GetFieldOfViewType() const override;

    void SetModelMatrix(const Transform& view) override;
    void SetModelMatrix(const Eigen::Vector3f& forward,
                        const Eigen::Vector3f& left,
                        const Eigen::Vector3f& up) override;

    void LookAt(const Eigen::Vector3f& center,
                const Eigen::Vector3f& eye,
                const Eigen::Vector3f& up) override;

    Eigen::Vector3f GetPosition() const override;
    Eigen::Vector3f GetForwardVector() const override;
    Eigen::Vector3f GetLeftVector() const override;
    Eigen::Vector3f GetUpVector() const override;
    Transform GetModelMatrix() const override;
    Transform GetViewMatrix() const override;
    Transform GetProjectionMatrix() const override;

    filament::Camera* GetNativeCamera() const { return camera_; }

private:
    filament::Camera* camera_ = nullptr;
    filament::Engine& engine_;
    double fov_;
    FovType fov_type_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
