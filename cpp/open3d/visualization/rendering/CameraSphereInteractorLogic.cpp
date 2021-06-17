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

#include "open3d/visualization/rendering/CameraSphereInteractorLogic.h"

namespace open3d {
namespace visualization {
namespace rendering {

CameraSphereInteractorLogic::CameraSphereInteractorLogic(Camera* c,
                                                         double min_far_plane)
    : CameraInteractorLogic(c, min_far_plane) {}

void CameraSphereInteractorLogic::Rotate(int dx, int dy) {
    float phi = phi_at_mousedown_ - float(M_PI) * dx / float(view_width_);
    float theta = theta_at_mousedown_ + float(M_PI) * dy / float(view_height_);

    float dist = r_at_mousedown_;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    float sin_phi = std::sin(phi);
    float cos_phi = std::cos(phi);
    Eigen::Vector3f eye(center_of_rotation_.x() + dist * cos_phi * cos_theta,
                        center_of_rotation_.y() + dist * sin_phi * cos_theta,
                        center_of_rotation_.z() + dist * sin_theta);
    Eigen::Vector3f up(-cos_phi * sin_theta, -sin_phi * sin_theta, cos_theta);
    Eigen::Vector3f forward = (center_of_rotation_ - eye).normalized();
    Eigen::Vector3f left = up.cross(forward);
    Camera::Transform::MatrixType mm;
    mm << -left.x(), up.x(), -forward.x(), eye.x(), -left.y(), up.y(),
            -forward.y(), eye.y(), -left.z(), up.z(), -forward.z(), eye.z(),
            0.0f, 0.0f, 0.0f, 1.0f;
    Camera::Transform m = Camera::Transform(mm);
    SetMatrix(m);
    camera_->SetModelMatrix(m);
}

void CameraSphereInteractorLogic::StartMouseDrag() {
    Super::StartMouseDrag();
    auto m = camera_->GetModelMatrix();
    Eigen::Vector3f eye(m(0, 3), m(1, 3), m(2, 3));
    Eigen::Vector3f up(m(0, 1), m(1, 1), m(2, 1));
    r_at_mousedown_ = (eye - center_of_rotation_).norm();
    eye = (eye - center_of_rotation_) / r_at_mousedown_;
    // Clamp coords to [-1, 1], since floating point error can result it being
    // a little outside those bounds.
    float eye_x = std::min(1.0f, std::max(-1.0f, eye.x()));
    float eye_y = std::min(1.0f, std::max(-1.0f, eye.y()));
    float eye_z = std::min(1.0f, std::max(-1.0f, eye.z()));
    float up_z = std::min(1.0f, std::max(-1.0f, up.z()));
    theta_at_mousedown_ = std::asin(eye_z);
    if (std::abs(eye_y) < 0.001 && std::abs(eye_x) < 0.001) {
        float up_x = std::min(1.0f, std::max(-1.0f, up.x()));
        float up_y = std::min(1.0f, std::max(-1.0f, up.y()));
        phi_at_mousedown_ = std::atan2(-up_y, -up_x);
    } else {
        phi_at_mousedown_ = std::atan2(eye_y, eye_x);
    }
    if (up_z < 0.0f) {
        if (theta_at_mousedown_ > 0) {
            theta_at_mousedown_ = M_PI - theta_at_mousedown_;
        } else {
            theta_at_mousedown_ = -M_PI - theta_at_mousedown_;
        }
        phi_at_mousedown_ += float(M_PI);
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
