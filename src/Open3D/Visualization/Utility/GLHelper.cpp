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

#include "Open3D/Visualization/Utility/GLHelper.h"

#include <Eigen/Dense>
#include <cmath>

namespace open3d {
namespace visualization {
namespace GLHelper {

GLMatrix4f LookAt(const Eigen::Vector3d &eye,
                  const Eigen::Vector3d &lookat,
                  const Eigen::Vector3d &up) {
    Eigen::Vector3d front_dir = (eye - lookat).normalized();
    Eigen::Vector3d up_dir = up.normalized();
    Eigen::Vector3d right_dir = up_dir.cross(front_dir).normalized();
    up_dir = front_dir.cross(right_dir).normalized();

    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    mat.block<1, 3>(0, 0) = right_dir.transpose();
    mat.block<1, 3>(1, 0) = up_dir.transpose();
    mat.block<1, 3>(2, 0) = front_dir.transpose();
    mat(0, 3) = -right_dir.dot(eye);
    mat(1, 3) = -up_dir.dot(eye);
    mat(2, 3) = -front_dir.dot(eye);
    mat(3, 3) = 1.0;
    return mat.cast<GLfloat>();
}

GLMatrix4f Perspective(double field_of_view_,
                       double aspect,
                       double z_near,
                       double z_far) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    double fov_rad = field_of_view_ / 180.0 * M_PI;
    double tan_half_fov = std::tan(fov_rad / 2.0);
    mat(0, 0) = 1.0 / aspect / tan_half_fov;
    mat(1, 1) = 1.0 / tan_half_fov;
    mat(2, 2) = -(z_far + z_near) / (z_far - z_near);
    mat(3, 2) = -1.0;
    mat(2, 3) = -2.0 * z_far * z_near / (z_far - z_near);
    return mat.cast<GLfloat>();
}

GLMatrix4f Ortho(double left,
                 double right,
                 double bottom,
                 double top,
                 double z_near,
                 double z_far) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    mat(0, 0) = 2.0 / (right - left);
    mat(1, 1) = 2.0 / (top - bottom);
    mat(2, 2) = -2.0 / (z_far - z_near);
    mat(0, 3) = -(right + left) / (right - left);
    mat(1, 3) = -(top + bottom) / (top - bottom);
    mat(2, 3) = -(z_far + z_near) / (z_far - z_near);
    mat(3, 3) = 1.0;
    return mat.cast<GLfloat>();
}

Eigen::Vector3d Project(const Eigen::Vector3d &point,
                        const GLMatrix4f &mvp_matrix,
                        const int width,
                        const int height) {
    Eigen::Vector4d pos = mvp_matrix.cast<double>() *
                          Eigen::Vector4d(point(0), point(1), point(2), 1.0);
    if (pos(3) == 0.0) {
        return Eigen::Vector3d::Zero();
    }
    pos /= pos(3);
    return Eigen::Vector3d((pos(0) * 0.5 + 0.5) * (double)width,
                           (pos(1) * 0.5 + 0.5) * (double)height,
                           (1.0 + pos(2)) * 0.5);
}

Eigen::Vector3d Unproject(const Eigen::Vector3d &screen_point,
                          const GLMatrix4f &mvp_matrix,
                          const int width,
                          const int height) {
    Eigen::Vector4d point =
            mvp_matrix.cast<double>().inverse() *
            Eigen::Vector4d(screen_point(0) / (double)width * 2.0 - 1.0,
                            screen_point(1) / (double)height * 2.0 - 1.0,
                            screen_point(2) * 2.0 - 1.0, 1.0);
    if (point(3) == 0.0) {
        return Eigen::Vector3d::Zero();
    }
    point /= point(3);
    return point.block<3, 1>(0, 0);
}

int ColorCodeToPickIndex(const Eigen::Vector4i &color) {
    if (color(0) == 255) {
        return -1;
    } else {
        return ((color(0) * 256 + color(1)) * 256 + color(2)) * 256 + color(3);
    }
}

}  // namespace GLHelper
}  // namespace visualization
}  // namespace open3d
