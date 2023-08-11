// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Avoid warning caused by redefinition of APIENTRY macro
// defined also in glfw3.h
#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>  // Make sure glew.h is included before gl.h
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <string>
#include <unordered_map>

namespace open3d {
namespace visualization {
namespace gl_util {

const static std::unordered_map<int, GLenum> texture_format_map_ = {
        {1, GL_RED}, {3, GL_RGB}, {4, GL_RGBA}};
const static std::unordered_map<int, GLenum> texture_type_map_ = {
        {1, GL_UNSIGNED_BYTE}, {2, GL_UNSIGNED_SHORT}, {4, GL_FLOAT}};

typedef Eigen::Matrix<GLfloat, 3, 1, Eigen::ColMajor> GLVector3f;
typedef Eigen::Matrix<GLfloat, 4, 1, Eigen::ColMajor> GLVector4f;
typedef Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> GLMatrix4f;

GLMatrix4f LookAt(const Eigen::Vector3d &eye,
                  const Eigen::Vector3d &lookat,
                  const Eigen::Vector3d &up);

GLMatrix4f Perspective(double field_of_view_,
                       double aspect,
                       double z_near,
                       double z_far);

GLMatrix4f Ortho(double left,
                 double right,
                 double bottom,
                 double top,
                 double z_near,
                 double z_far);

Eigen::Vector3d Project(const Eigen::Vector3d &point,
                        const GLMatrix4f &mvp_matrix,
                        const int width,
                        const int height);

Eigen::Vector3d Unproject(const Eigen::Vector3d &screen_point,
                          const GLMatrix4f &mvp_matrix,
                          const int width,
                          const int height);

int ColorCodeToPickIndex(const Eigen::Vector4i &color);

}  // namespace gl_util
}  // namespace visualization
}  // namespace open3d
