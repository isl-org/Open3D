// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/ViewControl.h"

#include "open3d/utility/Logging.h"

// Avoid warning caused by redefinition of APIENTRY macro
// defined also in glfw3.h
#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <Eigen/Dense>
#include <cmath>  // jspark

namespace open3d {
namespace visualization {

const double ViewControl::FIELD_OF_VIEW_MAX = 90.0;
const double ViewControl::FIELD_OF_VIEW_MIN = 5.0;
const double ViewControl::FIELD_OF_VIEW_DEFAULT = 60.0;
const double ViewControl::FIELD_OF_VIEW_STEP = 5.0;

const double ViewControl::ZOOM_DEFAULT = 0.7;
const double ViewControl::ZOOM_MIN = 0.02;
const double ViewControl::ZOOM_MAX = 2.0;
const double ViewControl::ZOOM_STEP = 0.02;

const double ViewControl::ROTATION_RADIAN_PER_PIXEL = 0.003;

void ViewControl::SetViewMatrices(
        const Eigen::Matrix4d
                &model_matrix /* = Eigen::Matrix4d::Identity()*/) {
    if (window_height_ <= 0 || window_width_ <= 0) {
        utility::LogWarning(
                "[ViewControl] SetViewPoint() failed because window height and "
                "width are not set.");
        return;
    }
    glViewport(0, 0, window_width_, window_height_);
    if (GetProjectionType() == ProjectionType::Perspective) {
        // Perspective projection
        z_near_ =
                constant_z_near_ > 0
                        ? constant_z_near_
                        : std::max(0.01 * bounding_box_.GetMaxExtent(),
                                   distance_ -
                                           3.0 * bounding_box_.GetMaxExtent());
        z_far_ = constant_z_far_ > 0
                         ? constant_z_far_
                         : distance_ + 3.0 * bounding_box_.GetMaxExtent();
        projection_matrix_ =
                gl_util::Perspective(field_of_view_, aspect_, z_near_, z_far_);
    } else {
        // Orthogonal projection
        // We use some black magic to support distance_ in orthogonal view
        z_near_ = constant_z_near_ > 0
                          ? constant_z_near_
                          : distance_ - 3.0 * bounding_box_.GetMaxExtent();
        z_far_ = constant_z_far_ > 0
                         ? constant_z_far_
                         : distance_ + 3.0 * bounding_box_.GetMaxExtent();
        projection_matrix_ =
                gl_util::Ortho(-aspect_ * view_ratio_, aspect_ * view_ratio_,
                               -view_ratio_, view_ratio_, z_near_, z_far_);
    }
    view_matrix_ = gl_util::LookAt(eye_, lookat_, up_);
    model_matrix_ = model_matrix.cast<GLfloat>();
    MVP_matrix_ = projection_matrix_ * view_matrix_ * model_matrix_;

    // uncomment to use the deprecated functions of legacy OpenGL
    // glMatrixMode(GL_PROJECTION);
    // glLoadIdentity();
    // glMatrixMode(GL_MODELVIEW);
    // glLoadIdentity();
    // glMultMatrixf(MVP_matrix_.data());
}

bool ViewControl::ConvertToViewParameters(ViewParameters &status) const {
    status.field_of_view_ = field_of_view_;
    status.zoom_ = zoom_;
    status.lookat_ = lookat_;
    status.up_ = up_;
    status.front_ = front_;
    status.boundingbox_min_ = bounding_box_.min_bound_;
    status.boundingbox_max_ = bounding_box_.max_bound_;
    return true;
}

bool ViewControl::ConvertFromViewParameters(const ViewParameters &status) {
    field_of_view_ = status.field_of_view_;
    zoom_ = status.zoom_;
    lookat_ = status.lookat_;
    up_ = status.up_;
    front_ = status.front_;
    bounding_box_.min_bound_ = status.boundingbox_min_;
    bounding_box_.max_bound_ = status.boundingbox_max_;
    SetProjectionParameters();
    return true;
}

void ViewControl::SetLookat(const Eigen::Vector3d &lookat) {
    lookat_ = lookat;
    SetProjectionParameters();
}

void ViewControl::SetUp(const Eigen::Vector3d &up) {
    up_ = up;
    SetProjectionParameters();
}

void ViewControl::SetFront(const Eigen::Vector3d &front) {
    front_ = front;
    SetProjectionParameters();
}

void ViewControl::SetZoom(const double zoom) {
    zoom_ = zoom;
    SetProjectionParameters();
}

bool ViewControl::ConvertToPinholeCameraParameters(
        camera::PinholeCameraParameters &parameters) {
    if (window_height_ <= 0 || window_width_ <= 0) {
        utility::LogWarning(
                "[ViewControl] ConvertToPinholeCameraParameters() failed "
                "because window height and width are not set.");
        return false;
    }
    if (GetProjectionType() == ProjectionType::Orthogonal) {
        utility::LogWarning(
                "[ViewControl] ConvertToPinholeCameraParameters() failed "
                "because orthogonal view cannot be translated to a pinhole "
                "camera.");
        return false;
    }
    SetProjectionParameters();
    auto intrinsic = camera::PinholeCameraIntrinsic();
    intrinsic.width_ = window_width_;
    intrinsic.height_ = window_height_;
    intrinsic.intrinsic_matrix_.setIdentity();
    double fov_rad = field_of_view_ / 180.0 * M_PI;
    double tan_half_fov = std::tan(fov_rad / 2.0);
    intrinsic.intrinsic_matrix_(0, 0) = intrinsic.intrinsic_matrix_(1, 1) =
            (double)window_height_ / tan_half_fov / 2.0;
    intrinsic.intrinsic_matrix_(0, 2) = (double)window_width_ / 2.0 - 0.5;
    intrinsic.intrinsic_matrix_(1, 2) = (double)window_height_ / 2.0 - 0.5;
    parameters.intrinsic_ = intrinsic;
    Eigen::Matrix4d extrinsic;
    extrinsic.setZero();
    Eigen::Vector3d front_dir = front_.normalized();
    Eigen::Vector3d up_dir = up_.normalized();
    Eigen::Vector3d right_dir = right_.normalized();
    extrinsic.block<1, 3>(0, 0) = right_dir.transpose();
    extrinsic.block<1, 3>(1, 0) = -up_dir.transpose();
    extrinsic.block<1, 3>(2, 0) = -front_dir.transpose();
    extrinsic(0, 3) = -right_dir.dot(eye_);
    extrinsic(1, 3) = up_dir.dot(eye_);
    extrinsic(2, 3) = front_dir.dot(eye_);
    extrinsic(3, 3) = 1.0;
    parameters.extrinsic_ = extrinsic;
    return true;
}

bool ViewControl::ConvertFromPinholeCameraParameters(
        const camera::PinholeCameraParameters &parameters,
        bool allow_arbitrary) {
    auto intrinsic = parameters.intrinsic_;
    auto extrinsic = parameters.extrinsic_;

    constexpr double threshold = 1.e-6;
    if (!allow_arbitrary &&
        (window_height_ <= 0 || window_width_ <= 0 ||
         window_height_ != intrinsic.height_ ||
         window_width_ != intrinsic.width_ ||
         std::abs(intrinsic.intrinsic_matrix_(0, 2) -
                  ((double)window_width_ / 2.0 - 0.5)) > threshold ||
         std::abs(intrinsic.intrinsic_matrix_(1, 2) -
                  ((double)window_height_ / 2.0 - 0.5)) > threshold)) {
        utility::LogWarning(
                "[ViewControl] ConvertFromPinholeCameraParameters() failed "
                "because window height and width do not match.");
        return false;
    }
    double tan_half_fov =
            (double)window_height_ / (intrinsic.intrinsic_matrix_(1, 1) * 2.0);
    double fov_rad = std::atan(tan_half_fov) * 2.0;
    double old_fov = field_of_view_;
    field_of_view_ = fov_rad * 180.0 / M_PI;
    if (!allow_arbitrary) {
        field_of_view_ = std::max(std::min(field_of_view_, FIELD_OF_VIEW_MAX),
                                  FIELD_OF_VIEW_MIN);
        if (GetProjectionType() == ProjectionType::Orthogonal) {
            field_of_view_ = old_fov;
            utility::LogWarning(
                    "[ViewControl] ConvertFromPinholeCameraParameters() failed "
                    "because field of view is impossible.");
            return false;
        }
    }
    right_ = extrinsic.block<1, 3>(0, 0).transpose();
    up_ = -extrinsic.block<1, 3>(1, 0).transpose();
    front_ = -extrinsic.block<1, 3>(2, 0).transpose();
    eye_ = extrinsic.block<3, 3>(0, 0).inverse() *
           (extrinsic.block<3, 1>(0, 3) * -1.0);

    auto bb_center = bounding_box_.GetCenter();
    double ideal_distance = std::abs((eye_ - bb_center).dot(front_));
    double ideal_zoom = ideal_distance *
                        std::tan(field_of_view_ * 0.5 / 180.0 * M_PI) /
                        bounding_box_.GetMaxExtent();
    zoom_ = ideal_zoom;
    if (!allow_arbitrary) {
        zoom_ = std::max(std::min(ideal_zoom, ZOOM_MAX), ZOOM_MIN);
    }
    view_ratio_ = zoom_ * bounding_box_.GetMaxExtent();
    distance_ = view_ratio_ / std::tan(field_of_view_ * 0.5 / 180.0 * M_PI);
    lookat_ = eye_ - front_ * distance_;
    return true;
}

ViewControl::ProjectionType ViewControl::GetProjectionType() const {
    if (field_of_view_ == FIELD_OF_VIEW_MIN) {
        return ProjectionType::Orthogonal;
    } else {
        return ProjectionType::Perspective;
    }
}

void ViewControl::Reset() {
    field_of_view_ = FIELD_OF_VIEW_DEFAULT;
    zoom_ = ZOOM_DEFAULT;
    lookat_ = bounding_box_.GetCenter();
    up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
    front_ = Eigen::Vector3d(0.0, 0.0, 1.0);
    SetProjectionParameters();
}

void ViewControl::SetProjectionParameters() {
    front_ = front_.normalized();
    right_ = up_.cross(front_).normalized();
    if (GetProjectionType() == ProjectionType::Perspective) {
        view_ratio_ = zoom_ * bounding_box_.GetMaxExtent();
        distance_ = view_ratio_ / std::tan(field_of_view_ * 0.5 / 180.0 * M_PI);
        eye_ = lookat_ + front_ * distance_;
    } else {
        view_ratio_ = zoom_ * bounding_box_.GetMaxExtent();
        distance_ =
                view_ratio_ / std::tan(FIELD_OF_VIEW_STEP * 0.5 / 180.0 * M_PI);
        eye_ = lookat_ + front_ * distance_;
    }

    ResetCameraLocalRotate();
}

void ViewControl::ChangeFieldOfView(double step) {
    field_of_view_ =
            std::max(std::min(field_of_view_ + step * FIELD_OF_VIEW_STEP,
                              FIELD_OF_VIEW_MAX),
                     FIELD_OF_VIEW_MIN);
    SetProjectionParameters();
}

void ViewControl::ChangeWindowSize(int width, int height) {
    window_width_ = width;
    window_height_ = height;
    aspect_ = (double)window_width_ / (double)window_height_;
    SetProjectionParameters();
}

geometry::Ray3D ViewControl::UnprojectPoint(double x, double y) const {
    if (!window_width_ || !window_height_) {
        return geometry::Ray3D(Eigen::Vector3d::Zero(),
                               Eigen::Vector3d::Zero());
    }

    gl_util::GLMatrix4f pvm = projection_matrix_ * view_matrix_ * model_matrix_;
    gl_util::GLMatrix4f inv = pvm.inverse();
    gl_util::GLVector4f clipping_pos(2.0f * (float)x / window_width_ - 1.0f,
                                     1.0f - 2.0f * (float)y / window_height_,
                                     1.0f, 1.0f);
    gl_util::GLVector4f homogeneous_pos = inv * clipping_pos;
    homogeneous_pos /= homogeneous_pos[3];
    Eigen::Vector3d world_pos = Eigen::Vector3d(
            homogeneous_pos[0], homogeneous_pos[1], homogeneous_pos[2]);

    if (GetProjectionType() == ProjectionType::Orthogonal) {
        Eigen::Vector3d camera_axis = lookat_ - eye_;
        camera_axis.normalize();
        Eigen::Vector3d delta = world_pos - eye_;
        double projection = delta.dot(camera_axis);
        Eigen::Vector3d origin = world_pos - projection * camera_axis;
        return geometry::Ray3D(origin, camera_axis);
    }

    return geometry::Ray3D(eye_, world_pos - eye_);
}

void ViewControl::Scale(double scale) {
    zoom_ = std::max(std::min(zoom_ + scale * ZOOM_STEP, ZOOM_MAX), ZOOM_MIN);
    SetProjectionParameters();
}

void ViewControl::Rotate(double x,
                         double y,
                         double xo /* = 0.0*/,
                         double yo /* = 0.0*/) {
    // some black magic to do rotation
    double alpha = x * ROTATION_RADIAN_PER_PIXEL;
    double beta = y * ROTATION_RADIAN_PER_PIXEL;
    front_ = (front_ * std::cos(alpha) - right_ * std::sin(alpha)).normalized();
    right_ = up_.cross(front_).normalized();
    front_ = (front_ * std::cos(beta) + up_ * std::sin(beta)).normalized();
    up_ = front_.cross(right_).normalized();
    SetProjectionParameters();
}

void ViewControl::Translate(double x,
                            double y,
                            double xo /* = 0.0*/,
                            double yo /* = 0.0*/) {
    Eigen::Vector3d shift = right_ * (-x) / window_height_ * view_ratio_ * 2.0 +
                            up_ * y / window_height_ * view_ratio_ * 2.0;
    eye_ += shift;
    lookat_ += shift;
    SetProjectionParameters();
}

void ViewControl::CameraLocalTranslate(double forward,
                                       double right,
                                       double up) {
    lookat_ += (-forward) * front_.normalized() + right * right_.normalized() +
               up * up_.normalized();
    SetProjectionParameters();
}

void ViewControl::CameraLocalRotate(double x,
                                    double y,
                                    double xo /* = 0.0*/,
                                    double yo /* = 0.0*/) {
    const double degrees_per_unit = 100 / distance_;
    const double x_shift = (-x) / window_height_ * view_ratio_ * 2.0;
    const double y_shift = y / window_height_ * view_ratio_ * 2.0;

    local_rotate_up_accum_ += y_shift;
    local_rotate_right_accum_ += x_shift;

    const auto m =
            Eigen::AngleAxisd(
                    -degrees_per_unit * local_rotate_up_accum_ * M_PI / 180,
                    start_local_rotate_right_) *
            Eigen::AngleAxisd(
                    degrees_per_unit * local_rotate_right_accum_ * M_PI / 180,
                    start_local_rotate_up_);

    front_ = m * start_local_rotate_front_;
    lookat_ = start_local_rotate_eye_ - front_ * distance_;
    up_ = start_local_rotate_up_;

    // Prevent SetProjectionParameters re-setting camera local rotation starts
    auto orig_up = start_local_rotate_up_;
    auto orig_right = start_local_rotate_right_;
    auto orig_eye = start_local_rotate_eye_;
    auto orig_front = start_local_rotate_front_;
    auto orig_lookat = start_local_rotate_lookat_;
    auto orig_up_accum = local_rotate_up_accum_;
    auto orig_right_accum = local_rotate_right_accum_;

    SetProjectionParameters();

    start_local_rotate_up_ = orig_up;
    start_local_rotate_right_ = orig_right;
    start_local_rotate_eye_ = orig_eye;
    start_local_rotate_front_ = orig_front;
    start_local_rotate_lookat_ = orig_lookat;
    local_rotate_up_accum_ = orig_up_accum;
    local_rotate_right_accum_ = orig_right_accum;
}

void ViewControl::ResetCameraLocalRotate() {
    start_local_rotate_up_ = up_;
    start_local_rotate_right_ = right_;
    start_local_rotate_eye_ = eye_;
    start_local_rotate_lookat_ = lookat_;
    start_local_rotate_front_ = front_;
    local_rotate_up_accum_ = 0;
    local_rotate_right_accum_ = 0;
}

void ViewControl::Roll(double x) {
    double alpha = x * ROTATION_RADIAN_PER_PIXEL;
    // Rotates up_ vector using Rodrigues' rotation formula.
    // front_ vector is an axis of rotation.
    up_ = up_ * std::cos(alpha) + front_.cross(up_) * std::sin(alpha) +
          front_ * (front_.dot(up_)) * (1.0 - std::cos(alpha));
    up_.normalized();
    SetProjectionParameters();
}

}  // namespace visualization
}  // namespace open3d
