// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/ViewControlWithEditing.h"

namespace open3d {
namespace visualization {

void ViewControlWithEditing::Reset() {
    if (IsLocked()) return;
    if (editing_mode_ == EditingMode::FreeMode) {
        ViewControl::Reset();
    } else {
        field_of_view_ = FIELD_OF_VIEW_MIN;
        zoom_ = ZOOM_DEFAULT;
        lookat_ = bounding_box_.GetCenter();
        switch (editing_mode_) {
            case EditingMode::OrthoPositiveX:
                up_ = Eigen::Vector3d(0.0, 0.0, 1.0);
                front_ = Eigen::Vector3d(1.0, 0.0, 0.0);
                break;
            case EditingMode::OrthoNegativeX:
                up_ = Eigen::Vector3d(0.0, 0.0, 1.0);
                front_ = Eigen::Vector3d(-1.0, 0.0, 0.0);
                break;
            case EditingMode::OrthoPositiveY:
                up_ = Eigen::Vector3d(1.0, 0.0, 0.0);
                front_ = Eigen::Vector3d(0.0, 1.0, 0.0);
                break;
            case EditingMode::OrthoNegativeY:
                up_ = Eigen::Vector3d(1.0, 0.0, 0.0);
                front_ = Eigen::Vector3d(0.0, -1.0, 0.0);
                break;
            case EditingMode::OrthoPositiveZ:
                up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
                front_ = Eigen::Vector3d(0.0, 0.0, 1.0);
                break;
            case EditingMode::OrthoNegativeZ:
                up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
                front_ = Eigen::Vector3d(0.0, 0.0, -1.0);
                break;
            default:
                break;
        }
        SetProjectionParameters();
    }
}

void ViewControlWithEditing::ChangeFieldOfView(double step) {
    if (IsLocked()) return;
    if (editing_mode_ == EditingMode::FreeMode) {
        ViewControl::ChangeFieldOfView(step);
    } else {
        // Do nothing. Lock field of view if we are in orthogonal editing mode.
    }
}

void ViewControlWithEditing::Scale(double scale) {
    if (IsLocked()) return;
    if (editing_mode_ == EditingMode::FreeMode) {
        ViewControl::Scale(scale);
    } else {
        ViewControl::Scale(scale);
    }
}

void ViewControlWithEditing::Rotate(double x, double y, double xo, double yo) {
    if (IsLocked()) return;
    if (editing_mode_ == EditingMode::FreeMode) {
        ViewControl::Rotate(x, y);
    } else {
        // In orthogonal editing mode, lock front, and rotate around it
        double x0 = xo - (window_width_ / 2.0 - 0.5);
        double y0 = window_height_ / 2.0 - 0.5 - yo;
        double x1 = xo + x - (window_width_ / 2.0 - 0.5);
        double y1 = window_height_ / 2.0 - 0.5 - yo - y;
        if ((std::abs(x0 * y0) < 0.5) || (std::abs(x1 * y1) < 0.5)) {
            // Too close to screen center, skip the rotation
        } else {
            double theta = std::atan2(y1, x1) - std::atan2(y0, x0);
            up_ = up_ * std::cos(theta) + right_ * std::sin(theta);
        }
        SetProjectionParameters();
    }
}

void ViewControlWithEditing::Translate(double x,
                                       double y,
                                       double xo,
                                       double yo) {
    if (IsLocked()) return;
    if (editing_mode_ == EditingMode::FreeMode) {
        ViewControl::Translate(x, y, xo, yo);
    } else {
        ViewControl::Translate(x, y, xo, yo);
    }
}

void ViewControlWithEditing::SetEditingMode(EditingMode mode) {
    if (IsLocked()) return;
    if (editing_mode_ == EditingMode::FreeMode) {
        ConvertToViewParameters(view_status_backup_);
    }
    editing_mode_ = mode;
    if (editing_mode_ == EditingMode::FreeMode) {
        ConvertFromViewParameters(view_status_backup_);
    } else {
        Reset();
    }
}

std::string ViewControlWithEditing::GetStatusString() const {
    std::string prefix;
    switch (editing_mode_) {
        case EditingMode::FreeMode:
            prefix = "free view";
            break;
        case EditingMode::OrthoPositiveX:
        case EditingMode::OrthoNegativeX:
            prefix = "orthogonal X axis view";
            break;
        case EditingMode::OrthoPositiveY:
        case EditingMode::OrthoNegativeY:
            prefix = "orthogonal Y axis view";
            break;
        case EditingMode::OrthoPositiveZ:
        case EditingMode::OrthoNegativeZ:
            prefix = "orthogonal Z axis view";
            break;
    }
    return prefix + (IsLocked() ? ", lock camera for editing" : "");
}

}  // namespace visualization
}  // namespace open3d
