// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/visualization/rendering/Camera.h"

namespace open3d {
namespace visualization {
namespace rendering {

/// Base class for rotating and dollying (translating along forward axis).
/// Could be used for a camera, or also something else, like a the
/// direction of a directional light.
class MatrixInteractorLogic {
public:
    virtual ~MatrixInteractorLogic();

    void SetViewSize(int width, int height);
    int GetViewWidth() const;
    int GetViewHeight() const;

    const geometry::AxisAlignedBoundingBox& GetBoundingBox() const;
    virtual void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds);

    Eigen::Vector3f GetCenterOfRotation() const;

    void SetMouseDownInfo(const Camera::Transform& matrix,
                          const Eigen::Vector3f& center_of_rotation);

    const Camera::Transform& GetMatrix() const;

    /// Rotates about an axis defined by dx * matrixLeft, dy * matrixUp.
    /// `dy` is assumed to be in window-style coordinates, that is, going
    /// up produces a negative dy. The axis goes through the center of
    /// rotation.
    virtual void Rotate(int dx, int dy);

    /// Same as Rotate() except that the dx-axis and the dy-axis are
    /// specified
    virtual void RotateWorld(int dx,
                             int dy,
                             const Eigen::Vector3f& x_axis,
                             const Eigen::Vector3f& y_axis);

    /// Rotates about the forward axis of the matrix
    virtual void RotateZ(int dx, int dy);

    virtual void RotateZWorld(int dx, int dy, const Eigen::Vector3f& forward);

    enum class DragType { MOUSE, WHEEL, TWO_FINGER };

    /// Moves the matrix along the forward axis. (This is one type
    /// of zoom.)
    virtual void Dolly(float dy, DragType drag_type);
    virtual void Dolly(float z_dist, Camera::Transform matrix);

private:
    Camera::Transform matrix_;

protected:
    int view_width_ = 1;
    int view_height_ = 1;
    double model_size_ = 20.0;
    geometry::AxisAlignedBoundingBox model_bounds_;
    Eigen::Vector3f center_of_rotation_;

    Camera::Transform matrix_at_mouse_down_;
    Eigen::Vector3f center_of_rotation_at_mouse_down_;

    void SetMatrix(const Camera::Transform& matrix);
    float CalcRotateRadians(int dx, int dy);
    float CalcRotateZRadians(int dx, int dy);
    float CalcDollyDist(float dy,
                        DragType drag_type,
                        const Camera::Transform& matrix);
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
