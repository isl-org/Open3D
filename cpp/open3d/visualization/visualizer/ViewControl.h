// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/camera/PinholeCameraParameters.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/Geometry.h"
#include "open3d/geometry/Line3D.h"
#include "open3d/visualization/utility/GLHelper.h"
#include "open3d/visualization/visualizer/ViewParameters.h"

namespace open3d {
namespace visualization {

/// \class ViewControl
///
/// \brief View controller for visualizer.
class ViewControl {
public:
    static const double FIELD_OF_VIEW_MAX;
    static const double FIELD_OF_VIEW_MIN;
    static const double FIELD_OF_VIEW_DEFAULT;
    static const double FIELD_OF_VIEW_STEP;

    static const double ZOOM_DEFAULT;
    static const double ZOOM_MIN;
    static const double ZOOM_MAX;
    static const double ZOOM_STEP;

    static const double ROTATION_RADIAN_PER_PIXEL;

    enum ProjectionType {
        Perspective = 0,
        Orthogonal = 1,
    };

public:
    virtual ~ViewControl() {}

    /// Function to set view points
    /// This function obtains OpenGL context and calls OpenGL functions to set
    /// the view point.
    void SetViewMatrices(
            const Eigen::Matrix4d &model_matrix = Eigen::Matrix4d::Identity());

    /// Function to get equivalent view parameters (support orthogonal)
    bool ConvertToViewParameters(ViewParameters &status) const;
    bool ConvertFromViewParameters(const ViewParameters &status);

    void SetLookat(const Eigen::Vector3d &lookat);
    void SetUp(const Eigen::Vector3d &up);
    void SetFront(const Eigen::Vector3d &front);
    void SetZoom(const double zoom);

    /// Function to get equivalent pinhole camera parameters (does not support
    /// orthogonal since it is not a real camera view).
    ///
    /// \param parameters The pinhole camera parameter to convert to.
    bool ConvertToPinholeCameraParameters(
            camera::PinholeCameraParameters &parameters);

    /// Function to get view controller from pinhole camera parameters.
    ///
    /// \param parameters The pinhole camera parameter to convert from.
    /// \param allow_arbitrary Allow an arbitrary pinhole camera parameters.
    /// This can be useful to render images or depthmaps without any restriction
    /// in window size, FOV and zoom.
    bool ConvertFromPinholeCameraParameters(
            const camera::PinholeCameraParameters &parameters,
            bool allow_arbitrary = false);

    ProjectionType GetProjectionType() const;
    void SetProjectionParameters();
    virtual void Reset();
    /// Function to change field of view.
    ///
    /// \param step The step to change field of view.
    virtual void ChangeFieldOfView(double step);
    virtual void ChangeWindowSize(int width, int height);

    /// Function to unproject a point on the window and obtain a ray from the
    /// camera to that point in 3D.
    ///
    /// \param x The coordinate of the point in x-axis, in pixels
    /// \param y The coordinate of the point in y-axis, in pixels
    geometry::Ray3D UnprojectPoint(double x, double y) const;

    /// Function to process scaling
    ///
    /// \param scale is the scale ratio.
    virtual void Scale(double scale);

    /// \brief Function to process rotation.
    ///
    /// Coordinates are measured in screen coordinates relative to the top-left
    /// corner of the window client area.
    ///
    /// \param x The distance the mouse cursor has moved in x-axis.
    /// \param y The distance the mouse cursor has moved in y-axis.
    /// \param xo Original point coordinate of the mouse in x-axis.
    /// \param yo Original point coordinate of the mouse in y-axis.
    virtual void Rotate(double x, double y, double xo = 0.0, double yo = 0.0);

    /// \brief Function to process translation
    ///
    /// Coordinates are measured in screen coordinates relative to the top-left
    /// corner of the window client area.
    ///
    /// \param x Distance the mouse cursor has moved in x-axis.
    /// \param y Distance the mouse cursor has moved in y-axis.
    /// \param xo Original point coordinate of the mouse in x-axis.
    /// \param yo Original point coordinate of the mouse in y-axis.
    virtual void Translate(double x,
                           double y,
                           double xo = 0.0,
                           double yo = 0.0);

    virtual void CameraLocalTranslate(double forward, double right, double up);
    virtual void CameraLocalRotate(double x,
                                   double y,
                                   double xo = 0.0,
                                   double yo = 0.0);
    virtual void ResetCameraLocalRotate();

    // Function to process rolling
    /// \param x is the distances the mouse cursor has moved.
    /// Coordinates are measured in screen coordinates relative to the top-left
    /// corner of the window client area.
    virtual void Roll(double x);

    const geometry::AxisAlignedBoundingBox &GetBoundingBox() const {
        return bounding_box_;
    }

    void ResetBoundingBox() { bounding_box_.Clear(); }

    void FitInGeometry(const geometry::Geometry &geometry) {
        if (geometry.Dimension() == 3) {
            bounding_box_ += ((const geometry::Geometry3D &)geometry)
                                     .GetAxisAlignedBoundingBox();
        }
        SetProjectionParameters();
    }

    /// Function to get field of view.
    double GetFieldOfView() const { return field_of_view_; }
    gl_util::GLMatrix4f GetMVPMatrix() const { return MVP_matrix_; }
    gl_util::GLMatrix4f GetProjectionMatrix() const {
        return projection_matrix_;
    }
    gl_util::GLMatrix4f GetViewMatrix() const { return view_matrix_; }
    gl_util::GLMatrix4f GetModelMatrix() const { return model_matrix_; }
    gl_util::GLVector3f GetEye() const { return eye_.cast<GLfloat>(); }
    gl_util::GLVector3f GetLookat() const { return lookat_.cast<GLfloat>(); }
    gl_util::GLVector3f GetUp() const { return up_.cast<GLfloat>(); }
    gl_util::GLVector3f GetFront() const { return front_.cast<GLfloat>(); }
    gl_util::GLVector3f GetRight() const { return right_.cast<GLfloat>(); }
    int GetWindowWidth() const { return window_width_; }
    int GetWindowHeight() const { return window_height_; }
    double GetZNear() const { return z_near_; }
    double GetZFar() const { return z_far_; }

    /// Function to change the near z-plane of the visualizer to a constant
    /// value, i.e., independent of zoom and bounding box size.
    ///
    /// \param z_near The depth of the near z-plane of the visualizer.
    void SetConstantZNear(double z_near) { constant_z_near_ = z_near; }
    /// Function to change the far z-plane of the visualizer to a constant
    /// value, i.e., independent of zoom and bounding box size.
    ///
    /// \param z_far The depth of the far z-plane of the visualizer.
    void SetConstantZFar(double z_far) { constant_z_far_ = z_far; }
    /// Function to remove a previously set constant z near value, i.e., near
    /// z-plane of the visualizer is dynamically set dependent on zoom and
    /// bounding box size.
    void UnsetConstantZNear() { constant_z_near_ = -1; }
    /// Function to remove a previously set constant z far value, i.e., far
    /// z-plane of the visualizer is dynamically set dependent on zoom and
    /// bounding box size.
    void UnsetConstantZFar() { constant_z_far_ = -1; }

protected:
    int window_width_ = 0;
    int window_height_ = 0;
    geometry::AxisAlignedBoundingBox bounding_box_;
    Eigen::Vector3d eye_;
    Eigen::Vector3d lookat_;
    Eigen::Vector3d up_;
    Eigen::Vector3d front_;
    Eigen::Vector3d right_;
    double distance_;
    double field_of_view_;
    double zoom_;
    double view_ratio_;
    double aspect_;
    double z_near_;
    double z_far_;
    double constant_z_near_ = -1;
    double constant_z_far_ = -1;
    gl_util::GLMatrix4f projection_matrix_;
    gl_util::GLMatrix4f view_matrix_;
    gl_util::GLMatrix4f model_matrix_;
    gl_util::GLMatrix4f MVP_matrix_;

    Eigen::Vector3d start_local_rotate_up_;
    Eigen::Vector3d start_local_rotate_right_;
    Eigen::Vector3d start_local_rotate_front_;
    Eigen::Vector3d start_local_rotate_eye_;
    Eigen::Vector3d start_local_rotate_lookat_;
    double local_rotate_up_accum_;
    double local_rotate_right_accum_;
};

}  // namespace visualization
}  // namespace open3d
