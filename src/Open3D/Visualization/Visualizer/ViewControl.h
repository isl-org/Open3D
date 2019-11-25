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

#pragma once

#include "Open3D/Camera/PinholeCameraParameters.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/Geometry.h"
#include "Open3D/Visualization/Utility/GLHelper.h"
#include "Open3D/Visualization/Visualizer/ViewParameters.h"

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

    /// Function to get equivalent pinhole camera parameters (does not support
    /// orthogonal since it is not a real camera view).
    ///
    /// \param The pinhole camera parameter to convert to.
    bool ConvertToPinholeCameraParameters(
            camera::PinholeCameraParameters &parameters);
    /// Function to get view controller from pinhole camera parameters.
    ///
    /// \param parameters The pinhole camera parameter to convert from.
    bool ConvertFromPinholeCameraParameters(
            const camera::PinholeCameraParameters &parameters);

    ProjectionType GetProjectionType() const;
    void SetProjectionParameters();
    virtual void Reset();
    /// Function to change field of view.
    ///
    /// \param step The step to change field of view.
    virtual void ChangeFieldOfView(double step);
    virtual void ChangeWindowSize(int width, int height);

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
    GLHelper::GLMatrix4f GetMVPMatrix() const { return MVP_matrix_; }
    GLHelper::GLMatrix4f GetProjectionMatrix() const {
        return projection_matrix_;
    }
    GLHelper::GLMatrix4f GetViewMatrix() const { return view_matrix_; }
    GLHelper::GLMatrix4f GetModelMatrix() const { return model_matrix_; }
    GLHelper::GLVector3f GetEye() const { return eye_.cast<GLfloat>(); }
    GLHelper::GLVector3f GetLookat() const { return lookat_.cast<GLfloat>(); }
    GLHelper::GLVector3f GetUp() const { return up_.cast<GLfloat>(); }
    GLHelper::GLVector3f GetFront() const { return front_.cast<GLfloat>(); }
    GLHelper::GLVector3f GetRight() const { return right_.cast<GLfloat>(); }
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
    GLHelper::GLMatrix4f projection_matrix_;
    GLHelper::GLMatrix4f view_matrix_;
    GLHelper::GLMatrix4f model_matrix_;
    GLHelper::GLMatrix4f MVP_matrix_;
};

}  // namespace visualization
}  // namespace open3d
