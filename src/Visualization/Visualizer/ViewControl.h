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

#include <Core/Geometry/Geometry.h>
#include <Core/Camera/PinholeCameraIntrinsic.h>
#include <Visualization/Visualizer/ViewParameters.h>
#include <Visualization/Utility/BoundingBox.h>
#include <Visualization/Utility/GLHelper.h>

namespace three {

class ViewControl
{
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
	/// Function to set view points
	/// This function obtains OpenGL context and calls OpenGL functions to set
	/// the view point.
	void SetViewMatrices(
			const Eigen::Matrix4d &model_matrix = Eigen::Matrix4d::Identity());

	/// Function to get equivalent view parameters (support orthogonal)
	bool ConvertToViewParameters(ViewParameters &status) const;
	bool ConvertFromViewParameters(const ViewParameters &status);

	/// Function to get equivalent pinhole camera parameters (does not support
	/// orthogonal since it is not a real camera view)
	bool ConvertToPinholeCameraParameters(
			PinholeCameraIntrinsic &intrinsic,
			Eigen::Matrix4d &extrinsic);
	bool ConvertFromPinholeCameraParameters(
			const PinholeCameraIntrinsic &intrinsic,
			const Eigen::Matrix4d &extrinsic);

	ProjectionType GetProjectionType() const;
	void SetProjectionParameters();
	virtual void Reset();
	virtual void ChangeFieldOfView(double step);
	virtual void ChangeWindowSize(int width, int height);

	// Function to process scaling
	/// \param scale is the relative distance mouse has scrolled.
	virtual void Scale(double scale);

	// Function to process rotation
	/// \param x and \param y are the distances the mouse cursor has moved.
	/// \param xo and \param yo are the original point coordinate the mouse
	/// cursor started to move from.
	/// Coordinates are measured in screen coordinates relative to the top-left
	/// corner of the window client area.
	virtual void Rotate(double x, double y, double xo = 0.0, double yo = 0.0);

	// Function to process translation
	/// \param x and \param y are the distances the mouse cursor has moved.
	/// \param xo and \param yo are the original point coordinate the mouse
	/// cursor started to move from.
	/// Coordinates are measured in screen coordinates relative to the top-left
	/// corner of the window client area.
	virtual void Translate(double x, double y, double xo = 0.0,
			double yo = 0.0);

	const BoundingBox &GetBoundingBox() const {
		return bounding_box_;
	}

	void ResetBoundingBox() {
		bounding_box_.Reset();
	}

	void FitInGeometry(const Geometry &geometry) {
		if (geometry.Dimension() == 3) {
			bounding_box_.FitInGeometry((const Geometry3D &)geometry);
		}
		SetProjectionParameters();
	}

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

protected:
	int window_width_ = 0;
	int window_height_ = 0;
	BoundingBox bounding_box_;
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
	GLHelper::GLMatrix4f projection_matrix_;
	GLHelper::GLMatrix4f view_matrix_;
	GLHelper::GLMatrix4f model_matrix_;
	GLHelper::GLMatrix4f MVP_matrix_;
};

}	// namespace three
