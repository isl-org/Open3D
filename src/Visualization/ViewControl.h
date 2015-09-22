// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <Core/Core.h>

#include "BoundingBox.h"
#include "VisualizerHelper.h"

namespace three {

class ViewControl {
public:
	const double FIELD_OF_VIEW_MAX = 90.0;
	const double FIELD_OF_VIEW_MIN = 0.0;
	const double FIELD_OF_VIEW_DEFAULT = 60.0;
	const double FIELD_OF_VIEW_STEP = 5.0;

	const double ZOOM_DEFAULT = 0.7;
	const double ZOOM_MIN = 0.1;
	const double ZOOM_MAX = 2.0;
	const double ZOOM_STEP = 0.02;

	const double ROTATION_RADIAN_PER_PIXEL = 0.003;

	enum ProjectionType {
		PROJECTION_PERSPECTIVE = 0,
		PROJECTION_ORTHOGONAL = 1,
	};


public:
	/// Function to set view points
	/// This function obtains OpenGL context and calls OpenGL functions to set
	/// the view point.
	void SetViewPoint();

	ProjectionType GetProjectionType();
	void Reset();
	void SetProjectionParameters();
	void ChangeFieldOfView(double step);
	void ChangeWindowSize(int width, int height);
	void Scale(double scale);
	void Rotate(double x, double y);
	void Translate(double x, double y);

	const BoundingBox &GetBoundingBox() const {
		return bounding_box_;
	}

	void AddGeometry(const Geometry &geometry) {
		bounding_box_.AddGeometry(geometry);
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

protected:
	int window_width_ = 0;
	int window_height_ = 0;
	BoundingBox bounding_box_;
	Eigen::Vector3d eye_;
	Eigen::Vector3d lookat_;
	Eigen::Vector3d up_;
	Eigen::Vector3d front_;
	double distance_;
	double field_of_view_;
	double zoom_;
	double view_ratio_;
	double aspect_;
	GLHelper::GLMatrix4f projection_matrix_;
	GLHelper::GLMatrix4f view_matrix_;
	GLHelper::GLMatrix4f model_matrix_;
	GLHelper::GLMatrix4f MVP_matrix_;
};

}	// namespace three
