// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
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

#include "ViewControl.h"

#include <Eigen/Dense>
#include <GLFW/glfw3.h>

namespace three{

const double ViewControl::FIELD_OF_VIEW_MAX = 90.0;
const double ViewControl::FIELD_OF_VIEW_MIN = 0.0;
const double ViewControl::FIELD_OF_VIEW_DEFAULT = 60.0;
const double ViewControl::FIELD_OF_VIEW_STEP = 5.0;

const double ViewControl::ZOOM_DEFAULT = 0.7;
const double ViewControl::ZOOM_MIN = 0.1;
const double ViewControl::ZOOM_MAX = 2.0;
const double ViewControl::ZOOM_STEP = 0.02;

const double ViewControl::ROTATION_RADIAN_PER_PIXEL = 0.003;

ViewControl::ProjectionType ViewControl::GetProjectionType()
{
	if (field_of_view_ > FIELD_OF_VIEW_MIN + FIELD_OF_VIEW_STEP / 2.0) {
		return PROJECTION_PERSPECTIVE;
	} else {
		return PROJECTION_ORTHOGONAL;
	}
}

void ViewControl::Reset()
{
	field_of_view_ = FIELD_OF_VIEW_DEFAULT;
	zoom_ = ZOOM_DEFAULT;
	lookat_ = bounding_box_.GetCenter();
	up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
	front_ = Eigen::Vector3d(0.0, 0.0, 1.0);
	SetProjectionParameters();
}

void ViewControl::SetProjectionParameters()
{
	if (GetProjectionType() == PROJECTION_PERSPECTIVE) {
		view_ratio_ = zoom_ * bounding_box_.GetSize();
		distance_ = view_ratio_ / 
				std::tan(field_of_view_ * 0.5 / 180.0 * M_PI);
		eye_ = lookat_ + front_ * distance_;
	} else {
		view_ratio_ = zoom_ * bounding_box_.GetSize();
		distance_ = view_ratio_ / 
				std::tan(FIELD_OF_VIEW_STEP * 0.5 / 180.0 * M_PI);
		eye_ = lookat_ + front_ * distance_;
	}
}

void ViewControl::ChangeFieldOfView(double step)
{
	double field_of_view_new = field_of_view_ + step * FIELD_OF_VIEW_STEP;
	if (field_of_view_new >= FIELD_OF_VIEW_MIN &&
			field_of_view_new <= FIELD_OF_VIEW_MAX)
	{
		field_of_view_ = field_of_view_new;
	}
	SetProjectionParameters();
}

void ViewControl::ChangeWindowSize(int width, int height)
{
	window_width_ = width;
	window_height_ = height;
	aspect_ = (double)window_width_ / (double)window_height_;
	SetProjectionParameters();
}

void ViewControl::Scale(double scale)
{
	double zoom_new = zoom_ + scale * ZOOM_STEP;
	if (zoom_new >= ZOOM_MIN && zoom_new <= ZOOM_MAX) {
		zoom_ = zoom_new;
	}
	SetProjectionParameters();
}

void ViewControl::Rotate(double x, double y)
{
	// some black magic to do rotation
	Eigen::Vector3d right = up_.cross(front_);
	double alpha = x * ROTATION_RADIAN_PER_PIXEL;
	double beta = y * ROTATION_RADIAN_PER_PIXEL;
	front_ = front_ * std::cos(alpha) + right * std::sin(alpha);
	right = up_.cross(front_);
	right.normalize();
	front_ = front_ * std::cos(beta) + up_ * std::sin(beta);
	front_.normalize();
	up_ = front_.cross(right);
	up_.normalize();
	SetProjectionParameters();
}

void ViewControl::Translate(double x, double y)
{
	Eigen::Vector3d right = up_.cross(front_);
	Eigen::Vector3d shift = 
			right * x / window_height_ * view_ratio_ * 2.0 +
			up_ * y / window_height_ * view_ratio_ * 2.0;
	eye_ += shift;
	lookat_ += shift;
	SetProjectionParameters();
}

void ViewControl::SetViewPoint()
{
	if (window_height_ <= 0 || window_width_ <= 0) {
		PrintWarning("[ViewControl] SetViewPoint() failed because window height and width are not set.");
		return;
	}
	glViewport(0, 0, window_width_, window_height_);
	if (GetProjectionType() == PROJECTION_PERSPECTIVE)
	{
		// Perspective projection
		projection_matrix_ = GLHelper::Perspective(field_of_view_, aspect_,
				std::max(0.0001, distance_ - 1.0 * bounding_box_.GetSize()),
				distance_ + 1.0 * bounding_box_.GetSize());
	} else {
		// Orthogonal projection
		// We use some black magic to support distance_ in orthogonal view
		projection_matrix_ = GLHelper::Ortho(
				-aspect_ * view_ratio_,	aspect_ * view_ratio_,
				-view_ratio_, view_ratio_,
				distance_ - 3.0 * bounding_box_.GetSize(),
				distance_ + 3.0 * bounding_box_.GetSize());
	}
	view_matrix_ = GLHelper::LookAt(eye_, lookat_, up_ );
	model_matrix_ = GLHelper::GLMatrix4f::Identity();
	MVP_matrix_ = projection_matrix_ * view_matrix_ * model_matrix_;
	
	// deprecated functions of legacy OpenGL
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMultMatrixf(MVP_matrix_.data());
}

}	// namespace three
