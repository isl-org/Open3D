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

#include "ViewControlWithEditing.h"

namespace three{

void ViewControlWithEditing::Reset()
{
	if (editing_mode_ == EDITING_FREEMODE) {
		ViewControl::Reset();
	} else {
		field_of_view_ = FIELD_OF_VIEW_MIN;
		zoom_ = ZOOM_DEFAULT;
		lookat_ = bounding_box_.GetCenter();
		switch (editing_mode_) {
		case EDITING_ORTHO_POSITIVE_X:
			up_ = Eigen::Vector3d(0.0, 0.0, 1.0);
			front_ = Eigen::Vector3d(1.0, 0.0, 0.0);
			break;
		case EDITING_ORTHO_NEGATIVE_X:
			up_ = Eigen::Vector3d(0.0, 0.0, 1.0);
			front_ = Eigen::Vector3d(-1.0, 0.0, 0.0);
			break;
		case EDITING_ORTHO_POSITIVE_Y:
			up_ = Eigen::Vector3d(1.0, 0.0, 0.0);
			front_ = Eigen::Vector3d(0.0, 1.0, 0.0);
			break;
		case EDITING_ORTHO_NEGATIVE_Y:
			up_ = Eigen::Vector3d(1.0, 0.0, 0.0);
			front_ = Eigen::Vector3d(0.0, -1.0, 0.0);
			break;
		case EDITING_ORTHO_POSITIVE_Z:
			up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
			front_ = Eigen::Vector3d(0.0, 0.0, 1.0);
			break;
		case EDITING_ORTHO_NEGATIVE_Z:
			up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
			front_ = Eigen::Vector3d(0.0, 0.0, -1.0);
			break;
		}
		SetProjectionParameters();
	}
}

void ViewControlWithEditing::ChangeFieldOfView(double step)
{
	if (editing_mode_ == EDITING_FREEMODE) {
		ViewControl::ChangeFieldOfView(step);
	} else {
		// Do nothing. Lock field of view if we are in orthogonal editing mode.
	}
}

void ViewControlWithEditing::Scale(double scale)
{
	if (editing_mode_ == EDITING_FREEMODE) {
		ViewControl::Scale(scale);
	} else {
		ViewControl::Scale(scale);
	}
}

void ViewControlWithEditing::Rotate(double x, double y, double xo, double yo)
{
	if (editing_mode_ == EDITING_FREEMODE) {
		ViewControl::Rotate(x, y);
	} else {
		// In orthogonal editing mode, lock front, and rotate around it
		double x0 = xo - (window_width_ / 2.0 - 0.5);
		double y0 = window_height_ / 2.0 - 0.5 - yo;
		double x1 = xo + x - (window_width_ / 2.0 - 0.5);
		double y1 = window_height_ / 2.0 - 0.5 - yo - y;
		if ((x0 < 0.5 && y0 < 0.5) || (x1 < 0.5 && y1 < 0.5)) {
			// Too close to screen center, skip the rotation
		} else {
			double theta = std::atan2(y1, x1) - std::atan2(y0, x0);
			up_ = up_ * std::cos(theta) - right_ * std::sin(theta);
		}
		SetProjectionParameters();
	}
}

void ViewControlWithEditing::Translate(double x, double y, double xo, double yo)
{
	if (editing_mode_ == EDITING_FREEMODE) {
		ViewControl::Translate(x, y, xo, yo);
	} else {
		ViewControl::Translate(x, y, xo, yo);
	}
}

void ViewControlWithEditing::SetEditingMode(EditingMode mode)
{
	if (editing_mode_ == EDITING_FREEMODE) {
		ConvertToViewParameters(view_status_backup_);
	}
	editing_mode_ = mode;
	if (editing_mode_ == EDITING_FREEMODE) {
		ConvertFromViewParameters(view_status_backup_);
	} else {
		Reset();
	}
}

std::string ViewControlWithEditing::GetStatusString()
{
	std::string prefix;
	switch (editing_mode_) {
	case EDITING_FREEMODE:
		prefix = "free view";
		break;
	case EDITING_ORTHO_POSITIVE_X:
	case EDITING_ORTHO_NEGATIVE_X:
		prefix = "orthogonal X asix";
		break;
	case EDITING_ORTHO_POSITIVE_Y:
	case EDITING_ORTHO_NEGATIVE_Y:
		prefix = "orthogonal Y asix";
		break;
	case EDITING_ORTHO_POSITIVE_Z:
	case EDITING_ORTHO_NEGATIVE_Z:
		prefix = "orthogonal Z asix";
		break;
	}
	return prefix;
}

}	// namespace three
