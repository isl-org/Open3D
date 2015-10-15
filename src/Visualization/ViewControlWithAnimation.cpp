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

#include "ViewControlWithAnimation.h"

namespace three{

void ViewControlWithAnimation::Reset()
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		ViewControl::Reset();
	}
}

void ViewControlWithAnimation::ChangeFieldOfView(double step)
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		ViewControl::ChangeFieldOfView(step);
	}
}

void ViewControlWithAnimation::Scale(double scale)
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		ViewControl::Scale(scale);
	}
}

void ViewControlWithAnimation::Rotate(double x, double y)
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		ViewControl::Rotate(x, y);
	}
}

void ViewControlWithAnimation::Translate(double x, double y)
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		ViewControl::Translate(x, y);
	}
}

void ViewControlWithAnimation::AddSpinKeyFrames(int num_of_key_frames/* = 20*/)
{
}

void ViewControlWithAnimation::ToggleTrajectoryLoop()
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		view_trajectory_.is_loop_ = !view_trajectory_.is_loop_;
	}
}

void ViewControlWithAnimation::ChangeTrajectoryInterval(int change)
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		view_trajectory_.ChangeInterval(change); 
	}
}

std::string ViewControlWithAnimation::GetStatusString()
{
	std::string prefix;
	switch (animation_mode_) {
	case ANIMATION_FREEMODE:
		prefix = "Editing ";
		break;
	case ANIMATION_PREVIEWMODE:
		prefix = "Previewing ";
		break;
	case ANIMATION_PLAYMODE:
		prefix = "Playing ";
		break;
	}
	char buffer[DEFAULT_IO_BUFFER_SIZE];
	if (animation_mode_ == ANIMATION_FREEMODE) {
		if (view_trajectory_.view_status_.empty()) {
			sprintf(buffer, "empty trajectory");
		} else {
			sprintf(buffer, "#%d keyframe (%d in total)", current_frame_ + 1,
					view_trajectory_.view_status_.size());
		}
	} else {
		if (view_trajectory_.view_status_.empty()) {
			sprintf(buffer, "empty trajectory");
		} else {
			sprintf(buffer, "#%d frame (%d in total)", current_keyframe_ + 1,
					view_trajectory_.NumOfFrames());
		}
	}
	return prefix + std::string(buffer);
}

bool ViewControlWithAnimation::StepForward()
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		if (current_frame_ + 1 < view_trajectory_.view_status_.size()) {
			current_frame_++;
			ConvertFromViewStatus(
					view_trajectory_.view_status_[current_frame_]);
			return true;
		} else {
			return false;
		}
	} else {
		if (current_keyframe_ + 1 < view_trajectory_.NumOfFrames()) {
			current_keyframe_++;
			ConvertFromViewStatus(
					view_trajectory_.GetInterpolatedFrame(current_keyframe_));
			return true;
		} else {
			return false;
		}
	}
}

bool ViewControlWithAnimation::StepBackward()
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		if (current_frame_ > 0) {
			current_frame_--;
			ConvertFromViewStatus(
					view_trajectory_.view_status_[current_frame_]);
			return true;
		} else {
			return false;
		}
	} else {
		if (current_keyframe_ > 0) {
			current_keyframe_--;
			ConvertFromViewStatus(
					view_trajectory_.GetInterpolatedFrame(current_keyframe_));
			return true;
		} else {
			return false;
		}
	}
}

ViewTrajectory::ViewStatus ViewControlWithAnimation::ConvertToViewStatus()
{
	ViewTrajectory::ViewStatus status;
	status.field_of_view = field_of_view_;
	status.zoom = zoom_;
	status.lookat = lookat_;
	status.up = up_;
	status.front = front_;
	return status;
}

void ViewControlWithAnimation::ConvertFromViewStatus(
		const ViewTrajectory::ViewStatus status)
{
	field_of_view_ = status.field_of_view;
	zoom_ = status.zoom;
	lookat_ = status.lookat;
	up_ = status.up;
	front_ = status.front;
	SetProjectionParameters();
}

}	// namespace three
