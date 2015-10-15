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

#include <IO/IO.h>

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

void ViewControlWithAnimation::AddKeyFrame()
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		ViewTrajectory::ViewStatus current_status = ConvertToViewStatus();
		if (view_trajectory_.view_status_.empty()) {
			view_trajectory_.view_status_.push_back(current_status);
			current_keyframe_ = 0.0;
		} else {
			size_t current_index = CurrentKeyframe();
			view_trajectory_.view_status_.insert(
					view_trajectory_.view_status_.begin() + current_index + 1,
					current_status);
			current_keyframe_ = current_index + 1.0;
		}
	}
}

void ViewControlWithAnimation::UpdateKeyFrame()
{
	if (animation_mode_ == ANIMATION_FREEMODE &&
			!view_trajectory_.view_status_.empty()) {
		view_trajectory_.view_status_[CurrentKeyframe()] = 
				ConvertToViewStatus();
	}
}

void ViewControlWithAnimation::DeleteKeyFrame()
{
	if (animation_mode_ == ANIMATION_FREEMODE &&
			!view_trajectory_.view_status_.empty()) {
		size_t current_index = CurrentKeyframe();
		view_trajectory_.view_status_.erase(
				view_trajectory_.view_status_.begin() + current_index);
		current_keyframe_ = RegularizeFrameIndex(current_index - 1.0,
				view_trajectory_.view_status_.size(),
				view_trajectory_.is_loop_);
	}
	SetViewControlFromTrajectory();
}

void ViewControlWithAnimation::AddSpinKeyFrames(int num_of_key_frames/* = 20*/)
{
	if (animation_mode_ == ANIMATION_FREEMODE) {
		double radian_per_step = M_PI * 2.0 / double(num_of_key_frames);
		for (int i = 0; i < num_of_key_frames; i++) {
			ViewControl::Rotate(radian_per_step / ROTATION_RADIAN_PER_PIXEL, 0);
			AddKeyFrame();
		}
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
			sprintf(buffer, "#%lu keyframe (%lu in total%s)",
					CurrentKeyframe() + 1,
					view_trajectory_.view_status_.size(),
					view_trajectory_.is_loop_ ? ", looped" : "");
		}
	} else {
		if (view_trajectory_.view_status_.empty()) {
			sprintf(buffer, "empty trajectory");
		} else {
			sprintf(buffer, "#%lu frame (%lu in total%s)", CurrentFrame() + 1,
					view_trajectory_.NumOfFrames(),
					view_trajectory_.is_loop_ ? ", looped" : "");
		}
	}
	return prefix + std::string(buffer);
}

void ViewControlWithAnimation::Step(double change)
{
	if (view_trajectory_.view_status_.empty()) {
		return;
	}
	if (animation_mode_ == ANIMATION_FREEMODE) {
		current_keyframe_ += change;
		current_keyframe_ = RegularizeFrameIndex(current_keyframe_,
				view_trajectory_.view_status_.size(),
				view_trajectory_.is_loop_);
	} else {
		current_frame_ += change;
		current_frame_ = RegularizeFrameIndex(current_frame_,
				view_trajectory_.NumOfFrames(), view_trajectory_.is_loop_);
	}
	SetViewControlFromTrajectory();
}

void ViewControlWithAnimation::GoToFirst()
{
	if (view_trajectory_.view_status_.empty()) {
		return;
	}
	if (animation_mode_ == ANIMATION_FREEMODE) {
		current_keyframe_ = 0.0;
	} else {
		current_frame_ = 0.0;
	}
	SetViewControlFromTrajectory();
}

void ViewControlWithAnimation::GoToLast()
{
	if (view_trajectory_.view_status_.empty()) {
		return;
	}
	if (animation_mode_ == ANIMATION_FREEMODE) {
		current_keyframe_ = view_trajectory_.view_status_.size() - 1.0;
	} else {
		current_frame_ = view_trajectory_.NumOfFrames() - 1.0;
	}
	SetViewControlFromTrajectory();
}

void ViewControlWithAnimation::TrajectoryCapture()
{
	if (view_trajectory_.view_status_.empty()) {
		return;
	}
	WriteViewTrajectory("ViewTrajectory_" + GetCurrentTimeStamp() + ".json",
			view_trajectory_);
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

double ViewControlWithAnimation::RegularizeFrameIndex(double current_frame,
		size_t num_of_frames, bool is_loop)
{
	if (num_of_frames == 0) {
		return 0.0;
	}
	double frame_index = current_frame;
	if (is_loop) {
		while (int(round(frame_index)) < 0) {
			frame_index += double(num_of_frames);
		}
		while (int(round(frame_index)) >= int(num_of_frames)) {
			frame_index -= double(num_of_frames);
		}
	} else {
		if (frame_index < 0.0) {
			frame_index = 0.0;
		}
		if (frame_index > num_of_frames - 1.0) {
			frame_index = num_of_frames - 1.0;
		}
	}
	return frame_index;
}

void ViewControlWithAnimation::SetViewControlFromTrajectory()
{
	if (view_trajectory_.view_status_.empty()) {
		return;
	}
	if (animation_mode_ == ANIMATION_FREEMODE) {
		ConvertFromViewStatus(view_trajectory_.view_status_[CurrentKeyframe()]);
	} else {
		ConvertFromViewStatus(
				view_trajectory_.GetInterpolatedFrame(CurrentFrame()));
	}
}

}	// namespace three
