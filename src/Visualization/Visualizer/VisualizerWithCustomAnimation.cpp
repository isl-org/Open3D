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

#include "VisualizerWithCustomAnimation.h"

#include <thread>

#include <Core/Utility/FileSystem.h>
#include <Core/Utility/Console.h>
#include <Core/Camera/PinholeCameraTrajectory.h>
#include <Visualization/Visualizer/ViewControlWithCustomAnimation.h>
#include <IO/ClassIO/IJsonConvertibleIO.h>

namespace three{

VisualizerWithCustomAnimation::VisualizerWithCustomAnimation()
{
}

VisualizerWithCustomAnimation::~VisualizerWithCustomAnimation()
{
}

void VisualizerWithCustomAnimation::PrintVisualizerHelp()
{
	Visualizer::PrintVisualizerHelp();
	PrintInfo("  -- Animation control --\n");
	PrintInfo("    Ctrl + F     : Enter freeview (editing) mode.\n");
	PrintInfo("    Ctrl + W     : Enter preview mode.\n");
	PrintInfo("    Ctrl + P     : Enter animation mode and play animation from beginning.\n");
	PrintInfo("    Ctrl + R     : Enter animation mode, play animation, and record screen.\n");
	PrintInfo("    Ctrl + G     : Enter animation mode, play animation, and record depth.\n");
	PrintInfo("    Ctrl + S     : Save the camera path into a json file.\n");
	PrintInfo("\n");
	PrintInfo("    -- In free view mode --\n");
	PrintInfo("    Ctrl + <-/-> : Go backward/forward a keyframe.\n");
	PrintInfo("    Ctrl + Wheel : Same as Ctrl + <-/->.\n");
	PrintInfo("    Ctrl + [/]   : Go to the first/last keyframe.\n");
	PrintInfo("    Ctrl + +/-   : Increase/decrease interval between keyframes.\n");
	PrintInfo("    Ctrl + L     : Turn on/off camera path as a loop.\n");
	PrintInfo("    Ctrl + A     : Add a keyframe right after the current keyframe.\n");
	PrintInfo("    Ctrl + U     : Update the current keyframe.\n");
	PrintInfo("    Ctrl + D     : Delete the current keyframe.\n");
	PrintInfo("    Ctrl + N     : Add 360 spin right after the current keyframe.\n");
	PrintInfo("    Ctrl + E     : Erase the entire camera path.\n");
	PrintInfo("\n");
	PrintInfo("    -- In preview mode --\n");
	PrintInfo("    Ctrl + <-/-> : Go backward/forward a frame.\n");
	PrintInfo("    Ctrl + Wheel : Same as Ctrl + <-/->.\n");
	PrintInfo("    Ctrl + [/]   : Go to beginning/end of the camera path.\n");
	PrintInfo("\n");
}

void VisualizerWithCustomAnimation::UpdateWindowTitle()
{
	if (window_ != NULL) {
		auto &view_control = (ViewControlWithCustomAnimation &)
				(*view_control_ptr_);
		std::string new_window_title = window_name_ + " - " +
				view_control.GetStatusString();
		glfwSetWindowTitle(window_, new_window_title.c_str());
	}
}

void VisualizerWithCustomAnimation::Play(bool recording/* = false*/,
		bool recording_depth/* = false*/,
		bool close_window_when_animation_ends/* = false*/)
{
	auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
	if (view_control.NumOfFrames() == 0) {
		PrintInfo("Abort playing due to empty trajectory.\n");
		return;
	}
	view_control.SetAnimationMode(
			ViewControlWithCustomAnimation::AnimationMode::PlayMode);
	is_redraw_required_ = true;
	UpdateWindowTitle();
	recording_file_index_ = 0;
	ResetConsoleProgress(view_control.NumOfFrames(), "Play animation: ");
	auto trajectory_ptr = std::make_shared<PinholeCameraTrajectory>();
	bool recording_trajectory = view_control.IsValidPinholeCameraTrajectory();
	if (recording) {
		if (recording_depth) {
			filesystem::MakeDirectoryHierarchy(recording_depth_basedir_);
		} else {
			filesystem::MakeDirectoryHierarchy(recording_image_basedir_);
		}
	}
	RegisterAnimationCallback(
			[=](Visualizer *vis) {
				// The lambda function captures no references to avoid dangling
				// references
				auto &view_control =
						(ViewControlWithCustomAnimation &)(*view_control_ptr_);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				recording_file_index_++;
				if (recording) {
					if (recording_trajectory) {
						Eigen::Matrix4d extrinsic;
						view_control.ConvertToPinholeCameraParameters(
								trajectory_ptr->intrinsic_, extrinsic);
						trajectory_ptr->extrinsic_.push_back(extrinsic);
					}
					char buffer[DEFAULT_IO_BUFFER_SIZE];
					if (recording_depth) {
						sprintf(buffer,
								recording_depth_filename_format_.c_str(),
								recording_file_index_);
						CaptureDepthImage(recording_depth_basedir_ +
								std::string(buffer), false);
					} else {
						sprintf(buffer,
								recording_image_filename_format_.c_str(),
								recording_file_index_);
						CaptureScreenImage(recording_image_basedir_ +
								std::string(buffer), false);
					}
				}
				view_control.Step(1.0);
				AdvanceConsoleProgress();
				if (view_control.IsPlayingEnd(recording_file_index_)) {
					view_control.SetAnimationMode(
							ViewControlWithCustomAnimation::
							AnimationMode::FreeMode);
					RegisterAnimationCallback(nullptr);
					if (recording && recording_trajectory) {
						if (recording_depth) {
							WriteIJsonConvertible(recording_depth_basedir_ +
									recording_depth_trajectory_filename_,
									*trajectory_ptr);
						} else {
							WriteIJsonConvertible(recording_image_basedir_ +
									recording_image_trajectory_filename_,
									*trajectory_ptr);
						}
					}
					if (close_window_when_animation_ends) {
						Close();
					}
				}
				UpdateWindowTitle();
				return false;
			});
}

bool VisualizerWithCustomAnimation::InitViewControl()
{
	view_control_ptr_ = std::unique_ptr<ViewControlWithCustomAnimation>(
			new ViewControlWithCustomAnimation);
	ResetViewPoint();
	return true;
}

void VisualizerWithCustomAnimation::KeyPressCallback(GLFWwindow *window,
		int key, int scancode, int action, int mods)
{
	auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
	if (action == GLFW_RELEASE || view_control.IsPlaying()) {
		return;
	}

	if (mods & GLFW_MOD_CONTROL) {
		switch (key) {
		case GLFW_KEY_F:
			view_control.SetAnimationMode(
					ViewControlWithCustomAnimation::AnimationMode::FreeMode);
			PrintDebug("[Visualizer] Enter freeview (editing) mode.\n");
			break;
		case GLFW_KEY_W:
			view_control.SetAnimationMode(
					ViewControlWithCustomAnimation::AnimationMode::PreviewMode);
			PrintDebug("[Visualizer] Enter preview mode.\n");
			break;
		case GLFW_KEY_P:
			Play(false);
			break;
		case GLFW_KEY_R:
			Play(true, false);
			break;
		case GLFW_KEY_G:
			Play(true, true);
			break;
		case GLFW_KEY_S:
			view_control.CaptureTrajectory();
			break;
		case GLFW_KEY_LEFT:
			view_control.Step(-1.0);
			break;
		case GLFW_KEY_RIGHT:
			view_control.Step(1.0);
			break;
		case GLFW_KEY_LEFT_BRACKET:
			view_control.GoToFirst();
			break;
		case GLFW_KEY_RIGHT_BRACKET:
			view_control.GoToLast();
			break;
		case GLFW_KEY_EQUAL:
			view_control.ChangeTrajectoryInterval(1);
			PrintDebug("[Visualizer] Trajectory interval set to %d.\n",
				view_control.GetTrajectoryInterval());
			break;
		case GLFW_KEY_MINUS:
			view_control.ChangeTrajectoryInterval(-1);
			PrintDebug("[Visualizer] Trajectory interval set to %d.\n",
				view_control.GetTrajectoryInterval());
			break;
		case GLFW_KEY_L:
			view_control.ToggleTrajectoryLoop();
			break;
		case GLFW_KEY_A:
			view_control.AddKeyFrame();
			PrintDebug("[Visualizer] Insert key frame; %d remaining.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_U:
			view_control.UpdateKeyFrame();
			PrintDebug("[Visualizer] Update key frame; %d remaining.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_D:
			view_control.DeleteKeyFrame();
			PrintDebug("[Visualizer] Delete last key frame; %d remaining.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_N:
			view_control.AddSpinKeyFrames();
			PrintDebug("[Visualizer] Insert spin key frames; %d remaining.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_E:
			view_control.ClearAllKeyFrames();
			PrintDebug("[Visualizer] Clear key frames; %d remaining.\n",
					view_control.NumOfKeyFrames());
			break;
		default:
			Visualizer::KeyPressCallback(window, key, scancode, action, mods);
			break;
		}
		is_redraw_required_ = true;
		UpdateWindowTitle();
	} else {
		Visualizer::KeyPressCallback(window, key, scancode, action, mods);
	}
}

void VisualizerWithCustomAnimation::MouseMoveCallback(GLFWwindow* window,
		double x, double y)
{
	auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
	if (view_control.IsPreviewing()) {
	} else if (view_control.IsPlaying()) {
	} else {
		Visualizer::MouseMoveCallback(window, x, y);
	}
}

void VisualizerWithCustomAnimation::MouseScrollCallback(GLFWwindow* window,
		double x, double y)
{
	auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
	if (view_control.IsPreviewing()) {
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
				glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
			view_control.Step(y);
			is_redraw_required_ = true;
			UpdateWindowTitle();
		}
	} else if (view_control.IsPlaying()) {
	} else {
		Visualizer::MouseScrollCallback(window, x, y);
	}
}

void VisualizerWithCustomAnimation::MouseButtonCallback(GLFWwindow* window,
		int button, int action, int mods)
{
	auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
	if (view_control.IsPreviewing()) {
	} else if (view_control.IsPlaying()) {
	} else {
		Visualizer::MouseButtonCallback(window, button, action, mods);
	}
}

}	// namespace three
