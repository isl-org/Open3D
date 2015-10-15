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

#include "VisualizerWithAnimation.h"
#include "ViewControlWithAnimation.h"

namespace three{

VisualizerWithAnimation::VisualizerWithAnimation()
{
}

VisualizerWithAnimation::~VisualizerWithAnimation()
{
}

void VisualizerWithAnimation::PrintVisualizerHelp()
{
	Visualizer::PrintVisualizerHelp();
	PrintInfo("  -- Animation control --\n");
	PrintInfo("    Ctrl + F     : Switch between free view (edit) mode and preview mode.\n");
	PrintInfo("    Ctrl + P     : Enter animation mode and play animation from beginning.\n");
	PrintInfo("    Ctrl + R     : Enter animation mode, play animation, and record screen.\n");
	PrintInfo("    Ctrl + S     : Save the camera path into a json file.\n");
	PrintInfo("\n");
	PrintInfo("    -- In free view mode --\n");
	PrintInfo("    Ctrl + <-/-> : Go backward/forward a keyframe.\n");
	PrintInfo("    Ctrl + Wheel : Same as Alt + <-/->.\n");
	PrintInfo("    Ctrl + [/]   : Go to the first/last keyframe.\n");
	PrintInfo("    Ctrl + +/-   : Increase/decrease interval between keyframes.\n");
	PrintInfo("    Ctrl + L     : Turn on/off camera path as a loop.\n");
	PrintInfo("    Ctrl + A     : Add a keyframe right after the current keyframe.\n");
	PrintInfo("    Ctrl + U     : Update the current keyframe.\n");
	PrintInfo("    Ctrl + D     : Delete the current keyframe.\n");
	PrintInfo("    Ctrl + N     : Add 360 spin right after the current keyframe.\n");
	PrintInfo("    Ctrl + C     : Clear the entire camera path.\n");
	PrintInfo("\n");
	PrintInfo("    -- In preview mode --\n");
	PrintInfo("    Ctrl + <-/-> : Go backward/forward a frame.\n");
	PrintInfo("    Ctrl + Wheel : Same as Alt + <-/->.\n");
	PrintInfo("    Ctrl + [/]   : Go to beginning/end of the camera path.\n");
	PrintInfo("\n");
}

void VisualizerWithAnimation::UpdateWindowTitle()
{
	if (window_ != NULL) {
		auto &view_control = (ViewControlWithAnimation &)(*view_control_ptr_);
		std::string new_window_title = window_name_ + " - " + 
				view_control.GetStatusString();
		glfwSetWindowTitle(window_, new_window_title.c_str());
	}
}

bool VisualizerWithAnimation::InitViewControl()
{
	view_control_ptr_ = std::unique_ptr<ViewControlWithAnimation>(
			new ViewControlWithAnimation);
	ResetViewPoint();
	return true;
}

void VisualizerWithAnimation::KeyPressCallback(GLFWwindow *window,
		int key, int scancode, int action, int mods)
{
	if (action == GLFW_RELEASE) {
		return;
	}

	if (mods & GLFW_MOD_CONTROL) {
		auto &view_control = (ViewControlWithAnimation &)(*view_control_ptr_);

		switch (key) {
		case GLFW_KEY_F:
			break;
		case GLFW_KEY_P:
			break;
		case GLFW_KEY_R:
			break;
		case GLFW_KEY_S:
			break;
		case GLFW_KEY_LEFT:
			view_control.Step(-1.0);
			break;
		case GLFW_KEY_RIGHT:
			view_control.Step(1.0);
			break;
		case GLFW_KEY_LEFT_BRACKET:
			break;
		case GLFW_KEY_RIGHT_BRACKET:
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
		case GLFW_KEY_C:
			view_control.ClearAllKeyFrames();
			PrintDebug("[Visualizer] Clear key frames; %d remaining.\n",
					view_control.NumOfKeyFrames());
			break;
		}
		is_redraw_required_ = true;
		UpdateWindowTitle();
	} else {
		Visualizer::KeyPressCallback(window, key, scancode, action, mods);
	}
}

void VisualizerWithAnimation::MouseScrollCallback(GLFWwindow* window, 
		double x, double y)
{
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
			glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
		auto &view_control = (ViewControlWithAnimation &)(*view_control_ptr_);
		view_control.Step(y);
		is_redraw_required_ = true;
		UpdateWindowTitle();
	} else {
		Visualizer::MouseScrollCallback(window, x, y);
	}
}

}	// namespace three
