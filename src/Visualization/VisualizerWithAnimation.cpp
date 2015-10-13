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
	PrintInfo("    Alt + F      : Switch between free view mode (default) and animation mode.\n");
	PrintInfo("    Alt + P      : Enter animation mode and play animation from beginning.\n");
	PrintInfo("    Alt + R      : Enter animation mode, play animation, and record screen.\n");
	PrintInfo("    Alt + +/-    : Increase/decrease interval between key frames.\n");
	PrintInfo("    Alt + L      : Turn on/off camera path as a loop.\n");
	PrintInfo("    Alt + S      : Save the camera path into a json file.\n");
	PrintInfo("\n");
	PrintInfo("    -- In free view mode --\n");
	PrintInfo("    Alt + A      : Add current camera pose to the camera path.\n");
	PrintInfo("    Alt + D      : Delete the last camera pose in the camera path.\n");
	PrintInfo("    Alt + N      : Insert 360 spin into the camera path.\n");
	PrintInfo("    Alt + C      : Clear the camera path.\n");
	PrintInfo("\n");
	PrintInfo("    -- In animation mode --\n");
	PrintInfo("    Alt + <-/->  : Go backward/forward a frame.\n");
	PrintInfo("    Alt + [/]    : Go to beginning/end of the camera path.\n");
	PrintInfo("\n");
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

	if (mods & GLFW_MOD_ALT) {
		auto &view_control = (ViewControlWithAnimation &)(*view_control_ptr_);

		switch (key) {
		case GLFW_KEY_F:
			break;
		case GLFW_KEY_P:
			break;
		case GLFW_KEY_R:
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
			break;
		case GLFW_KEY_S:
			break;
		case GLFW_KEY_A:
			view_control.AddLastKeyFrame();
			PrintDebug("[Visualizer] Add key frame, # is %d.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_D:
			view_control.DeleteLastKeyFrame();
			PrintDebug("[Visualizer] Delete last key frame, # is %d.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_N:
			view_control.AddSpinKeyFrames();
			PrintDebug("[Visualizer] Add spin key frames, # is %d.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_C:
			view_control.ClearAllKeyFrames();
			PrintDebug("[Visualizer] Clear key frames, # is %d.\n",
					view_control.NumOfKeyFrames());
			break;
		case GLFW_KEY_LEFT:
			break;
		case GLFW_KEY_RIGHT:
			break;
		case GLFW_KEY_LEFT_BRACKET:
			break;
		case GLFW_KEY_RIGHT_BRACKET:
			break;
		}

		is_redraw_required_ = true;
	} else {
		Visualizer::KeyPressCallback(window, key, scancode, action, mods);
	}
}

}	// namespace three
