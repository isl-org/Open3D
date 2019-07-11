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

#include "Open3D/Visualization/Visualizer/VisualizerWithCustomAnimation.h"

#include <thread>

#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "Open3D/IO/ClassIO/IJsonConvertibleIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithCustomAnimation.h"

namespace open3d {
namespace visualization {

VisualizerWithCustomAnimation::VisualizerWithCustomAnimation() {}

VisualizerWithCustomAnimation::~VisualizerWithCustomAnimation() {}

void VisualizerWithCustomAnimation::PrintVisualizerHelp() {
    Visualizer::PrintVisualizerHelp();
    // clang-format off
    utility::LogInfo("  -- Animation control --\n");
    utility::LogInfo("    Ctrl + F     : Enter freeview (editing) mode.\n");
    utility::LogInfo("    Ctrl + W     : Enter preview mode.\n");
    utility::LogInfo("    Ctrl + P     : Enter animation mode and play animation from beginning.\n");
    utility::LogInfo("    Ctrl + R     : Enter animation mode, play animation, and record screen.\n");
    utility::LogInfo("    Ctrl + G     : Enter animation mode, play animation, and record depth.\n");
    utility::LogInfo("    Ctrl + S     : Save the camera path into a json file.\n");
    utility::LogInfo("\n");
    utility::LogInfo("    -- In free view mode --\n");
    utility::LogInfo("    Ctrl + <-/-> : Go backward/forward a keyframe.\n");
    utility::LogInfo("    Ctrl + Wheel : Same as Ctrl + <-/->.\n");
    utility::LogInfo("    Ctrl + [/]   : Go to the first/last keyframe.\n");
    utility::LogInfo("    Ctrl + +/-   : Increase/decrease interval between keyframes.\n");
    utility::LogInfo("    Ctrl + L     : Turn on/off camera path as a loop.\n");
    utility::LogInfo("    Ctrl + A     : Add a keyframe right after the current keyframe.\n");
    utility::LogInfo("    Ctrl + U     : Update the current keyframe.\n");
    utility::LogInfo("    Ctrl + D     : Delete the current keyframe.\n");
    utility::LogInfo("    Ctrl + N     : Add 360 spin right after the current keyframe.\n");
    utility::LogInfo("    Ctrl + E     : Erase the entire camera path.\n");
    utility::LogInfo("\n");
    utility::LogInfo("    -- In preview mode --\n");
    utility::LogInfo("    Ctrl + <-/-> : Go backward/forward a frame.\n");
    utility::LogInfo("    Ctrl + Wheel : Same as Ctrl + <-/->.\n");
    utility::LogInfo("    Ctrl + [/]   : Go to beginning/end of the camera path.\n");
    utility::LogInfo("\n");
    // clang-format on
}

void VisualizerWithCustomAnimation::UpdateWindowTitle() {
    if (window_ != NULL) {
        auto &view_control =
                (ViewControlWithCustomAnimation &)(*view_control_ptr_);
        std::string new_window_title =
                window_name_ + " - " + view_control.GetStatusString();
        glfwSetWindowTitle(window_, new_window_title.c_str());
    }
}

void VisualizerWithCustomAnimation::Play(
        bool recording /* = false*/,
        bool recording_depth /* = false*/,
        bool close_window_when_animation_ends /* = false*/) {
    auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
    if (view_control.NumOfFrames() == 0) {
        utility::LogWarning("Abort playing due to empty trajectory.\n");
        return;
    }
    view_control.SetAnimationMode(
            ViewControlWithCustomAnimation::AnimationMode::PlayMode);
    is_redraw_required_ = true;
    UpdateWindowTitle();
    recording_file_index_ = 0;
    utility::ConsoleProgressBar progress_bar(view_control.NumOfFrames(),
                                             "Play animation: ");
    auto trajectory_ptr = std::make_shared<camera::PinholeCameraTrajectory>();
    bool recording_trajectory = view_control.IsValidPinholeCameraTrajectory();
    if (recording) {
        if (recording_depth) {
            utility::filesystem::MakeDirectoryHierarchy(
                    recording_depth_basedir_);
        } else {
            utility::filesystem::MakeDirectoryHierarchy(
                    recording_image_basedir_);
        }
    }
    RegisterAnimationCallback([=, &progress_bar](Visualizer *vis) {
        // The lambda function captures no references to avoid dangling
        // references
        auto &view_control =
                (ViewControlWithCustomAnimation &)(*view_control_ptr_);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        recording_file_index_++;
        if (recording) {
            if (recording_trajectory) {
                auto parameter = camera::PinholeCameraParameters();
                view_control.ConvertToPinholeCameraParameters(parameter);
                trajectory_ptr->parameters_.push_back(parameter);
            }
            std::string buffer;
            if (recording_depth) {
                buffer = fmt::format(recording_depth_filename_format_.c_str(),
                                     recording_file_index_);
                CaptureDepthImage(
                        recording_depth_basedir_ + std::string(buffer), false);
            } else {
                buffer = fmt::format(recording_image_filename_format_.c_str(),
                                     recording_file_index_);
                CaptureScreenImage(
                        recording_image_basedir_ + std::string(buffer), false);
            }
        }
        view_control.Step(1.0);
        ++progress_bar;
        if (view_control.IsPlayingEnd(recording_file_index_)) {
            view_control.SetAnimationMode(
                    ViewControlWithCustomAnimation::AnimationMode::FreeMode);
            RegisterAnimationCallback(nullptr);
            if (recording && recording_trajectory) {
                if (recording_depth) {
                    io::WriteIJsonConvertible(
                            recording_depth_basedir_ +
                                    recording_depth_trajectory_filename_,
                            *trajectory_ptr);
                } else {
                    io::WriteIJsonConvertible(
                            recording_image_basedir_ +
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

bool VisualizerWithCustomAnimation::InitViewControl() {
    view_control_ptr_ = std::unique_ptr<ViewControlWithCustomAnimation>(
            new ViewControlWithCustomAnimation);
    ResetViewPoint();
    return true;
}

void VisualizerWithCustomAnimation::KeyPressCallback(
        GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
    if (action == GLFW_RELEASE || view_control.IsPlaying()) {
        return;
    }

    if (mods & GLFW_MOD_CONTROL) {
        switch (key) {
            case GLFW_KEY_F:
                view_control.SetAnimationMode(ViewControlWithCustomAnimation::
                                                      AnimationMode::FreeMode);
                utility::LogDebug(
                        "[Visualizer] Enter freeview (editing) mode.\n");
                break;
            case GLFW_KEY_W:
                view_control.SetAnimationMode(
                        ViewControlWithCustomAnimation::AnimationMode::
                                PreviewMode);
                utility::LogDebug("[Visualizer] Enter preview mode.\n");
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
                utility::LogDebug(
                        "[Visualizer] Trajectory interval set to {}.\n",
                        view_control.GetTrajectoryInterval());
                break;
            case GLFW_KEY_MINUS:
                view_control.ChangeTrajectoryInterval(-1);
                utility::LogDebug(
                        "[Visualizer] Trajectory interval set to {}.\n",
                        view_control.GetTrajectoryInterval());
                break;
            case GLFW_KEY_L:
                view_control.ToggleTrajectoryLoop();
                break;
            case GLFW_KEY_A:
                view_control.AddKeyFrame();
                utility::LogDebug(
                        "[Visualizer] Insert key frame; {} remaining.\n",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_U:
                view_control.UpdateKeyFrame();
                utility::LogDebug(
                        "[Visualizer] Update key frame; {} remaining.\n",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_D:
                view_control.DeleteKeyFrame();
                utility::LogDebug(
                        "[Visualizer] Delete last key frame; {} remaining.\n",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_N:
                view_control.AddSpinKeyFrames();
                utility::LogDebug(
                        "[Visualizer] Insert spin key frames; {} remaining.\n",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_E:
                view_control.ClearAllKeyFrames();
                utility::LogDebug(
                        "[Visualizer] Clear key frames; {} remaining.\n",
                        view_control.NumOfKeyFrames());
                break;
            default:
                Visualizer::KeyPressCallback(window, key, scancode, action,
                                             mods);
                break;
        }
        is_redraw_required_ = true;
        UpdateWindowTitle();
    } else {
        Visualizer::KeyPressCallback(window, key, scancode, action, mods);
    }
}

void VisualizerWithCustomAnimation::MouseMoveCallback(GLFWwindow *window,
                                                      double x,
                                                      double y) {
    auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
    if (view_control.IsPreviewing()) {
    } else if (view_control.IsPlaying()) {
    } else {
        Visualizer::MouseMoveCallback(window, x, y);
    }
}

void VisualizerWithCustomAnimation::MouseScrollCallback(GLFWwindow *window,
                                                        double x,
                                                        double y) {
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

void VisualizerWithCustomAnimation::MouseButtonCallback(GLFWwindow *window,
                                                        int button,
                                                        int action,
                                                        int mods) {
    auto &view_control = (ViewControlWithCustomAnimation &)(*view_control_ptr_);
    if (view_control.IsPreviewing()) {
    } else if (view_control.IsPlaying()) {
    } else {
        Visualizer::MouseButtonCallback(window, button, action, mods);
    }
}

}  // namespace visualization
}  // namespace open3d
