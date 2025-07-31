// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/VisualizerWithCustomAnimation.h"

#include <thread>

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressBar.h"
#include "open3d/visualization/visualizer/ViewControlWithCustomAnimation.h"

namespace open3d {
namespace visualization {

VisualizerWithCustomAnimation::VisualizerWithCustomAnimation() {}

VisualizerWithCustomAnimation::~VisualizerWithCustomAnimation() {}

void VisualizerWithCustomAnimation::PrintVisualizerHelp() {
    Visualizer::PrintVisualizerHelp();
    // clang-format off
    utility::LogInfo("  -- Animation control --");
    utility::LogInfo("    Ctrl + F     : Enter freeview (editing) mode.");
    utility::LogInfo("    Ctrl + W     : Enter preview mode.");
    utility::LogInfo("    Ctrl + P     : Enter animation mode and play animation from beginning.");
    utility::LogInfo("    Ctrl + R     : Enter animation mode, play animation, and record screen.");
    utility::LogInfo("    Ctrl + G     : Enter animation mode, play animation, and record depth.");
    utility::LogInfo("    Ctrl + S     : Save the camera path into a json file.");
    utility::LogInfo("");
    utility::LogInfo("    -- In free view mode --");
    utility::LogInfo("    Ctrl + <-/-> : Go backward/forward a keyframe.");
    utility::LogInfo("    Ctrl + Wheel : Same as Ctrl + <-/->.");
    utility::LogInfo("    Ctrl + [/]   : Go to the first/last keyframe.");
    utility::LogInfo("    Ctrl + +/-   : Increase/decrease interval between keyframes.");
    utility::LogInfo("    Ctrl + L     : Turn on/off camera path as a loop.");
    utility::LogInfo("    Ctrl + A     : Add a keyframe right after the current keyframe.");
    utility::LogInfo("    Ctrl + U     : Update the current keyframe.");
    utility::LogInfo("    Ctrl + D     : Delete the current keyframe.");
    utility::LogInfo("    Ctrl + N     : Add 360 spin right after the current keyframe.");
    utility::LogInfo("    Ctrl + E     : Erase the entire camera path.");
    utility::LogInfo("");
    utility::LogInfo("    -- In preview mode --");
    utility::LogInfo("    Ctrl + <-/-> : Go backward/forward a frame.");
    utility::LogInfo("    Ctrl + Wheel : Same as Ctrl + <-/->.");
    utility::LogInfo("    Ctrl + [/]   : Go to beginning/end of the camera path.");
    utility::LogInfo("");
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
        utility::LogWarning("Abort playing due to empty trajectory.");
        return;
    }
    view_control.SetAnimationMode(
            ViewControlWithCustomAnimation::AnimationMode::PlayMode);
    is_redraw_required_ = true;
    UpdateWindowTitle();
    recording_file_index_ = 0;
    auto progress_bar_ptr = std::make_shared<utility::ProgressBar>(
            view_control.NumOfFrames(), "Play animation: ");
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
    RegisterAnimationCallback([this, recording, recording_depth,
                               close_window_when_animation_ends,
                               recording_trajectory, trajectory_ptr,
                               progress_bar_ptr](Visualizer *vis) {
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
            if (recording_depth) {
                auto buffer = fmt::format(
                        fmt::runtime(recording_depth_basedir_ +
                                     recording_depth_filename_format_),
                        recording_file_index_);
                CaptureDepthImage(buffer, false);
            } else {
                auto buffer = fmt::format(
                        fmt::runtime(recording_depth_basedir_ +
                                     recording_image_filename_format_),
                        recording_file_index_);
                CaptureScreenImage(buffer, false);
            }
        }
        view_control.Step(1.0);
        ++(*progress_bar_ptr);
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
                        "[Visualizer] Enter freeview (editing) mode.");
                break;
            case GLFW_KEY_W:
                view_control.SetAnimationMode(
                        ViewControlWithCustomAnimation::AnimationMode::
                                PreviewMode);
                utility::LogDebug("[Visualizer] Enter preview mode.");
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
                utility::LogDebug("[Visualizer] Trajectory interval set to {}.",
                                  view_control.GetTrajectoryInterval());
                break;
            case GLFW_KEY_MINUS:
                view_control.ChangeTrajectoryInterval(-1);
                utility::LogDebug("[Visualizer] Trajectory interval set to {}.",
                                  view_control.GetTrajectoryInterval());
                break;
            case GLFW_KEY_L:
                view_control.ToggleTrajectoryLoop();
                break;
            case GLFW_KEY_A:
                view_control.AddKeyFrame();
                utility::LogDebug(
                        "[Visualizer] Insert key frame; {} remaining.",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_U:
                view_control.UpdateKeyFrame();
                utility::LogDebug(
                        "[Visualizer] Update key frame; {} remaining.",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_D:
                view_control.DeleteKeyFrame();
                utility::LogDebug(
                        "[Visualizer] Delete last key frame; {} remaining.",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_N:
                view_control.AddSpinKeyFrames();
                utility::LogDebug(
                        "[Visualizer] Insert spin key frames; {} remaining.",
                        view_control.NumOfKeyFrames());
                break;
            case GLFW_KEY_E:
                view_control.ClearAllKeyFrames();
                utility::LogDebug(
                        "[Visualizer] Clear key frames; {} remaining.",
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
