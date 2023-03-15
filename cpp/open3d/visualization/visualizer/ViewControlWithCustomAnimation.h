// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/visualization/visualizer/ViewControl.h"
#include "open3d/visualization/visualizer/ViewTrajectory.h"

namespace open3d {
namespace visualization {

class ViewControlWithCustomAnimation : public ViewControl {
public:
    enum AnimationMode {
        FreeMode = 0,
        PreviewMode = 1,
        PlayMode = 2,
    };

public:
    virtual ~ViewControlWithCustomAnimation() {}

    void Reset() override;
    void ChangeFieldOfView(double step) override;
    void Scale(double scale) override;
    void Rotate(double x, double y, double xo, double yo) override;
    void Translate(double x, double y, double xo, double yo) override;

    void SetAnimationMode(AnimationMode mode);
    void AddKeyFrame();
    void UpdateKeyFrame();
    void DeleteKeyFrame();
    void AddSpinKeyFrames(int num_of_key_frames = 20);
    void ClearAllKeyFrames() { view_trajectory_.view_status_.clear(); }
    size_t NumOfKeyFrames() const {
        return view_trajectory_.view_status_.size();
    }
    size_t NumOfFrames() const { return view_trajectory_.NumOfFrames(); }
    void ToggleTrajectoryLoop() {
        if (animation_mode_ == AnimationMode::FreeMode) {
            view_trajectory_.is_loop_ = !view_trajectory_.is_loop_;
        }
    }
    void ChangeTrajectoryInterval(int change) {
        if (animation_mode_ == AnimationMode::FreeMode) {
            view_trajectory_.ChangeInterval(change);
        }
    }
    int GetTrajectoryInterval() const { return view_trajectory_.interval_; }
    std::string GetStatusString() const;
    void Step(double change);
    void GoToFirst();
    void GoToLast();
    bool CaptureTrajectory(const std::string &filename = "");
    bool LoadTrajectoryFromJsonFile(const std::string &filename);
    bool LoadTrajectoryFromCameraTrajectory(
            const camera::PinholeCameraTrajectory &camera_trajectory);
    bool IsPreviewing() {
        return animation_mode_ == AnimationMode::PreviewMode;
    }
    bool IsPlaying() { return animation_mode_ == AnimationMode::PlayMode; }
    bool IsPlayingEnd(size_t num) {
        return (IsPlaying() && num >= view_trajectory_.NumOfFrames());
    }
    bool IsValidPinholeCameraTrajectory() const;

protected:
    size_t CurrentFrame() const { return (size_t)round(current_frame_); }
    size_t CurrentKeyframe() const { return (size_t)round(current_keyframe_); }
    double RegularizeFrameIndex(double current_frame,
                                size_t num_of_frames,
                                bool is_loop);
    void SetViewControlFromTrajectory();

protected:
    AnimationMode animation_mode_ = AnimationMode::FreeMode;
    ViewTrajectory view_trajectory_;
    double current_frame_ = 0.0;
    double current_keyframe_ = 0.0;
};

}  // namespace visualization
}  // namespace open3d
