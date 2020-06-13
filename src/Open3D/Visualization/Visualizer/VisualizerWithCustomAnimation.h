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

#pragma once

#include "Open3D/Visualization/Visualizer/Visualizer.h"

namespace open3d {
namespace visualization {

class VisualizerWithCustomAnimation : public Visualizer {
public:
    VisualizerWithCustomAnimation();
    ~VisualizerWithCustomAnimation() override;
    VisualizerWithCustomAnimation(const VisualizerWithCustomAnimation &) =
            delete;
    VisualizerWithCustomAnimation &operator=(
            const VisualizerWithCustomAnimation &) = delete;

public:
    void PrintVisualizerHelp() override;
    void UpdateWindowTitle() override;
    void Play(bool recording = false,
              bool recording_depth = false,
              bool close_window_when_animation_ends = false);
    void RegisterRecordingImageFormat(const std::string &basedir,
                                      const std::string &format,
                                      const std::string &trajectory) {
        recording_image_basedir_ = basedir;
        recording_image_filename_format_ = format;
        recording_image_trajectory_filename_ = trajectory;
    }
    void RegisterRecordingDepthFormat(const std::string &basedir,
                                      const std::string &format,
                                      const std::string &trajectory) {
        recording_depth_basedir_ = basedir;
        recording_depth_filename_format_ = format;
        recording_depth_trajectory_filename_ = trajectory;
    }

protected:
    bool InitViewControl() override;
    void MouseMoveCallback(GLFWwindow *window, double x, double y) override;
    void MouseScrollCallback(GLFWwindow *window, double x, double y) override;
    void MouseButtonCallback(GLFWwindow *window,
                             int button,
                             int action,
                             int mods) override;
    void KeyPressCallback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods) override;

protected:
    std::string recording_image_basedir_ = "image/";
    std::string recording_image_filename_format_ = "image_{:06d}.png";
    std::string recording_image_trajectory_filename_ = "image_trajectory.json";
    std::string recording_depth_basedir_ = "depth/";
    std::string recording_depth_filename_format_ = "depth_{:06d}.png";
    std::string recording_depth_trajectory_filename_ = "depth_trajectory.json";
    size_t recording_file_index_ = 0;
};

}  // namespace visualization
}  // namespace open3d
