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

#include "Open3D/Open3D.h"
#include "Tools/ManuallyAlignPointCloud/AlignmentSession.h"

namespace open3d {

class VisualizerForAlignment : public visualization::Visualizer {
public:
    VisualizerForAlignment(visualization::VisualizerWithEditing &source,
                           visualization::VisualizerWithEditing &target,
                           double voxel_size = -1.0,
                           double max_correspondence_distance = -1.0,
                           bool with_scaling = true,
                           bool use_dialog = true,
                           const std::string &polygon_filename = "",
                           const std::string &directory = "")
        : source_visualizer_(source),
          target_visualizer_(target),
          voxel_size_(voxel_size),
          max_correspondence_distance_(max_correspondence_distance),
          with_scaling_(with_scaling),
          use_dialog_(use_dialog),
          polygon_filename_(polygon_filename),
          default_directory_(directory) {}
    ~VisualizerForAlignment() override {}

public:
    void PrintVisualizerHelp() override;
    bool AddSourceAndTarget(std::shared_ptr<geometry::PointCloud> source,
                            std::shared_ptr<geometry::PointCloud> target);

protected:
    void KeyPressCallback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods) override;
    bool SaveSessionToFile(const std::string &filename);
    bool LoadSessionFromFile(const std::string &filename);
    bool AlignWithManualAnnotation();
    void PrintTransformation();
    void EvaluateAlignmentAndSave(const std::string &filename);

protected:
    visualization::VisualizerWithEditing &source_visualizer_;
    visualization::VisualizerWithEditing &target_visualizer_;
    double voxel_size_ = -1.0;
    double max_correspondence_distance_ = -1.0;
    bool with_scaling_ = true;
    bool use_dialog_ = true;
    Eigen::Matrix4d transformation_ = Eigen::Matrix4d::Identity();
    std::string polygon_filename_ = "";
    std::shared_ptr<geometry::PointCloud> source_copy_ptr_;
    std::shared_ptr<geometry::PointCloud> target_copy_ptr_;
    AlignmentSession alignment_session_;
    std::string default_directory_;
};

}  // namespace open3d
