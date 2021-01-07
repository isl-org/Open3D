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

#include <string>
#include <vector>

#include "open3d/pipelines/registration/PoseGraph.h"
#include "open3d/t/pipelines/slac/ControlGrid.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

using PoseGraph = open3d::pipelines::registration::PoseGraph;
/// Simultaneous Localization and Calibration: Self-Calibration of Consumer
/// Depth Cameras, CVPR 2014 Qian-Yi Zhou and Vladlen Koltun
struct SLACOptimizerOption {
    int max_iterations_ = 10;
    float voxel_size_ = 0.05;
    bool correspondence_debug_ = false;
    bool grid_debug_ = false;
    std::string device_ = "CPU:0";
    std::string buffer_folder_ = "";
};

/// Estimate a shared control grid for all fragments for scene reconstruction,
/// implemented in https://github.com/qianyizh/ElasticReconstruction.
std::pair<PoseGraph, ControlGrid> RunSLACOptimizerForFragments(
        const std::vector<std::string>& fragment_fnames,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerOption& option);

PoseGraph RunRigidOptimizerForFragments(
        const std::vector<std::string>& fragment_fnames,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerOption& option);

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
