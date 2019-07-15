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

#include <cstdio>

#include "Open3D/Open3D.h"
#include "Open3D/Registration/GlobalOptimization.h"
#include "Open3D/Registration/PoseGraph.h"

using namespace open3d;

int main(int argc, char **argv) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 2) {
        PrintOpen3DVersion();
        // clang-format off
        utility::LogInfo("Usage:\n");
        utility::LogInfo("    > PoseGraph [posegraph_for_optimization].json\n");
        utility::LogInfo("    The program will :\n");
        utility::LogInfo("    1) Generate random PoseGraph\n");
        utility::LogInfo("    2) Save random PoseGraph as test_pose_graph.json\n");
        utility::LogInfo("    3) Reads PoseGraph from test_pose_graph.json\n");
        utility::LogInfo("    4) Save loaded PoseGraph as test_pose_graph_copy.json\n");
        utility::LogInfo("    5) Load PoseGraph from [posegraph_for_optimization].json\n");
        utility::LogInfo("    6) Optimize PoseGraph\n");
        utility::LogInfo("    7) Save PoseGraph to pose_graph_optimized.json\n");
        // clang-format on
        return 1;
    }

    // test posegraph read and write
    registration::PoseGraph pose_graph_test;
    pose_graph_test.nodes_.push_back(
            registration::PoseGraphNode(Eigen::Matrix4d::Random()));
    pose_graph_test.nodes_.push_back(
            registration::PoseGraphNode(Eigen::Matrix4d::Random()));
    pose_graph_test.edges_.push_back(
            registration::PoseGraphEdge(0, 1, Eigen::Matrix4d::Random(),
                                        Eigen::Matrix6d::Random(), false, 1.0));
    pose_graph_test.edges_.push_back(
            registration::PoseGraphEdge(0, 2, Eigen::Matrix4d::Random(),
                                        Eigen::Matrix6d::Random(), true, 0.2));
    io::WritePoseGraph("test_pose_graph.json", pose_graph_test);
    registration::PoseGraph pose_graph;
    io::ReadPoseGraph("test_pose_graph.json", pose_graph);
    io::WritePoseGraph("test_pose_graph_copy.json", pose_graph);

    // testing posegraph optimization
    auto pose_graph_input = io::CreatePoseGraphFromFile(argv[1]);
    registration::GlobalOptimizationConvergenceCriteria criteria;
    registration::GlobalOptimizationOption option;
    registration::GlobalOptimizationLevenbergMarquardt optimization_method;
    registration::GlobalOptimization(*pose_graph_input, optimization_method,
                                     criteria, option);
    auto pose_graph_input_prunned =
            registration::CreatePoseGraphWithoutInvalidEdges(*pose_graph_input,
                                                             option);
    io::WritePoseGraph("pose_graph_optimized.json", *pose_graph_input_prunned);

    return 0;
}
