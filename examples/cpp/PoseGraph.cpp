// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/PoseGraph.h"

#include <cstdio>

#include "open3d/Open3D.h"
#include "open3d/pipelines/registration/GlobalOptimization.h"

using namespace open3d;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > PoseGraph [posegraph_for_optimization].json");
    utility::LogInfo("      The program will :");
    utility::LogInfo("      1) Generate random PoseGraph");
    utility::LogInfo("      2) Save random PoseGraph as test_pose_graph.json");
    utility::LogInfo("      3) Reads PoseGraph from test_pose_graph.json");
    utility::LogInfo("      4) Save loaded PoseGraph as test_pose_graph_copy.json");
    utility::LogInfo("      5) Load PoseGraph from [posegraph_for_optimization].json");
    utility::LogInfo("      6) Optimize PoseGraph");
    utility::LogInfo("      7) Save PoseGraph to pose_graph_optimized.json");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    // test posegraph read and write
    pipelines::registration::PoseGraph pose_graph_test;
    pose_graph_test.nodes_.push_back(
            pipelines::registration::PoseGraphNode(Eigen::Matrix4d::Random()));
    pose_graph_test.nodes_.push_back(
            pipelines::registration::PoseGraphNode(Eigen::Matrix4d::Random()));
    pose_graph_test.edges_.push_back(pipelines::registration::PoseGraphEdge(
            0, 1, Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), false,
            1.0));
    pose_graph_test.edges_.push_back(pipelines::registration::PoseGraphEdge(
            0, 2, Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), true,
            0.2));
    io::WritePoseGraph("test_pose_graph.json", pose_graph_test);
    pipelines::registration::PoseGraph pose_graph;
    io::ReadPoseGraph("test_pose_graph.json", pose_graph);
    io::WritePoseGraph("test_pose_graph_copy.json", pose_graph);

    // testing posegraph optimization
    auto pose_graph_input = io::CreatePoseGraphFromFile(argv[1]);
    pipelines::registration::GlobalOptimizationConvergenceCriteria criteria;
    pipelines::registration::GlobalOptimizationOption option;
    pipelines::registration::GlobalOptimizationLevenbergMarquardt
            optimization_method;
    pipelines::registration::GlobalOptimization(
            *pose_graph_input, optimization_method, criteria, option);
    auto pose_graph_input_prunned =
            pipelines::registration::CreatePoseGraphWithoutInvalidEdges(
                    *pose_graph_input, option);
    io::WritePoseGraph("pose_graph_optimized.json", *pose_graph_input_prunned);

    return 0;
}
