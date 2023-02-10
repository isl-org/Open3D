// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <chrono>
#include <functional>
#include <memory>

#include "open3d/Open3D.h"

using namespace open3d;

bool f_traverse(const std::shared_ptr<geometry::OctreeNode>& node,
                const std::shared_ptr<geometry::OctreeNodeInfo>& node_info) {
    if (auto internal_node =
                std::dynamic_pointer_cast<geometry::OctreeInternalNode>(node)) {
        if (auto internal_point_node = std::dynamic_pointer_cast<
                    geometry::OctreeInternalPointNode>(internal_node)) {
            int num_children = 0;
            for (const auto& c : internal_point_node->children_) {
                if (c) num_children++;
            }
            utility::LogInfo(
                    "Internal node at depth {} with origin {} has "
                    "{} children and {} points",
                    node_info->depth_, node_info->origin_, num_children,
                    internal_point_node->indices_.size());
        }
    } else if (auto leaf_node = std::dynamic_pointer_cast<
                       geometry::OctreePointColorLeafNode>(node)) {
        utility::LogInfo(
                "Node at depth {} with origin {} has"
                "color {} and {} points",
                node_info->depth_, node_info->origin_, leaf_node->color_,
                leaf_node->indices_.size());
        // utility::LogInfo("Indices: {}", leaf_node->indices_);
    } else {
        utility::LogError("Unknown node type");
    }

    return false;
}

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Octree [pointcloud_filename]");
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

    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    constexpr int max_depth = 3;
    auto octree = std::make_shared<geometry::Octree>(max_depth);
    octree->ConvertFromPointCloud(*pcd);

    octree->Traverse(f_traverse);

    std::cout << std::endl << std::endl;
    auto start = std::chrono::steady_clock::now();
    auto result = octree->LocateLeafNode(Eigen::Vector3d::Zero());
    auto end = std::chrono::steady_clock::now();
    utility::LogInfo(
            "Located in {} usec",
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                    .count());
    if (auto point_node =
                std::dynamic_pointer_cast<geometry::OctreePointColorLeafNode>(
                        result.first)) {
        utility::LogInfo(
                "Found leaf node at depth {} with origin {} and {} indices",
                result.second->depth_, result.second->origin_,
                point_node->indices_.size());
    }
    std::cout << std::endl << std::endl;

    visualization::DrawGeometries({pcd, octree});
}
