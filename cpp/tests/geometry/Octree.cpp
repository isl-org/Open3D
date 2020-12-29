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

#include "open3d/geometry/Octree.h"

#include <json/json.h>

#include <iostream>
#include <memory>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(Octree, ConstructorWithoutSize) {
    geometry::Octree octree(10);
    ExpectEQ(octree.origin_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(octree.size_, 0);
}

TEST(Octree, ConstructorWithSize) {
    geometry::Octree octree(0, Eigen::Vector3d(-1, -1, -1), 2);
    ExpectEQ(octree.origin_, Eigen::Vector3d(-1, -1, -1));
    EXPECT_EQ(octree.size_, 2);
}

TEST(Octree, ZeroDepth) {
    geometry::Octree octree(0, Eigen::Vector3d(-1, -1, -1), 2);
    Eigen::Vector3d point(0, 0, 0);
    Eigen::Vector3d color(0, 0.1, 0.2);

    octree.InsertPoint(point, geometry::OctreeColorLeafNode::GetInitFunction(),
                       geometry::OctreeColorLeafNode::GetUpdateFunction(color));
    if (auto leaf_node =
                std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                        octree.root_node_)) {
        ExpectEQ(leaf_node->color_, color);
    } else {
        throw std::runtime_error("Leaf node must be OctreeColorLeafNode");
    }
}

TEST(Octree, ZeroDepthOutOfBound) {
    geometry::Octree octree(0, Eigen::Vector3d(-1, -1, -1), 2);

    Eigen::Vector3d point_out(10, 10, 10);  // Outside bound
    Eigen::Vector3d color_out(0, 0.1, 0.2);
    octree.InsertPoint(
            point_out, geometry::OctreeColorLeafNode::GetInitFunction(),
            geometry::OctreeColorLeafNode::GetUpdateFunction(color_out));

    Eigen::Vector3d point_in(0, 0, 0);  // Within bound
    Eigen::Vector3d color_in(0.1, 0.2, 0.3);
    octree.InsertPoint(
            point_in, geometry::OctreeColorLeafNode::GetInitFunction(),
            geometry::OctreeColorLeafNode::GetUpdateFunction(color_in));

    if (auto leaf_node =
                std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                        octree.root_node_)) {
        ExpectEQ(leaf_node->color_, color_in);
    } else {
        throw std::runtime_error("Leaf node must be OctreeColorLeafNode");
    }
}

TEST(Octree, ZeroDepthValueOverwrite) {
    geometry::Octree octree(0, Eigen::Vector3d(-1, -1, -1), 2);

    Eigen::Vector3d point_old(0, 0, 0);
    Eigen::Vector3d color_old(0.1, 0.2, 0.3);
    Eigen::Vector3d point_new(0.01, 0.01, 0.01);
    Eigen::Vector3d color_new(0.4, 0.5, 0.6);

    octree.InsertPoint(
            point_old, geometry::OctreeColorLeafNode::GetInitFunction(),
            geometry::OctreeColorLeafNode::GetUpdateFunction(color_old));
    if (auto leaf_node =
                std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                        octree.root_node_)) {
        ExpectEQ(leaf_node->color_, color_old);
    } else {
        throw std::runtime_error("Leaf node must be OctreeLeafNode");
    }

    octree.InsertPoint(
            point_new, geometry::OctreeColorLeafNode::GetInitFunction(),
            geometry::OctreeColorLeafNode::GetUpdateFunction(color_new));
    if (auto leaf_node =
                std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                        octree.root_node_)) {
        ExpectEQ(leaf_node->color_, color_new);
    } else {
        throw std::runtime_error("Leaf node must be OctreeLeafNode");
    }
}

TEST(Octree, EightCubes) {
    // Build octree
    std::vector<Eigen::Vector3d> points{
            Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(1.5, 0.5, 0.5),
            Eigen::Vector3d(0.5, 1.5, 0.5), Eigen::Vector3d(1.5, 1.5, 0.5),
            Eigen::Vector3d(0.5, 0.5, 1.5), Eigen::Vector3d(1.5, 0.5, 1.5),
            Eigen::Vector3d(0.5, 1.5, 1.5), Eigen::Vector3d(1.5, 1.5, 1.5),
    };
    std::vector<Eigen::Vector3d> colors{
            Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.1, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.1, 0.0), Eigen::Vector3d(0.1, 0.1, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.1), Eigen::Vector3d(0.1, 0.0, 0.1),
            Eigen::Vector3d(0.0, 0.1, 0.1), Eigen::Vector3d(0.1, 0.1, 0.1),
    };
    geometry::Octree octree(1, Eigen::Vector3d(0, 0, 0), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        octree.InsertPoint(
                points[i], geometry::OctreeColorLeafNode::GetInitFunction(),
                geometry::OctreeColorLeafNode::GetUpdateFunction(colors[i]));
    }

    // Check dimensions
    ExpectEQ(octree.origin_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(octree.size_, 2);

    // Check node values
    if (auto root_node =
                std::dynamic_pointer_cast<geometry::OctreeInternalNode>(
                        octree.root_node_)) {
        for (size_t i = 0; i < 8; ++i) {
            if (auto leaf_node = std::dynamic_pointer_cast<
                        geometry::OctreeColorLeafNode>(
                        root_node->children_[i])) {
                ExpectEQ(leaf_node->color_, colors[i]);
            } else {
                throw std::runtime_error(
                        "Leaf node must be OctreeColorLeafNode");
            };
        }
    } else {
        throw std::runtime_error("Root node must be OctreeInternalNode");
    }
}

TEST(Octree, EightCubesLeafPointIndices) {
    // Build octree
    std::vector<Eigen::Vector3d> points{
            Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(1.5, 0.5, 0.5),
            Eigen::Vector3d(0.5, 1.5, 0.5), Eigen::Vector3d(1.5, 1.5, 0.5),
            Eigen::Vector3d(0.5, 0.5, 1.5), Eigen::Vector3d(1.5, 0.5, 1.5),
            Eigen::Vector3d(0.5, 1.5, 1.5), Eigen::Vector3d(1.5, 1.5, 1.5),
    };
    std::vector<Eigen::Vector3d> colors{
            Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.1, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.1, 0.0), Eigen::Vector3d(0.1, 0.1, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.1), Eigen::Vector3d(0.1, 0.0, 0.1),
            Eigen::Vector3d(0.0, 0.1, 0.1), Eigen::Vector3d(0.1, 0.1, 0.1),
    };
    geometry::Octree octree(1, Eigen::Vector3d(0, 0, 0), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        octree.InsertPoint(
                points[i],
                geometry::OctreePointColorLeafNode::GetInitFunction(),
                geometry::OctreePointColorLeafNode::GetUpdateFunction(
                        i, colors[i]));
    }

    // Check dimensions
    ExpectEQ(octree.origin_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(octree.size_, 2);

    // Check node values
    if (auto root_node =
                std::dynamic_pointer_cast<geometry::OctreeInternalNode>(
                        octree.root_node_)) {
        for (size_t i = 0; i < 8; ++i) {
            if (auto leaf_node = std::dynamic_pointer_cast<
                        geometry::OctreePointColorLeafNode>(
                        root_node->children_[i])) {
                ExpectEQ(leaf_node->color_, colors[i]);
                EXPECT_EQ(leaf_node->indices_.size(), 1);
                EXPECT_EQ(leaf_node->indices_[0], i);
            } else {
                throw std::runtime_error(
                        "Leaf node must be OctreePointColorLeafNode");
            };
        }
    } else {
        throw std::runtime_error("Root node must be OctreeInternalNode");
    }
}

TEST(Octree, EightCubesLeafAndInternalPointIndices) {
    // Build octree
    std::vector<Eigen::Vector3d> points{
            Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(1.5, 0.5, 0.5),
            Eigen::Vector3d(0.5, 1.5, 0.5), Eigen::Vector3d(1.5, 1.5, 0.5),
            Eigen::Vector3d(0.5, 0.5, 1.5), Eigen::Vector3d(1.5, 0.5, 1.5),
            Eigen::Vector3d(0.5, 1.5, 1.5), Eigen::Vector3d(1.5, 1.5, 1.5),
    };
    std::vector<Eigen::Vector3d> colors{
            Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.1, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.1, 0.0), Eigen::Vector3d(0.1, 0.1, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.1), Eigen::Vector3d(0.1, 0.0, 0.1),
            Eigen::Vector3d(0.0, 0.1, 0.1), Eigen::Vector3d(0.1, 0.1, 0.1),
    };
    geometry::Octree octree(1, Eigen::Vector3d(0, 0, 0), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        octree.InsertPoint(
                points[i],
                geometry::OctreePointColorLeafNode::GetInitFunction(),
                geometry::OctreePointColorLeafNode::GetUpdateFunction(
                        i, colors[i]),
                geometry::OctreeInternalPointNode::GetInitFunction(),
                geometry::OctreeInternalPointNode::GetUpdateFunction(i));
    }

    // Check dimensions
    ExpectEQ(octree.origin_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(octree.size_, 2);

    // Check node values
    if (auto root_node =
                std::dynamic_pointer_cast<geometry::OctreeInternalPointNode>(
                        octree.root_node_)) {
        EXPECT_EQ(root_node->indices_.size(), 8);
        for (size_t i = 0; i < root_node->indices_.size(); i++) {
            EXPECT_EQ(root_node->indices_[i], i);
        }
        for (size_t i = 0; i < 8; ++i) {
            if (auto leaf_node = std::dynamic_pointer_cast<
                        geometry::OctreePointColorLeafNode>(
                        root_node->children_[i])) {
                ExpectEQ(leaf_node->color_, colors[i]);
                EXPECT_EQ(leaf_node->indices_.size(), 1);
                EXPECT_EQ(leaf_node->indices_[0], i);
            } else {
                throw std::runtime_error(
                        "Leaf node must be OctreePointColorLeafNode");
            };
        }
    } else {
        throw std::runtime_error("Root node must be OctreeInternalPointNode");
    }
}

TEST(Octree, EightCubesTraverse) {
    // Data
    std::vector<Eigen::Vector3d> points{
            Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(1.5, 0.5, 0.5),
            Eigen::Vector3d(0.5, 1.5, 0.5), Eigen::Vector3d(1.5, 1.5, 0.5),
            Eigen::Vector3d(0.5, 0.5, 1.5), Eigen::Vector3d(1.5, 0.5, 1.5),
            Eigen::Vector3d(0.5, 1.5, 1.5), Eigen::Vector3d(1.5, 1.5, 1.5),
    };
    std::vector<Eigen::Vector3d> colors{
            Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.1, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.1, 0.0), Eigen::Vector3d(0.1, 0.1, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.1), Eigen::Vector3d(0.1, 0.0, 0.1),
            Eigen::Vector3d(0.0, 0.1, 0.1), Eigen::Vector3d(0.1, 0.1, 0.1),
    };

    // Callback function
    std::vector<Eigen::Vector3d> colors_traversed;
    std::vector<size_t> child_indices_traversed;
    auto f = [&colors_traversed, &child_indices_traversed](
                     const std::shared_ptr<geometry::OctreeNode>& node,
                     const std::shared_ptr<geometry::OctreeNodeInfo>& node_info)
            -> bool {
        if (auto leaf_node =
                    std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                            node)) {
            colors_traversed.push_back(leaf_node->color_);
            child_indices_traversed.push_back(node_info->child_index_);
        }
        return false;
    };

    // Check tree depth 1, we know child index in this case
    geometry::Octree octree_1(1, Eigen::Vector3d(0, 0, 0), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        octree_1.InsertPoint(
                points[i], geometry::OctreeColorLeafNode::GetInitFunction(),
                geometry::OctreeColorLeafNode::GetUpdateFunction(colors[i]));
    }
    colors_traversed.clear();
    child_indices_traversed.clear();
    octree_1.Traverse(f);
    EXPECT_EQ(colors_traversed.size(), 8u);
    for (size_t i = 0; i < 8; ++i) {
        ExpectEQ(colors_traversed[i], colors[i]);
    }
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(child_indices_traversed[i], i);
    }

    // Check tree depth 5
    geometry::Octree octree_5(5, Eigen::Vector3d(0, 0, 0), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        octree_5.InsertPoint(
                points[i], geometry::OctreeColorLeafNode::GetInitFunction(),
                geometry::OctreeColorLeafNode::GetUpdateFunction(colors[i]));
    }
    colors_traversed.clear();
    child_indices_traversed.clear();
    octree_5.Traverse(f);
    EXPECT_EQ(colors_traversed.size(), 8u);
    ExpectEQ(colors_traversed, colors);
}

TEST(Octree, FragmentPLYCheckClone) {
    // Build src_octree
    geometry::PointCloud pcd;
    io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.ply", pcd);
    geometry::Octree src_octree(5);
    src_octree.ConvertFromPointCloud(pcd, 0.01);

    // Build dst_octree clone
    geometry::Octree dst_octree(src_octree);

    // Also checks the equal operator
    EXPECT_TRUE(src_octree == dst_octree);
}

TEST(Octree, EqualOperatorSpecialCase) {
    geometry::Octree src_octree;
    geometry::Octree dst_octree;
    EXPECT_TRUE(src_octree == dst_octree);
}

TEST(Octree, FragmentPLYLocate) {
    // Build src_octree
    geometry::PointCloud pcd;
    io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.ply", pcd);
    size_t max_depth = 5;
    geometry::Octree octree(max_depth);
    octree.ConvertFromPointCloud(pcd, 0.01);

    // Try locating a few points
    for (size_t idx = 0; idx < pcd.points_.size(); idx += 200) {
        const Eigen::Vector3d& point = pcd.points_[idx];
        std::shared_ptr<geometry::OctreeLeafNode> node;
        std::shared_ptr<geometry::OctreeNodeInfo> node_info;
        std::tie(node, node_info) = octree.LocateLeafNode(point);
        EXPECT_TRUE(geometry::Octree::IsPointInBound(point, node_info->origin_,
                                                     node_info->size_));
        EXPECT_EQ(node_info->depth_, max_depth);
        EXPECT_EQ(node_info->size_, octree.size_ / pow(2, max_depth));
    }
}

TEST(Octree, ConvertFromPointCloudBoundSinglePoint) {
    geometry::Octree octree(10);
    geometry::PointCloud pcd;
    pcd.points_.push_back(Eigen::Vector3d(0, 0, 0));
    pcd.colors_.push_back(Eigen::Vector3d(0, 0.1, 0.2));
    octree.ConvertFromPointCloud(pcd, 0.01);
    ExpectEQ(octree.origin_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(octree.size_, 0.01);
}

TEST(Octree, ConvertFromPointCloudBoundTwoPoints) {
    geometry::Octree octree(10);
    geometry::PointCloud pcd;
    pcd.points_.push_back(Eigen::Vector3d(0, 0, 0));
    pcd.points_.push_back(Eigen::Vector3d(0, 2, 4));
    pcd.colors_.push_back(Eigen::Vector3d(0, 0.1, 0.2));
    pcd.colors_.push_back(Eigen::Vector3d(0.3, 0.4, 0.5));
    octree.ConvertFromPointCloud(pcd, 0.01);
    ExpectEQ(octree.origin_, Eigen::Vector3d(-2, -1, 0));  // Auto-centered
    EXPECT_EQ(octree.size_, 4.04);  // 4.04 = 4 * (1 + 0.01)
}

TEST(Octree, Visualization) {
    geometry::PointCloud pcd;
    io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.ply", pcd);
    auto octree = std::make_shared<geometry::Octree>(6);
    octree->ConvertFromPointCloud(pcd, 0.01);
    // Uncomment the line below for visualization test
    // visualization::DrawGeometries({octree});
}

TEST(Octree, ConvertToJsonValue) {
    geometry::PointCloud pcd;
    io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.ply", pcd);
    size_t max_depth = 5;
    geometry::Octree src_octree(max_depth);
    src_octree.ConvertFromPointCloud(pcd, 0.01);

    Json::Value json_value;
    src_octree.ConvertToJsonValue(json_value);

    geometry::Octree dst_octree;
    dst_octree.ConvertFromJsonValue(json_value);

    EXPECT_TRUE(src_octree == dst_octree);
}

}  // namespace tests
}  // namespace open3d
