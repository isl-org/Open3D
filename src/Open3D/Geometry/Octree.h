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

#include <cstddef>
#include <memory>
#include <vector>

#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Utility/IJsonConvertible.h"

namespace open3d {
namespace geometry {

class PointCloud;
class VoxelGrid;

/// Design decision: do not store origin and size of a node in OctreeNode
/// OctreeNodeInfo is computed on the fly
class OctreeNodeInfo {
public:
    OctreeNodeInfo() : origin_(0, 0, 0), size_(0), depth_(0), child_index_(0) {}
    OctreeNodeInfo(const Eigen::Vector3d& origin,
                   const double& size,
                   const size_t& depth,
                   const size_t& child_index)
        : origin_(origin),
          size_(size),
          depth_(depth),
          child_index_(child_index) {}
    ~OctreeNodeInfo() {}

public:
    Eigen::Vector3d origin_;
    double size_;
    size_t depth_;
    size_t child_index_;
};

/// OctreeNode class
/// Design decision: do not store origin and size of a node
///     - Good: better space efficiency
///     - Bad: need to recompute origin and size when traversing
class OctreeNode : public utility::IJsonConvertible {
public:
    OctreeNode() {}
    virtual ~OctreeNode() {}

    /// Factory function to construct an OctreeNode by parsing the json value.
    static std::shared_ptr<OctreeNode> ConstructFromJsonValue(
            const Json::Value& value);
};

/// Children node ordering conventions are as follows.
///
/// For illustration, assume,
/// - root_node: origin == (0, 0, 0), size == 2
///
/// Then,
/// - children_[0]: origin == (0, 0, 0), size == 1
/// - children_[1]: origin == (1, 0, 0), size == 1, along X-axis next to child 0
/// - children_[2]: origin == (0, 1, 0), size == 1, along Y-axis next to child 0
/// - children_[3]: origin == (1, 1, 0), size == 1, in X-Y plane
/// - children_[4]: origin == (0, 0, 1), size == 1, along Z-axis next to child 0
/// - children_[5]: origin == (1, 0, 1), size == 1, in X-Z plane
/// - children_[6]: origin == (0, 1, 1), size == 1, in Y-Z plane
/// - children_[7]: origin == (1, 1, 1), size == 1, furthest from child 0
class OctreeInternalNode : public OctreeNode {
public:
    OctreeInternalNode() : children_(8) {}
    static std::shared_ptr<OctreeNodeInfo> GetInsertionNodeInfo(
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const Eigen::Vector3d& point);

    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;

public:
    // Use vector instead of C-array for Pybind11, otherwise, need to define
    // more helper functions
    // https://github.com/pybind/pybind11/issues/546#issuecomment-265707318
    std::vector<std::shared_ptr<OctreeNode>> children_;
};

class OctreeLeafNode : public OctreeNode {
public:
    virtual bool operator==(const OctreeLeafNode& other) const = 0;
    virtual std::shared_ptr<OctreeLeafNode> Clone() const = 0;
};

class OctreeColorLeafNode : public OctreeLeafNode {
public:
    bool operator==(const OctreeLeafNode& other) const override;
    std::shared_ptr<OctreeLeafNode> Clone() const override;
    static std::function<std::shared_ptr<OctreeLeafNode>()> GetInitFunction();
    static std::function<void(std::shared_ptr<OctreeLeafNode>)>
    GetUpdateFunction(const Eigen::Vector3d& color);

    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;
    // TODO: flexible data, with lambda function for handling node
    Eigen::Vector3d color_ = Eigen::Vector3d(0, 0, 0);
};

class Octree : public Geometry3D, public utility::IJsonConvertible {
public:
    Octree()
        : Geometry3D(Geometry::GeometryType::Octree),
          origin_(0, 0, 0),
          size_(0),
          max_depth_(0) {}
    Octree(const size_t& max_depth)
        : Geometry3D(Geometry::GeometryType::Octree),
          origin_(0, 0, 0),
          size_(0),
          max_depth_(max_depth) {}
    Octree(const size_t& max_depth,
           const Eigen::Vector3d& origin,
           const double& size)
        : Geometry3D(Geometry::GeometryType::Octree),
          origin_(origin),
          size_(size),
          max_depth_(max_depth) {}
    Octree(const Octree& src_octree);
    ~Octree() override {}

public:
    Octree& Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const override;
    Octree& Transform(const Eigen::Matrix4d& transformation) override;
    Octree& Translate(const Eigen::Vector3d& translation,
                      bool relative = true) override;
    Octree& Scale(const double scale, bool center = true) override;
    Octree& Rotate(const Eigen::Vector3d& rotation,
                   bool center = true,
                   RotationType type = RotationType::XYZ) override;
    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;

public:
    void ConvertFromPointCloud(const geometry::PointCloud& point_cloud,
                               double size_expand = 0.01);

    /// Root of the octree
    std::shared_ptr<OctreeNode> root_node_ = nullptr;

    /// Global min bound (include). A point is within bound iff
    /// origin_ <= point < origin_ + size_
    Eigen::Vector3d origin_;

    /// Outer bounding box edge size for the whole octree. A point is within
    /// bound iff origin_ <= point < origin_ + size_
    double size_;

    /// Max depth of octree. The depth is defined as the distance from the
    /// deepest leaf node to root. A tree with only the root node has depth 0.
    size_t max_depth_;

    /// Insert point
    void InsertPoint(
            const Eigen::Vector3d& point,
            const std::function<std::shared_ptr<OctreeLeafNode>()>& f_init,
            const std::function<void(std::shared_ptr<OctreeLeafNode>)>&
                    f_update);

    /// DFS traversal of Octree from the root, with callback function called
    /// for each node
    void Traverse(
            const std::function<void(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f);

    /// Const version of Traverse. DFS traversal of Octree from the root, with
    /// callback function called for each node
    void Traverse(
            const std::function<void(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f) const;

    std::pair<std::shared_ptr<OctreeLeafNode>, std::shared_ptr<OctreeNodeInfo>>
    LocateLeafNode(const Eigen::Vector3d& point) const;

    /// Return true if point within bound, that is,
    /// origin <= point < origin + size
    static bool IsPointInBound(const Eigen::Vector3d& point,
                               const Eigen::Vector3d& origin,
                               const double& size);

    /// Returns true if the Octree is completely the same, used for testing
    bool operator==(const Octree& other) const;

    /// Convert to voxel grid
    std::shared_ptr<geometry::VoxelGrid> ToVoxelGrid() const;

    /// Convert from voxel grid
    void CreateFromVoxelGrid(const geometry::VoxelGrid& voxel_grid);

private:
    static void TraverseRecurse(
            const std::shared_ptr<OctreeNode>& node,
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const std::function<void(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f);

    void InsertPointRecurse(
            const std::shared_ptr<OctreeNode>& node,
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const Eigen::Vector3d& point,
            const std::function<std::shared_ptr<OctreeLeafNode>()>& f_init,
            const std::function<void(std::shared_ptr<OctreeLeafNode>)>&
                    f_update);
};

}  // namespace geometry
}  // namespace open3d
