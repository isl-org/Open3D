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

#include "open3d/geometry/Geometry3D.h"
#include "open3d/utility/IJsonConvertible.h"

namespace open3d {
namespace geometry {

class PointCloud;
class VoxelGrid;

/// \class OctreeNodeInfo
///
/// \brief OctreeNode's information.
///
/// OctreeNodeInfo is computed on the fly, not stored with the Node.
class OctreeNodeInfo {
public:
    /// \brief Default Constructor.
    ///
    /// Initializes all values as 0.
    OctreeNodeInfo() : origin_(0, 0, 0), size_(0), depth_(0), child_index_(0) {}

    /// \brief Parameterized Constructor.
    ///
    /// \param origin Origin coordinate of the node
    /// \param size Size of the node.
    /// \param depth  Depth of the node to the root. The root is of depth 0.
    /// \param child_index Node’s child index of itself.
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
    /// Origin coordinate of the node.
    Eigen::Vector3d origin_;
    /// Size of the node.
    double size_;
    /// Depth of the node to the root. The root is of depth 0.
    size_t depth_;
    /// Node’s child index of itself. For non-root nodes, child_index is 0~7;
    /// root node’s child_index is -1.
    size_t child_index_;
};

/// \class OctreeNode
///
/// \brief The base class for octree node.
///
/// Design decision: do not store origin and size of a node
///     - Good: better space efficiency.
///     - Bad: need to recompute origin and size when traversing.
class OctreeNode : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    ///
    OctreeNode() {}
    virtual ~OctreeNode() {}

    /// Factory function to construct an OctreeNode by parsing the json value.
    static std::shared_ptr<OctreeNode> ConstructFromJsonValue(
            const Json::Value& value);
};

/// \class OctreeInternalNode
///
/// \brief OctreeInternalNode class, containing OctreeNode children.
///
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
    /// \brief Default Constructor.
    ///
    OctreeInternalNode() : children_(8) {}
    static std::shared_ptr<OctreeNodeInfo> GetInsertionNodeInfo(
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const Eigen::Vector3d& point);

    /// \brief Get lambda function for initializing OctreeInternalNode.
    ///
    /// When the init function is called, an empty OctreeInternalNode is
    /// created.
    static std::function<std::shared_ptr<OctreeInternalNode>()>
    GetInitFunction();

    /// \brief Get lambda function for updating OctreeInternalNode.
    ///
    /// This update function does nothing.
    static std::function<void(std::shared_ptr<OctreeInternalNode>)>
    GetUpdateFunction();

    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;

public:
    /// Use vector instead of C-array for Pybind11, otherwise, need to define
    /// more helper functions
    /// https://github.com/pybind/pybind11/issues/546#issuecomment-265707318
    /// List of children Nodes.
    std::vector<std::shared_ptr<OctreeNode>> children_;
};

/// \class OctreeInternalPointNode
///
/// \brief OctreeInternalPointNode class is an OctreeInternalNode containing
/// a list of indices which is the union of all its children's list of indices.
class OctreeInternalPointNode : public OctreeInternalNode {
public:
    /// \brief Default Constructor.
    ///
    OctreeInternalPointNode() : OctreeInternalNode() {}

    /// \brief Get lambda function for initializing OctreeInternalPointNode.
    ///
    /// When the init function is called, an empty OctreeInternalPointNode is
    /// created.
    static std::function<std::shared_ptr<OctreeInternalNode>()>
    GetInitFunction();

    /// \brief Get lambda function for updating OctreeInternalPointNode.
    ///
    /// When called, the update function adds the input point index to the
    /// corresponding node's list of indices of children points.
    static std::function<void(std::shared_ptr<OctreeInternalNode>)>
    GetUpdateFunction(size_t idx);

    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;

public:
    /// Indices of points associated with any children of this node
    std::vector<size_t> indices_;
};

/// \class OctreeLeafNode
///
/// \brief OctreeLeafNode base class.
class OctreeLeafNode : public OctreeNode {
public:
    virtual bool operator==(const OctreeLeafNode& other) const = 0;
    /// Clone this OctreeLeafNode.
    virtual std::shared_ptr<OctreeLeafNode> Clone() const = 0;
};

/// \class OctreeColorLeafNode
///
/// \brief OctreeColorLeafNode class is an OctreeLeafNode containing color.
class OctreeColorLeafNode : public OctreeLeafNode {
public:
    bool operator==(const OctreeLeafNode& other) const override;

    /// Clone this OctreeLeafNode.
    std::shared_ptr<OctreeLeafNode> Clone() const override;

    /// \brief Get lambda function for initializing OctreeLeafNode.
    ///
    /// When the init function is called, an empty OctreeColorLeafNode is
    /// created.
    static std::function<std::shared_ptr<OctreeLeafNode>()> GetInitFunction();

    /// \brief Get lambda function for updating OctreeLeafNode.
    ///
    /// When called, the update function updates the corresponding node with the
    /// input color.
    ///
    /// \param color Color of the node.
    static std::function<void(std::shared_ptr<OctreeLeafNode>)>
    GetUpdateFunction(const Eigen::Vector3d& color);

    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;
    /// TODO: flexible data, with lambda function for handling node
    /// Color of the node.
    Eigen::Vector3d color_ = Eigen::Vector3d(0, 0, 0);
};

/// \class OctreePointColorLeafNode
///
/// \brief OctreePointColorLeafNode class is an OctreeColorLeafNode containing
/// a list of indices corresponding to the point cloud points contained in
/// this leaf node.
class OctreePointColorLeafNode : public OctreeColorLeafNode {
public:
    bool operator==(const OctreeLeafNode& other) const override;
    /// Clone this OctreeLeafNode.
    std::shared_ptr<OctreeLeafNode> Clone() const override;

    /// \brief Get lambda function for initializing OctreeLeafNode.
    ///
    /// When the init function is called, an empty OctreePointColorLeafNode is
    /// created.
    static std::function<std::shared_ptr<OctreeLeafNode>()> GetInitFunction();

    /// \brief Get lambda function for updating OctreeLeafNode.
    ///
    /// When called, the update function adds the point cloud point index to
    /// the corresponding node index list.
    ///
    /// \param idx Point cloud point index associated with node.
    /// \param color Color of the node.
    static std::function<void(std::shared_ptr<OctreeLeafNode>)>
    GetUpdateFunction(size_t index, const Eigen::Vector3d& color);

    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;

    /// Associated point indices with this node.
    std::vector<size_t> indices_;
};

/// \class Octree
///
/// \brief Octree datastructure.
class Octree : public Geometry3D, public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    Octree()
        : Geometry3D(Geometry::GeometryType::Octree),
          origin_(0, 0, 0),
          size_(0),
          max_depth_(0) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param max_depth Sets the value of the max depth of the Octree.
    Octree(const size_t& max_depth)
        : Geometry3D(Geometry::GeometryType::Octree),
          origin_(0, 0, 0),
          size_(0),
          max_depth_(max_depth) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param max_depth Sets the value of the max depth of the Octree.
    /// \param origin Sets the global min bound of the Octree.
    /// \param size Sets the outer bounding box edge size for the whole octree.
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
    Octree& Scale(const double scale, const Eigen::Vector3d& center) override;
    Octree& Rotate(const Eigen::Matrix3d& R,
                   const Eigen::Vector3d& center) override;
    bool ConvertToJsonValue(Json::Value& value) const override;
    bool ConvertFromJsonValue(const Json::Value& value) override;

public:
    /// \brief Convert octree from point cloud.
    ///
    /// \param point_cloud Input point cloud.
    /// \param size_expand A small expansion size such that the octree is
    /// slightly bigger than the original point cloud bounds to accomodate all
    /// points.
    void ConvertFromPointCloud(const geometry::PointCloud& point_cloud,
                               double size_expand = 0.01);

    /// Root of the octree.
    std::shared_ptr<OctreeNode> root_node_ = nullptr;

    /// Global min bound (include). A point is within bound iff
    /// origin_ <= point < origin_ + size_.
    Eigen::Vector3d origin_;

    /// Outer bounding box edge size for the whole octree. A point is within
    /// bound iff origin_ <= point < origin_ + size_.
    double size_;

    /// Max depth of octree. The depth is defined as the distance from the
    /// deepest leaf node to root. A tree with only the root node has depth 0.
    size_t max_depth_;

    /// \brief Insert a point to the octree.
    ///
    /// \param point Coordinates of the point.
    /// \param fl_init Initialization fcn used to create new leaf node
    /// (if needed) associated with the point.
    /// \param fl_update Update fcn used to update the leaf node
    /// associated with the point.
    /// \param fi_init Initialize fcn used to create a new internal node
    /// (if needed) which is an ancestor of the point's leaf node. If omitted,
    /// the default OctreeInternalNode function is used.
    /// \param fi_update Update fcn used to update the internal node
    /// which is an ancestor of the point's leaf node. If omitted, the default
    /// OctreeInternalNode function is used.
    void InsertPoint(
            const Eigen::Vector3d& point,
            const std::function<std::shared_ptr<OctreeLeafNode>()>& fl_init,
            const std::function<void(std::shared_ptr<OctreeLeafNode>)>&
                    fl_update,
            const std::function<std::shared_ptr<OctreeInternalNode>()>&
                    fi_init = nullptr,
            const std::function<void(std::shared_ptr<OctreeInternalNode>)>&
                    fi_update = nullptr);

    /// \brief DFS traversal of Octree from the root, with callback function
    /// called for each node.
    ///
    /// \param f  Callback which fires with each traversed internal/leaf node.
    /// Arguments supply information about the node being traversed and other
    /// node-specific data. A Boolean return value is used for early-stopping.
    /// If f returns true, children of this node will not be traversed.
    void Traverse(
            const std::function<bool(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f);

    /// \brief Const version of Traverse. DFS traversal of Octree from the root,
    /// with callback function called for each node.
    ///
    /// \param f  Callback which fires with each traversed internal/leaf node.
    /// Arguments supply information about the node being traversed and other
    /// node-specific data. A Boolean return value is used for early-stopping.
    /// If f returns true, children of this node will not be traversed.
    void Traverse(
            const std::function<bool(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f) const;

    std::pair<std::shared_ptr<OctreeLeafNode>, std::shared_ptr<OctreeNodeInfo>>

    /// \brief Returns leaf OctreeNode and OctreeNodeInfo where the querypoint
    /// should reside.
    ///
    /// \param point Coordinates of the point.
    LocateLeafNode(const Eigen::Vector3d& point) const;

    /// \brief Return true if point within bound, that is,
    /// origin <= point < origin + size.
    ///
    /// \param point Coordinates of the point.
    /// \param origin Origin coordinates.
    /// \param size Size of the Octree.
    static bool IsPointInBound(const Eigen::Vector3d& point,
                               const Eigen::Vector3d& origin,
                               const double& size);

    /// Returns true if the Octree is completely the same, used for testing.
    bool operator==(const Octree& other) const;

    /// Convert to VoxelGrid.
    std::shared_ptr<geometry::VoxelGrid> ToVoxelGrid() const;

    /// Convert from voxel grid.
    void CreateFromVoxelGrid(const geometry::VoxelGrid& voxel_grid);

private:
    static void TraverseRecurse(
            const std::shared_ptr<OctreeNode>& node,
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const std::function<bool(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f);

    void InsertPointRecurse(
            const std::shared_ptr<OctreeNode>& node,
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const Eigen::Vector3d& point,
            const std::function<std::shared_ptr<OctreeLeafNode>()>& f_l_init,
            const std::function<void(std::shared_ptr<OctreeLeafNode>)>&
                    f_l_update,
            const std::function<std::shared_ptr<OctreeInternalNode>()>&
                    f_i_init,
            const std::function<void(std::shared_ptr<OctreeInternalNode>)>&
                    f_i_update);
};

}  // namespace geometry
}  // namespace open3d
