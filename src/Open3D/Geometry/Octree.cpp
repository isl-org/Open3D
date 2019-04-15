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

#include "Open3D/Geometry/Octree.h"

#include <json/json.h>
#include <Eigen/Dense>
#include <algorithm>
#include <unordered_map>

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

std::shared_ptr<OctreeNodeInfo> OctreeInternalNode::GetInsertionNodeInfo(
        const std::shared_ptr<OctreeNodeInfo>& node_info,
        const Eigen::Vector3d& point) {
    if (!Octree::IsPointInBound(point, node_info->origin_, node_info->size_)) {
        throw std::runtime_error(
                "Internal error: cannot insert to child since point not in "
                "parent node bound.");
    }

    double child_size = node_info->size_ / 2.0;
    size_t x_index = point(0) < node_info->origin_(0) + child_size ? 0 : 1;
    size_t y_index = point(1) < node_info->origin_(1) + child_size ? 0 : 1;
    size_t z_index = point(2) < node_info->origin_(2) + child_size ? 0 : 1;
    size_t child_index = x_index + y_index * 2 + z_index * 4;
    Eigen::Vector3d child_origin =
            node_info->origin_ + Eigen::Vector3d(x_index * child_size,
                                                 y_index * child_size,
                                                 z_index * child_size);
    auto child_node_info = std::make_shared<OctreeNodeInfo>(
            child_origin, child_size, node_info->depth_ + 1, child_index);
    return child_node_info;
}

Octree::Octree(const Octree& src_octree)
    : Geometry3D(Geometry::GeometryType::Octree),
      max_depth_(src_octree.max_depth_),
      origin_(src_octree.origin_),
      size_(src_octree.size_) {
    // First traversal: clone nodes without edges
    std::unordered_map<std::shared_ptr<OctreeNode>, std::shared_ptr<OctreeNode>>
            map_src_to_dst_node;
    auto f_build_map =
            [&map_src_to_dst_node](
                    const std::shared_ptr<OctreeNode>& src_node,
                    const std::shared_ptr<OctreeNodeInfo>& src_node_info)
            -> void {
        if (auto src_internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(src_node)) {
            auto dst_internal_node = std::make_shared<OctreeInternalNode>();
            map_src_to_dst_node[src_internal_node] = dst_internal_node;
        } else if (auto src_leaf_node =
                           std::dynamic_pointer_cast<OctreeLeafNode>(
                                   src_node)) {
            auto dst_leaf_node = std::make_shared<OctreeLeafNode>();
            dst_leaf_node->color_ = src_leaf_node->color_;
            map_src_to_dst_node[src_leaf_node] = dst_leaf_node;
        } else {
            throw std::runtime_error("Internal error: unknown node type");
        }
    };
    src_octree.Traverse(f_build_map);

    // Second traversal: add edges
    auto f_clone_edges =
            [&map_src_to_dst_node](
                    const std::shared_ptr<OctreeNode>& src_node,
                    const std::shared_ptr<OctreeNodeInfo>& src_node_info)
            -> void {
        if (auto src_internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(src_node)) {
            auto dst_internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(
                            map_src_to_dst_node.at(src_internal_node));
            for (size_t child_index = 0; child_index < 8; child_index++) {
                auto src_child_node = src_internal_node->children_[child_index];
                if (src_child_node != nullptr) {
                    auto dst_child_node =
                            map_src_to_dst_node.at(src_child_node);
                    dst_internal_node->children_[child_index] = dst_child_node;
                }
            }
        }
    };
    src_octree.Traverse(f_clone_edges);

    // Save root node to dst_octree (this octree)
    root_node_ = map_src_to_dst_node.at(src_octree.root_node_);
}

bool Octree::operator==(const Octree& that) const {
    // Check basic properties
    bool rc = true;
    rc = rc && origin_.isApprox(that.origin_);
    rc = rc && size_ == that.size_;
    rc = rc && max_depth_ == that.max_depth_;
    if (!rc) {
        return rc;
    }

    // Assign and check node ids
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t> map_node_to_id;
    std::unordered_map<size_t, std::shared_ptr<OctreeNode>> map_id_to_node;
    size_t next_id = 0;
    auto f_assign_node_id =
            [&map_node_to_id, &map_id_to_node, &next_id](
                    const std::shared_ptr<OctreeNode>& node,
                    const std::shared_ptr<OctreeNodeInfo>& node_info) -> void {
        map_node_to_id[node] = next_id;
        map_id_to_node[next_id] = node;
        next_id++;
    };

    map_node_to_id.clear();
    map_id_to_node.clear();
    next_id = 0;
    Traverse(f_assign_node_id);
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t>
            this_map_node_to_id = map_node_to_id;
    std::unordered_map<size_t, std::shared_ptr<OctreeNode>>
            this_map_id_to_node = map_id_to_node;
    size_t num_nodes = next_id;

    map_node_to_id.clear();
    map_id_to_node.clear();
    next_id = 0;
    that.Traverse(f_assign_node_id);
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t>
            that_map_node_to_id = map_node_to_id;
    std::unordered_map<size_t, std::shared_ptr<OctreeNode>>
            that_map_id_to_node = map_id_to_node;

    rc = rc && this_map_node_to_id.size() == num_nodes &&
         that_map_node_to_id.size() == num_nodes &&
         this_map_id_to_node.size() == num_nodes &&
         that_map_id_to_node.size() == num_nodes;
    if (!rc) {
        return rc;
    }

    // Check nodes
    for (size_t id = 0; id < num_nodes; ++id) {
        std::shared_ptr<OctreeNode> this_node = this_map_id_to_node.at(id);
        std::shared_ptr<OctreeNode> that_node = that_map_id_to_node.at(id);
        // Check if internal node
        auto is_same_node_type = false;
        auto this_internal_node =
                std::dynamic_pointer_cast<OctreeInternalNode>(this_node);
        auto that_internal_node =
                std::dynamic_pointer_cast<OctreeInternalNode>(that_node);
        if (this_internal_node != nullptr && that_internal_node != nullptr) {
            is_same_node_type = true;
            for (size_t child_index = 0; child_index < 8; child_index++) {
                const std::shared_ptr<OctreeNode>& this_child =
                        this_internal_node->children_[child_index];
                int this_child_id = -1;
                if (this_child != nullptr) {
                    this_child_id = int(this_map_node_to_id.at(this_child));
                }
                const std::shared_ptr<OctreeNode>& that_child =
                        that_internal_node->children_[child_index];
                int that_child_id = -1;
                if (that_child != nullptr) {
                    that_child_id = int(that_map_node_to_id.at(that_child));
                }
                rc = rc && this_child_id == that_child_id;
            }
        }
        // Check if leaf node
        auto this_leaf_node =
                std::dynamic_pointer_cast<OctreeLeafNode>(this_node);
        auto that_leaf_node =
                std::dynamic_pointer_cast<OctreeLeafNode>(that_node);
        if (this_leaf_node != nullptr && that_leaf_node != nullptr) {
            is_same_node_type = true;
            rc = rc && this_leaf_node->color_.isApprox(this_leaf_node->color_);
        }
        // Handle case where node types are different
        rc = rc && is_same_node_type;
    }

    return rc;
}

void Octree::Clear() {
    // Inherited Clear function
    root_node_ = nullptr;
    origin_.setZero();
    size_ = 0;
}

void Octree::Clear(bool reset_bounds) {
    if (reset_bounds) {
        Clear();
    } else {
        root_node_ = nullptr;
    }
}

bool Octree::IsEmpty() const { throw std::runtime_error("Not implemented"); }
Eigen::Vector3d Octree::GetMinBound() const {
    throw std::runtime_error("Not implemented");
}

Eigen::Vector3d Octree::GetMaxBound() const {
    throw std::runtime_error("Not implemented");
}

void Octree::Transform(const Eigen::Matrix4d& transformation) {
    throw std::runtime_error("Not implemented");
}

void Octree::ConvertFromPointCloud(const geometry::PointCloud& point_cloud,
                                   bool reset_bounds,
                                   double size_expand) {
    if (size_expand > 1 || size_expand < 0) {
        throw std::runtime_error("size_expand shall be between 0 and 1");
    }

    // Set bounds
    Clear(reset_bounds);
    if (reset_bounds) {
        // Reset with automatic centering
        Eigen::Array3d min_bound = point_cloud.GetMinBound();
        Eigen::Array3d max_bound = point_cloud.GetMaxBound();
        Eigen::Array3d center = (min_bound + max_bound) / 2;
        Eigen::Array3d half_sizes = center - min_bound;
        double max_half_size = half_sizes.maxCoeff();
        origin_ = min_bound.min(center - max_half_size);
        if (max_half_size == 0) {
            size_ = size_expand;
        } else {
            size_ = max_half_size * 2 * (1 + size_expand);
        }
    }

    // Insert points
    for (size_t idx = 0; idx < point_cloud.points_.size(); idx++) {
        const Eigen::Vector3d& point = point_cloud.points_[idx];
        const Eigen::Vector3d& color = point_cloud.colors_[idx];
        InsertPoint(point, color);
    }
}

void Octree::InsertPoint(const Eigen::Vector3d& point,
                         const Eigen::Vector3d& color) {
    if (root_node_ == nullptr) {
        if (max_depth_ == 0) {
            root_node_ = std::make_shared<OctreeLeafNode>();
        } else {
            root_node_ = std::make_shared<OctreeInternalNode>();
        }
    }
    auto root_node_info =
            std::make_shared<OctreeNodeInfo>(origin_, size_, 0, 0);
    InsertPointRecurse(root_node_, root_node_info, point, color);
}

void Octree::InsertPointRecurse(
        const std::shared_ptr<OctreeNode>& node,
        const std::shared_ptr<OctreeNodeInfo>& node_info,
        const Eigen::Vector3d& point,
        const Eigen::Vector3d& color) {
    if (!IsPointInBound(point, node_info->origin_, node_info->size_)) {
        return;
    }
    if (node_info->depth_ > max_depth_) {
        return;
    } else if (node_info->depth_ == max_depth_) {
        if (auto leaf_node = std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
            leaf_node->color_ = color;
        } else {
            throw std::runtime_error(
                    "Internal error: leaf node must be OctreeLeafNode");
        }
    } else {
        if (auto internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(node)) {
            // Get child node info
            std::shared_ptr<OctreeNodeInfo> child_node_info =
                    internal_node->GetInsertionNodeInfo(node_info, point);

            // Init child node if not yet initialized
            size_t child_index = child_node_info->child_index_;
            if (internal_node->children_[child_index] == nullptr) {
                if (node_info->depth_ == max_depth_ - 1) {
                    internal_node->children_[child_index] =
                            std::make_shared<OctreeLeafNode>();
                } else {
                    internal_node->children_[child_index] =
                            std::make_shared<OctreeInternalNode>();
                }
            }

            // Insert to the child
            InsertPointRecurse(internal_node->children_[child_index],
                               child_node_info, point, color);
        } else {
            throw std::runtime_error(
                    "Internal error: internal node must be OctreeInternalNode");
        }
    }
}

bool Octree::IsPointInBound(const Eigen::Vector3d& point,
                            const Eigen::Vector3d& origin,
                            const double& size) {
    bool rc = (Eigen::Array3d(origin) <= Eigen::Array3d(point)).all() &&
              (Eigen::Array3d(point) < Eigen::Array3d(origin) + size).all();
    return rc;
}

void Octree::Traverse(
        const std::function<void(const std::shared_ptr<OctreeNode>&,
                                 const std::shared_ptr<OctreeNodeInfo>&)>& f) {
    // root_node_'s child index is 0, though it isn't a child node
    TraverseRecurse(root_node_,
                    std::make_shared<OctreeNodeInfo>(origin_, size_, 0, 0), f);
}

void Octree::Traverse(
        const std::function<void(const std::shared_ptr<OctreeNode>&,
                                 const std::shared_ptr<OctreeNodeInfo>&)>& f)
        const {
    // root_node_'s child index is 0, though it isn't a child node
    TraverseRecurse(root_node_,
                    std::make_shared<OctreeNodeInfo>(origin_, size_, 0, 0), f);
}

void Octree::TraverseRecurse(
        const std::shared_ptr<OctreeNode>& node,
        const std::shared_ptr<OctreeNodeInfo>& node_info,
        const std::function<void(const std::shared_ptr<OctreeNode>&,
                                 const std::shared_ptr<OctreeNodeInfo>&)>& f) {
    if (node == nullptr) {
        return;
    } else if (auto internal_node =
                       std::dynamic_pointer_cast<OctreeInternalNode>(node)) {
        f(internal_node, node_info);
        double child_size = node_info->size_ / 2.0;

        for (size_t child_index = 0; child_index < 8; ++child_index) {
            size_t x_index = child_index % 2;
            size_t y_index = (child_index / 2) % 2;
            size_t z_index = (child_index / 4) % 2;

            auto child_node = internal_node->children_[child_index];
            Eigen::Vector3d child_node_origin =
                    node_info->origin_ +
                    Eigen::Vector3d(x_index, y_index, z_index) * child_size;
            auto child_node_info = std::make_shared<OctreeNodeInfo>(
                    child_node_origin, child_size, node_info->depth_ + 1,
                    child_index);
            TraverseRecurse(child_node, child_node_info, f);
        }
    } else if (auto leaf_node =
                       std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
        f(leaf_node, node_info);
    } else {
        throw std::runtime_error("Internal error: unknown node type");
    }
}

std::pair<std::shared_ptr<OctreeLeafNode>, std::shared_ptr<OctreeNodeInfo>>
Octree::LocateLeafNode(const Eigen::Vector3d& point) const {
    // TODO: add early stoping to callback function when the target has been
    //       found, i.e. add return value to callback function.
    // TODO: consider adding node's depth as parameter to the callback function.
    std::shared_ptr<OctreeLeafNode> target_leaf_node = nullptr;
    std::shared_ptr<OctreeNodeInfo> target_leaf_node_info = nullptr;
    auto f_locate_leaf_node =
            [&target_leaf_node, &target_leaf_node_info, &point](
                    const std::shared_ptr<OctreeNode>& node,
                    const std::shared_ptr<OctreeNodeInfo>& node_info) -> void {
        if (IsPointInBound(point, node_info->origin_, node_info->size_)) {
            if (auto leaf_node =
                        std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
                target_leaf_node = leaf_node;
                target_leaf_node_info = node_info;
            }
        }
    };
    Traverse(f_locate_leaf_node);
    return std::make_pair(target_leaf_node, target_leaf_node_info);
}

bool Octree::ConvertToJsonValue(Json::Value& value) const {
    // Return code for conversion
    bool rc = true;

    // Assign id to each node
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t> map_node_to_id;
    size_t next_id = 0;
    auto f_assign_node_id =
            [&map_node_to_id, &next_id](
                    const std::shared_ptr<OctreeNode>& node,
                    const std::shared_ptr<OctreeNodeInfo>& node_info) -> void {
        map_node_to_id[node] = next_id++;
    };
    Traverse(f_assign_node_id);

    // Write nodes to value
    value["class_name"] = "Octree";
    value["nodes"] = Json::arrayValue;
    value["nodes"].resize(next_id);
    auto f_convert_nodes =
            [&map_node_to_id, &value, &rc](
                    const std::shared_ptr<OctreeNode>& node,
                    const std::shared_ptr<OctreeNodeInfo>& node_info) -> void {
        Json::Value json_node;
        size_t id = map_node_to_id.at(node);
        json_node["id"] = Json::Int64(id);
        if (auto internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(node)) {
            // Internal node has 8 children
            json_node["class_name"] = "OctreeInternalNode";
            json_node["children"] = Json::arrayValue;
            json_node["children"].resize(8);
            for (size_t child_index = 0; child_index < 8; ++child_index) {
                const std::shared_ptr<OctreeNode>& child_node =
                        internal_node->children_[child_index];
                if (child_node == nullptr) {
                    json_node["children"][Json::ArrayIndex(child_index)] =
                            Json::Int64(-1);
                } else {
                    const size_t& child_id = map_node_to_id.at(child_node);
                    json_node["children"][Json::ArrayIndex(child_index)] =
                            Json::Int64(child_id);
                }
            }
        } else if (auto leaf_node =
                           std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
            // Leaf node has color
            json_node["class_name"] = "OctreeLeafNode";
            if (!EigenVector3dToJsonArray(leaf_node->color_,
                                          json_node["color"])) {
                utility::PrintError("Leaf node does not have color");
                rc = false;
            }
        } else {
            utility::PrintError("Internal error: unknown node type");
            rc = false;
        }
        value["nodes"][Json::ArrayIndex(id)] = json_node;
    };
    Traverse(f_convert_nodes);

    // Write other info
    if (root_node_ == nullptr) {
        value["root_node"] = -1;
    } else {
        value["root_node"] = Json::Int64(map_node_to_id.at(root_node_));
    }
    rc = rc && EigenVector3dToJsonArray(origin_, value["origin"]);
    value["size"] = size_;
    value["max_depth"] = Json::Int64(max_depth_);

    return rc;
}

bool Octree::ConvertFromJsonValue(const Json::Value& value) {
    if (value.isObject() == false) {
        utility::PrintWarning(
                "Octree read JSON failed: unsupported json format.\n");
        return false;
    }
    if (value.get("class_name", "") != "Octree") {
        return false;
    }

    bool rc = true;

    // Get octree attributes
    rc = EigenVector3dFromJsonArray(origin_, value["origin"]);
    size_ = value.get("size", 0.0).asDouble();
    max_depth_ = value.get("max_depth", 0).asInt64();

    // Create nodes
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t> map_node_to_id;
    std::unordered_map<size_t, std::shared_ptr<OctreeNode>> map_id_to_node;
    for (size_t i = 0; i < value["nodes"].size(); ++i) {
        Json::Value json_node = value["nodes"].get(Json::ArrayIndex(i), 0);
        size_t id = json_node.get("id", 0).asInt64();
        std::shared_ptr<OctreeNode> node;
        if (json_node.get("class_name", "") == "OctreeInternalNode") {
            node = std::make_shared<OctreeInternalNode>();
        } else if (json_node.get("class_name", "") == "OctreeLeafNode") {
            node = std::make_shared<OctreeLeafNode>();
            auto leaf_node = std::dynamic_pointer_cast<OctreeLeafNode>(node);
            rc = rc && EigenVector3dFromJsonArray(leaf_node->color_,
                                                  json_node["color"]);
        } else {
            rc = false;
        }
        if (map_id_to_node.find(id) != map_id_to_node.end()) {
            utility::PrintError("Duplicated node id in json");
            rc = false;
        } else {
            map_id_to_node[id] = node;
        }
    }

    // Assign root node
    int root_node_id =
            value.get("root_node", -1).asInt64();  // Use int since -1
    if (root_node_id != -1) {
        root_node_ = map_id_to_node.at(root_node_id);
    }

    // Create edges
    for (size_t i = 0; i < value["nodes"].size(); ++i) {
        Json::Value json_node = value["nodes"].get(Json::ArrayIndex(i), 0);
        size_t id = json_node.get("id", 0).asInt64();
        if (json_node.get("class_name", "") == "OctreeInternalNode") {
            auto internal_node = std::dynamic_pointer_cast<OctreeInternalNode>(
                    map_id_to_node.at(id));
            const std::shared_ptr<OctreeNode>& node = map_id_to_node.at(id);
            for (size_t child_index = 0; child_index < 8; child_index++) {
                int child_id = json_node["children"]
                                       .get(Json::ArrayIndex(child_index), -1)
                                       .asInt64();
                if (child_id == -1) {
                    internal_node->children_[child_index] = nullptr;
                } else {
                    internal_node->children_[child_index] =
                            map_id_to_node.at(child_id);
                }
            }
        }
    }

    return rc;
}

}  // namespace geometry
}  // namespace open3d
