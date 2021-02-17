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

#include <sstream>
#include <unordered_map>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/VoxelGrid.h"
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"
#include "pybind/geometry/octree.h"

namespace open3d {
namespace geometry {

static const std::unordered_map<std::string, std::string>
        map_octree_argument_docstrings = {
                {"origin", "Origin coordinates."},
                {"size", "Size of the Octree."},
                {"color", "Color of the point."},
                {"point", "Coordinates of the point."},
                {"max_depth", "Maximum depth of the octree."},
                {"point_cloud", "Input point cloud."},
                {"size_expand",
                 "A small expansion size such that the octree is slightly "
                 "bigger than the original point cloud bounds to accomodate "
                 "all points."}};

void pybind_octree(py::module &m) {
    // OctreeNodeInfo
    // Binds std::shared_ptr<...> to avoid non-allocated free in python code
    py::class_<OctreeNodeInfo, std::shared_ptr<OctreeNodeInfo>>
            octree_node_info(m, "OctreeNodeInfo",
                             "OctreeNode's information. OctreeNodeInfo is "
                             "computed on the fly, "
                             "not stored with the Node.");
    octree_node_info.def(py::init([](const Eigen::Vector3d &origin, double size,
                                     size_t depth, size_t child_index) {
                             return new OctreeNodeInfo(origin, size, depth,
                                                       child_index);
                         }),
                         "origin"_a, "size"_a, "depth"_a, "child_index"_a);
    octree_node_info
            .def("__repr__",
                 [](const OctreeNodeInfo &node_info) {
                     std::ostringstream repr;
                     repr << "OctreeNodeInfo with origin ["
                          << node_info.origin_(0) << ", "
                          << node_info.origin_(1) << ", "
                          << node_info.origin_(2) << "]";
                     repr << ", size " << node_info.size_;
                     repr << ", depth " << node_info.depth_;
                     repr << ", child_index " << node_info.child_index_;
                     return repr.str();
                 })
            .def_readwrite(
                    "origin", &OctreeNodeInfo::origin_,
                    "(3, 1) float numpy array: Origin coordinate of the node.")
            .def_readwrite("size", &OctreeNodeInfo::size_,
                           "float: Size of the node.")
            .def_readwrite("depth", &OctreeNodeInfo::depth_,
                           "int: Depth of the node to the root. The root is of "
                           "depth 0.")
            .def_readwrite(
                    "child_index", &OctreeNodeInfo::child_index_,
                    "int: Node's child index of itself. For non-root nodes, "
                    "child_index is 0~7; root node's child_index is -1.");
    docstring::ClassMethodDocInject(m, "OctreeNodeInfo", "__init__");

    // OctreeNode
    py::class_<OctreeNode, PyOctreeNode<OctreeNode>,
               std::shared_ptr<OctreeNode>>
            octree_node(m, "OctreeNode", "The base class for octree node.");
    octree_node.def("__repr__", [](const OctreeNode &octree_node) {
        return "OctreeNode instance.";
    });
    docstring::ClassMethodDocInject(m, "OctreeNode", "__init__");

    // OctreeInternalNode
    py::class_<OctreeInternalNode, PyOctreeNode<OctreeInternalNode>,
               std::shared_ptr<OctreeInternalNode>, OctreeNode>
            octree_internal_node(m, "OctreeInternalNode",
                                 "OctreeInternalNode class, containing "
                                 "OctreeNode children.");
    octree_internal_node
            .def("__repr__",
                 [](const OctreeInternalNode &internal_node) {
                     size_t num_children = 0;
                     for (const std::shared_ptr<OctreeNode> &child :
                          internal_node.children_) {
                         if (child != nullptr) {
                             num_children++;
                         }
                     }
                     std::ostringstream repr;
                     repr << "OctreeInternalNode with " << num_children
                          << " non-empty child nodes";
                     return repr.str();
                 })
            .def_static(
                    "get_init_function", &OctreeInternalNode::GetInitFunction,
                    "Get lambda function for initializing OctreeInternalNode. "
                    "When the init function is called, an empty "
                    "OctreeInternalNode is created.")
            .def_static("get_update_function",
                        &OctreeInternalNode::GetUpdateFunction,
                        "Get lambda function for updating OctreeInternalNode. "
                        "This update function does nothing.");
    py::detail::bind_default_constructor<OctreeInternalNode>(
            octree_internal_node);
    py::detail::bind_copy_functions<OctreeInternalNode>(octree_internal_node);
    octree_internal_node.def_readwrite("children",
                                       &OctreeInternalNode::children_,
                                       "List of children Nodes.");
    docstring::ClassMethodDocInject(m, "OctreeInternalNode", "__init__");

    // OctreeInternalPointNode
    py::class_<OctreeInternalPointNode, PyOctreeNode<OctreeInternalPointNode>,
               std::shared_ptr<OctreeInternalPointNode>, OctreeInternalNode>
            octree_internal_point_node(
                    m, "OctreeInternalPointNode",
                    "OctreeInternalPointNode class is an "
                    "OctreeInternalNode with a list of point "
                    "indices (from point cloud) belonging to "
                    "children of this node.");
    octree_internal_point_node
            .def("__repr__",
                 [](const OctreeInternalPointNode &internal_point_node) {
                     size_t num_children = 0;
                     for (const std::shared_ptr<OctreeNode> &child :
                          internal_point_node.children_) {
                         if (child != nullptr) {
                             num_children++;
                         }
                     }
                     std::ostringstream repr;
                     repr << "OctreeInternalPointNode with " << num_children
                          << " non-empty child nodes and "
                          << internal_point_node.indices_.size() << " points";
                     return repr.str();
                 })
            .def_readwrite("indices", &OctreeInternalPointNode::indices_,
                           "List of point cloud point indices "
                           "contained in children nodes.")
            .def_static("get_init_function",
                        &OctreeInternalPointNode::GetInitFunction,
                        "Get lambda function for initializing "
                        "OctreeInternalPointNode. "
                        "When the init function is called, an empty "
                        "OctreeInternalPointNode is created.")
            .def_static(
                    "get_update_function",
                    &OctreeInternalPointNode::GetUpdateFunction,
                    "Get lambda function for updating OctreeInternalPointNode. "
                    "When called, the update function adds the input "
                    "point index to the corresponding node's list of "
                    "indices of children points.");
    py::detail::bind_default_constructor<OctreeInternalPointNode>(
            octree_internal_point_node);
    py::detail::bind_copy_functions<OctreeInternalPointNode>(
            octree_internal_point_node);
    octree_internal_point_node.def_readwrite(
            "children", &OctreeInternalPointNode::children_,
            "List of children Nodes.");
    docstring::ClassMethodDocInject(m, "OctreeInternalPointNode", "__init__");

    // OctreeLeafNode
    py::class_<OctreeLeafNode, PyOctreeLeafNode<OctreeLeafNode>,
               std::shared_ptr<OctreeLeafNode>, OctreeNode>
            octree_leaf_node(m, "OctreeLeafNode", "OctreeLeafNode base class.");
    octree_leaf_node
            .def("__repr__",
                 [](const OctreeLeafNode &leaf_node) {
                     std::ostringstream repr;
                     repr << "OctreeLeafNode base class";
                     return repr.str();
                 })
            .def("__eq__", &OctreeLeafNode::operator==, "other"_a,
                 "Check equality of OctreeLeafNode.")
            .def("clone", &OctreeLeafNode::Clone, "Clone this OctreeLeafNode.");

    docstring::ClassMethodDocInject(m, "OctreeLeafNode", "__init__");

    // OctreeColorLeafNode
    py::class_<OctreeColorLeafNode, PyOctreeLeafNode<OctreeColorLeafNode>,
               std::shared_ptr<OctreeColorLeafNode>, OctreeLeafNode>
            octree_color_leaf_node(m, "OctreeColorLeafNode",
                                   "OctreeColorLeafNode class is an "
                                   "OctreeLeafNode containing color.");
    octree_color_leaf_node
            .def("__repr__",
                 [](const OctreeColorLeafNode &color_leaf_node) {
                     std::ostringstream repr;
                     repr << "OctreeColorLeafNode with color ["
                          << color_leaf_node.color_(0) << ", "
                          << color_leaf_node.color_(1) << ", "
                          << color_leaf_node.color_(2) << "]";
                     return repr.str();
                 })
            .def_readwrite("color", &OctreeColorLeafNode::color_,
                           "(3, 1) float numpy array: Color of the node.")
            .def_static("get_init_function",
                        &OctreeColorLeafNode::GetInitFunction,
                        "Get lambda function for initializing OctreeLeafNode. "
                        "When the init function is called, an empty "
                        "OctreeColorLeafNode is created.")
            .def_static("get_update_function",
                        &OctreeColorLeafNode::GetUpdateFunction, "color"_a,
                        "Get lambda function for updating OctreeLeafNode. When "
                        "called, the update function updates the corresponding "
                        "node with the input color.");

    py::detail::bind_default_constructor<OctreeColorLeafNode>(
            octree_color_leaf_node);
    py::detail::bind_copy_functions<OctreeColorLeafNode>(
            octree_color_leaf_node);

    // OctreePointColorLeafNode
    py::class_<OctreePointColorLeafNode,
               PyOctreeLeafNode<OctreePointColorLeafNode>,
               std::shared_ptr<OctreePointColorLeafNode>, OctreeLeafNode>
            octree_point_color_leaf_node(m, "OctreePointColorLeafNode",
                                         "OctreePointColorLeafNode class is an "
                                         "OctreeLeafNode containing color.");
    octree_point_color_leaf_node
            .def("__repr__",
                 [](const OctreePointColorLeafNode &color_leaf_node) {
                     std::ostringstream repr;
                     repr << "OctreePointColorLeafNode with color ["
                          << color_leaf_node.color_(0) << ", "
                          << color_leaf_node.color_(1) << ", "
                          << color_leaf_node.color_(2) << "] "
                          << "containing " << color_leaf_node.indices_.size()
                          << " points.";
                     return repr.str();
                 })
            .def_readwrite("color", &OctreePointColorLeafNode::color_,
                           "(3, 1) float numpy array: Color of the node.")
            .def_readwrite("indices", &OctreePointColorLeafNode::indices_,
                           "List of point cloud point indices "
                           "contained in this leaf node.")
            .def_static("get_init_function",
                        &OctreePointColorLeafNode::GetInitFunction,
                        "Get lambda function for initializing OctreeLeafNode. "
                        "When the init function is called, an empty "
                        "OctreePointColorLeafNode is created.")
            .def_static("get_update_function",
                        &OctreePointColorLeafNode::GetUpdateFunction, "idx"_a,
                        "color"_a,
                        "Get lambda function for updating OctreeLeafNode. When "
                        "called, the update function updates the corresponding "
                        "node with the new point index and the input color.");

    py::detail::bind_default_constructor<OctreePointColorLeafNode>(
            octree_point_color_leaf_node);
    py::detail::bind_copy_functions<OctreePointColorLeafNode>(
            octree_point_color_leaf_node);

    // Octree
    py::class_<Octree, PyGeometry3D<Octree>, std::shared_ptr<Octree>,
               Geometry3D>
            octree(m, "Octree", "Octree datastructure.");
    py::detail::bind_default_constructor<Octree>(octree);
    py::detail::bind_copy_functions<Octree>(octree);
    octree.def(py::init([](size_t max_depth) { return new Octree(max_depth); }),
               "max_depth"_a)
            .def(py::init([](size_t max_depth, const Eigen::Vector3d &origin,
                             double size) {
                     return new Octree(max_depth, origin, size);
                 }),
                 "max_depth"_a, "origin"_a, "size"_a)
            .def("__repr__",
                 [](const Octree &octree) {
                     std::ostringstream repr;
                     repr << "Octree with ";
                     repr << "origin: [" << octree.origin_(0) << ", "
                          << octree.origin_(1) << ", " << octree.origin_(2)
                          << "]";
                     repr << ", size: " << octree.size_;
                     repr << ", max_depth: " << octree.max_depth_;
                     return repr.str();
                 })
            .def("insert_point", &Octree::InsertPoint, "point"_a, "f_init"_a,
                 "f_update"_a, "fi_init"_a = nullptr, "fi_update"_a = nullptr,
                 "Insert a point to the octree.")
            .def("traverse",
                 py::overload_cast<const std::function<bool(
                         const std::shared_ptr<OctreeNode> &,
                         const std::shared_ptr<OctreeNodeInfo> &)> &>(
                         &Octree::Traverse, py::const_),
                 "f"_a,
                 "DFS traversal of the octree from the root, with a "
                 "callback function f being called for each node.")
            .def("locate_leaf_node", &Octree::LocateLeafNode, "point"_a,
                 "Returns leaf OctreeNode and OctreeNodeInfo where the query"
                 "point should reside.")
            .def_static("is_point_in_bound", &Octree::IsPointInBound, "point"_a,
                        "origin"_a, "size"_a,
                        "Return true if point within bound, that is, origin<= "
                        "point < origin + size")
            .def("convert_from_point_cloud", &Octree::ConvertFromPointCloud,
                 "point_cloud"_a, "size_expand"_a = 0.01,
                 "Convert octree from point cloud.")
            .def("to_voxel_grid", &Octree::ToVoxelGrid, "Convert to VoxelGrid.")
            .def("create_from_voxel_grid", &Octree::CreateFromVoxelGrid,
                 "voxel_grid"_a
                 "Convert from VoxelGrid.")
            .def_readwrite("root_node", &Octree::root_node_,
                           "OctreeNode: The root octree node.")
            .def_readwrite("origin", &Octree::origin_,
                           "(3, 1) float numpy array: Global min bound "
                           "(include). A point is within bound iff origin <= "
                           "point < origin + size.")
            .def_readwrite("size", &Octree::size_,
                           "float: Outer bounding box edge size for the whole "
                           "octree. A point is within bound iff origin <= "
                           "point < origin + size.")
            .def_readwrite("max_depth", &Octree::max_depth_,
                           "int: Maximum depth of the octree. The depth is "
                           "defined as the distance from the deepest leaf node "
                           "to root. A tree with only the root node has depth "
                           "0.");

    docstring::ClassMethodDocInject(m, "Octree", "__init__");
    docstring::ClassMethodDocInject(m, "Octree", "insert_point",
                                    map_octree_argument_docstrings);
    docstring::ClassMethodDocInject(m, "Octree", "locate_leaf_node",
                                    map_octree_argument_docstrings);
    docstring::ClassMethodDocInject(m, "Octree", "is_point_in_bound",
                                    map_octree_argument_docstrings);
    docstring::ClassMethodDocInject(m, "Octree", "convert_from_point_cloud",
                                    map_octree_argument_docstrings);
    docstring::ClassMethodDocInject(m, "Octree", "to_voxel_grid",
                                    map_octree_argument_docstrings);
    docstring::ClassMethodDocInject(
            m, "Octree", "create_from_voxel_grid",
            {{"voxel_grid", "geometry.VoxelGrid: The source voxel grid."}});
}

void pybind_octree_methods(py::module &m) {}

}  // namespace geometry
}  // namespace open3d
