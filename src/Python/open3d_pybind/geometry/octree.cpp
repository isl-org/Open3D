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

#include <sstream>
#include <unordered_map>

#include "Open3D/Geometry/Octree.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/VoxelGrid.h"

#include "open3d_pybind/docstring.h"
#include "open3d_pybind/geometry/geometry.h"
#include "open3d_pybind/geometry/geometry_trampoline.h"
#include "open3d_pybind/geometry/octree.h"

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
    // geometry::OctreeNodeInfo
    // Binds std::shared_ptr<...> to avoid non-allocated free in python code
    py::class_<geometry::OctreeNodeInfo,
               std::shared_ptr<geometry::OctreeNodeInfo>>
            octree_node_info(m, "OctreeNodeInfo",
                             "OctreeNode's information. OctreeNodeInfo is "
                             "computed on the fly, "
                             "not stored with the Node.");
    octree_node_info.def(py::init([](const Eigen::Vector3d &origin, double size,
                                     size_t depth, size_t child_index) {
                             return new geometry::OctreeNodeInfo(
                                     origin, size, depth, child_index);
                         }),
                         "origin"_a, "size"_a, "depth"_a, "child_index"_a);
    octree_node_info
            .def("__repr__",
                 [](const geometry::OctreeNodeInfo &node_info) {
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
                    "origin", &geometry::OctreeNodeInfo::origin_,
                    "(3, 1) float numpy array: Origin coordinate of the node.")
            .def_readwrite("size", &geometry::OctreeNodeInfo::size_,
                           "float: Size of the node.")
            .def_readwrite("depth", &geometry::OctreeNodeInfo::depth_,
                           "int: Depth of the node to the root. The root is of "
                           "depth 0.")
            .def_readwrite(
                    "child_index", &geometry::OctreeNodeInfo::child_index_,
                    "int: Node's child index of itself. For non-root nodes, "
                    "child_index is 0~7; root node's child_index is -1.");
    docstring::ClassMethodDocInject(m, "OctreeNodeInfo", "__init__");

    // geometry::OctreeNode
    py::class_<geometry::OctreeNode, PyOctreeNode<geometry::OctreeNode>,
               std::shared_ptr<geometry::OctreeNode>>
            octree_node(m, "OctreeNode", "The base class for octree node.");
    octree_node.def("__repr__", [](const geometry::OctreeNode &octree_node) {
        return "geometry::OctreeNode instance.";
    });
    docstring::ClassMethodDocInject(m, "OctreeNode", "__init__");

    // geometry::OctreeInternalNode
    py::class_<geometry::OctreeInternalNode,
               PyOctreeNode<geometry::OctreeInternalNode>,
               std::shared_ptr<geometry::OctreeInternalNode>,
               geometry::OctreeNode>
            octree_internal_node(m, "OctreeInternalNode",
                                 "OctreeInternalNode class, containing "
                                 "OctreeNode children.");
    octree_internal_node.def(
            "__repr__", [](const geometry::OctreeInternalNode &internal_node) {
                size_t num_children = 0;
                for (const std::shared_ptr<geometry::OctreeNode> &child :
                     internal_node.children_) {
                    if (child != nullptr) {
                        num_children++;
                    }
                }
                std::ostringstream repr;
                repr << "OctreeInternalNode with " << num_children
                     << " non-empty child nodes";
                return repr.str();
            });
    py::detail::bind_default_constructor<geometry::OctreeInternalNode>(
            octree_internal_node);
    py::detail::bind_copy_functions<geometry::OctreeInternalNode>(
            octree_internal_node);
    octree_internal_node.def_readwrite("children",
                                       &geometry::OctreeInternalNode::children_,
                                       "List of children Nodes.");
    docstring::ClassMethodDocInject(m, "OctreeInternalNode", "__init__");

    // geometry::OctreeLeafNode
    py::class_<geometry::OctreeLeafNode,
               PyOctreeLeafNode<geometry::OctreeLeafNode>,
               std::shared_ptr<geometry::OctreeLeafNode>, geometry::OctreeNode>
            octree_leaf_node(m, "OctreeLeafNode", "OctreeLeafNode base class.");
    octree_leaf_node
            .def("__repr__",
                 [](const geometry::OctreeLeafNode &leaf_node) {
                     std::ostringstream repr;
                     repr << "OctreeLeafNode base class";
                     return repr.str();
                 })
            .def("__eq__", &geometry::OctreeLeafNode::operator==, "other"_a,
                 "Check equality of OctreeLeafNode.")
            .def("clone", &geometry::OctreeLeafNode::Clone,
                 "Clone this OctreeLeafNode.");

    docstring::ClassMethodDocInject(m, "OctreeLeafNode", "__init__");

    // geometry::OctreeColorLeafNode
    py::class_<geometry::OctreeColorLeafNode,
               PyOctreeLeafNode<geometry::OctreeColorLeafNode>,
               std::shared_ptr<geometry::OctreeColorLeafNode>,
               geometry::OctreeLeafNode>
            octree_color_leaf_node(m, "OctreeColorLeafNode",
                                   "OctreeColorLeafNode class is an "
                                   "OctreeLeafNode containing color.");
    octree_color_leaf_node
            .def("__repr__",
                 [](const geometry::OctreeColorLeafNode &color_leaf_node) {
                     std::ostringstream repr;
                     repr << "OctreeColorLeafNode with color ["
                          << color_leaf_node.color_(0) << ", "
                          << color_leaf_node.color_(1) << ", "
                          << color_leaf_node.color_(2) << "]";
                     return repr.str();
                 })
            .def_readwrite("color", &geometry::OctreeColorLeafNode::color_,
                           "(3, 1) float numpy array: Color of the node.")
            .def_static("get_init_function",
                        &geometry::OctreeColorLeafNode::GetInitFunction,
                        "Get lambda function for initializing OctreeLeafNode. "
                        "When the init function is called, an empty "
                        "OctreeColorLeafNode is created.")
            .def_static("get_update_function",
                        &geometry::OctreeColorLeafNode::GetUpdateFunction,
                        "color"_a,
                        "Get lambda function for updating OctreeLeafNode. When "
                        "called, the update function update the corresponding "
                        "node with the input color.");

    py::detail::bind_default_constructor<geometry::OctreeColorLeafNode>(
            octree_color_leaf_node);
    py::detail::bind_copy_functions<geometry::OctreeColorLeafNode>(
            octree_color_leaf_node);

    // geometry::Octree
    py::class_<geometry::Octree, PyGeometry3D<geometry::Octree>,
               std::shared_ptr<geometry::Octree>, geometry::Geometry3D>
            octree(m, "Octree", "Octree datastructure.");
    py::detail::bind_default_constructor<geometry::Octree>(octree);
    py::detail::bind_copy_functions<geometry::Octree>(octree);
    octree.def(py::init([](size_t max_depth) {
                   return new geometry::Octree(max_depth);
               }),
               "max_depth"_a)
            .def(py::init([](size_t max_depth, const Eigen::Vector3d &origin,
                             double size) {
                     return new geometry::Octree(max_depth, origin, size);
                 }),
                 "max_depth"_a, "origin"_a, "size"_a)
            .def("__repr__",
                 [](const geometry::Octree &octree) {
                     std::ostringstream repr;
                     repr << "geometry::Octree with ";
                     repr << "origin: [" << octree.origin_(0) << ", "
                          << octree.origin_(1) << ", " << octree.origin_(2)
                          << "]";
                     repr << ", size: " << octree.size_;
                     repr << ", max_depth: " << octree.max_depth_;
                     return repr.str();
                 })
            .def("insert_point", &geometry::Octree::InsertPoint, "point"_a,
                 "f_init"_a, "f_update"_a, "Insert a point to the octree.")
            .def("locate_leaf_node", &geometry::Octree::LocateLeafNode,
                 "point"_a,
                 "Returns leaf OctreeNode and OctreeNodeInfo where the query"
                 "point should reside.")
            .def_static("is_point_in_bound", &geometry::Octree::IsPointInBound,
                        "point"_a, "origin"_a, "size"_a,
                        "Return true if point within bound, that is, origin<= "
                        "point < origin + size")
            .def("convert_from_point_cloud",
                 &geometry::Octree::ConvertFromPointCloud, "point_cloud"_a,
                 "size_expand"_a = 0.01, "Convert octree from point cloud.")
            .def("to_voxel_grid", &geometry::Octree::ToVoxelGrid,
                 "Convert to VoxelGrid.")
            .def("create_from_voxel_grid",
                 &geometry::Octree::CreateFromVoxelGrid,
                 "voxel_grid"_a
                 "Convert from VoxelGrid.")
            .def_readwrite("root_node", &geometry::Octree::root_node_,
                           "OctreeNode: The root octree node.")
            .def_readwrite("origin", &geometry::Octree::origin_,
                           "(3, 1) float numpy array: Global min bound "
                           "(include). A point is within bound iff origin <= "
                           "point < origin + size.")
            .def_readwrite("size", &geometry::Octree::size_,
                           "float: Outer bounding box edge size for the whole "
                           "octree. A point is within bound iff origin <= "
                           "point < origin + size.")
            .def_readwrite("max_depth", &geometry::Octree::max_depth_,
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
