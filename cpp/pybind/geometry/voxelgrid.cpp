// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/geometry/VoxelGrid.h"

#include <sstream>

#include "open3d/camera/PinholeCameraParameters.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/Octree.h"
#include "open3d/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_voxelgrid(py::module &m) {
    py::class_<Voxel, std::shared_ptr<Voxel>> voxel(
            m, "Voxel", "Base Voxel class, containing grid id and color");
    py::detail::bind_default_constructor<Voxel>(voxel);
    py::detail::bind_copy_functions<Voxel>(voxel);
    voxel.def("__repr__",
              [](const Voxel &voxel) {
                  std::ostringstream repr;
                  repr << "Voxel with grid_index: (" << voxel.grid_index_(0)
                       << ", " << voxel.grid_index_(1) << ", "
                       << voxel.grid_index_(2) << "), color: ("
                       << voxel.color_(0) << ", " << voxel.color_(1) << ", "
                       << voxel.color_(2) << ")";
                  return repr.str();
              })
            .def(py::init([](const Eigen::Vector3i &grid_index) {
                     return new Voxel(grid_index);
                 }),
                 "grid_index"_a)
            .def(py::init([](const Eigen::Vector3i &grid_index,
                             const Eigen::Vector3d &color) {
                     return new Voxel(grid_index, color);
                 }),
                 "grid_index"_a, "color"_a)
            .def_readwrite("grid_index", &Voxel::grid_index_,
                           "Int numpy array of shape (3,): Grid coordinate "
                           "index of the voxel.")
            .def_readwrite(
                    "color", &Voxel::color_,
                    "Float64 numpy array of shape (3,): Color of the voxel.");

    py::class_<VoxelGrid, PyGeometry3D<VoxelGrid>, std::shared_ptr<VoxelGrid>,
               Geometry3D>
            voxelgrid(m, "VoxelGrid",
                      "VoxelGrid is a collection of voxels which are aligned "
                      "in grid.");
    py::detail::bind_default_constructor<VoxelGrid>(voxelgrid);
    py::detail::bind_copy_functions<VoxelGrid>(voxelgrid);
    voxelgrid
            .def("__repr__",
                 [](const VoxelGrid &voxelgrid) {
                     return std::string("VoxelGrid with ") +
                            std::to_string(voxelgrid.voxels_.size()) +
                            " voxels.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("get_voxels", &VoxelGrid::GetVoxels,
                 "Returns List of ``Voxel``: Voxels contained in voxel grid. "
                 "Changes to the voxels returned from this method"
                 "are not reflected in the voxel grid.")
            .def("has_colors", &VoxelGrid::HasColors,
                 "Returns ``True`` if the voxel grid contains voxel colors.")
            .def("has_voxels", &VoxelGrid::HasVoxels,
                 "Returns ``True`` if the voxel grid contains voxels.")
            .def("get_voxel", &VoxelGrid::GetVoxel, "point"_a,
                 "Returns voxel index given query point.")
            .def("check_if_included", &VoxelGrid::CheckIfIncluded, "queries"_a,
                 "Element-wise check if a query in the list is included in "
                 "the VoxelGrid. Queries are double precision and "
                 "are mapped to the closest voxel.")
            .def("carve_depth_map", &VoxelGrid::CarveDepthMap, "depth_map"_a,
                 "camera_params"_a, "keep_voxels_outside_image"_a = false,
                 "Remove all voxels from the VoxelGrid where none of the "
                 "boundary points of the voxel projects to depth value that is "
                 "smaller, or equal than the projected depth of the boundary "
                 "point. If keep_voxels_outside_image is true then voxels are "
                 "only carved if all boundary points project to a valid image "
                 "location.")
            .def("carve_silhouette", &VoxelGrid::CarveSilhouette,
                 "silhouette_mask"_a, "camera_params"_a,
                 "keep_voxels_outside_image"_a = false,
                 "Remove all voxels from the VoxelGrid where none of the "
                 "boundary points of the voxel projects to a valid mask pixel "
                 "(pixel value > 0). If keep_voxels_outside_image is true then "
                 "voxels are only carved if all boundary points project to a "
                 "valid image location.")
            .def("to_octree", &VoxelGrid::ToOctree, "max_depth"_a,
                 "Convert to Octree.")
            .def("create_from_octree", &VoxelGrid::CreateFromOctree, "octree"_a,
                 "Convert from Octree.")
            .def("get_voxel_center_coordinate",
                 &VoxelGrid::GetVoxelCenterCoordinate, "idx"_a,
                 "Returns the center coordinate of a voxel given its grid "
                 "index.")
            .def("get_voxel_bounding_points",
                 &VoxelGrid::GetVoxelBoundingPoints, "index"_a,
                 "Returns the 8 bounding points of a voxel given its grid "
                 "index.")
            .def_static("create_dense", &VoxelGrid::CreateDense,
                        "Creates a voxel grid where every voxel is set (hence "
                        "dense). This is a useful starting point for voxel "
                        "carving",
                        "origin"_a, "color"_a, "voxel_size"_a, "width"_a,
                        "height"_a, "depth"_a)
            .def_static("create_from_point_cloud",
                        &VoxelGrid::CreateFromPointCloud,
                        "Creates a VoxelGrid from a given PointCloud. The "
                        "color value of a given  voxel is the average color "
                        "value of the points that fall into it (if the "
                        "PointCloud has colors). The bounds of the created "
                        "VoxelGrid are computed from the PointCloud.",
                        "input"_a, "voxel_size"_a)
            .def_static("create_from_point_cloud_within_bounds",
                        &VoxelGrid::CreateFromPointCloudWithinBounds,
                        "Creates a VoxelGrid from a given PointCloud. The "
                        "color value of a given voxel is the average color "
                        "value of the points that fall into it (if the "
                        "PointCloud has colors). The bounds of the created "
                        "VoxelGrid are defined by the given parameters.",
                        "input"_a, "voxel_size"_a, "min_bound"_a, "max_bound"_a)
            .def_static("create_from_triangle_mesh",
                        &VoxelGrid::CreateFromTriangleMesh,
                        "Creates a VoxelGrid from a given TriangleMesh. No "
                        "color information is converted. The bounds of the "
                        "created VoxelGrid are computed from the  "
                        "TriangleMesh.",
                        "input"_a, "voxel_size"_a)
            .def_static(
                    "create_from_triangle_mesh_within_bounds",
                    &VoxelGrid::CreateFromTriangleMeshWithinBounds,
                    "Creates a VoxelGrid from a given TriangleMesh. No color "
                    "information is converted. The bounds "
                    "of the created VoxelGrid are defined by the given "
                    "parameters",
                    "input"_a, "voxel_size"_a, "min_bound"_a, "max_bound"_a)
            .def_readwrite("origin", &VoxelGrid::origin_,
                           "``float64`` vector of length 3: Coorindate of the "
                           "origin point.")
            .def_readwrite("voxel_size", &VoxelGrid::voxel_size_,
                           "``float64`` Size of the voxel.");
    docstring::ClassMethodDocInject(m, "VoxelGrid", "has_colors");
    docstring::ClassMethodDocInject(m, "VoxelGrid", "has_voxels");
    docstring::ClassMethodDocInject(m, "VoxelGrid", "get_voxel",
                                    {{"point", "The query point."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "check_if_included",
            {{"query", "a list of voxel indices to check."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "carve_depth_map",
            {{"depth_map", "Depth map (Image) used for VoxelGrid carving."},
             {"camera_parameters",
              "PinholeCameraParameters used to record the given depth_map."},
             {"keep_voxels_outside_image",
              "retain voxels that don't project"
              " to pixels in the image"}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "carve_silhouette",
            {{"silhouette_mask",
              "Silhouette mask (Image) used for VoxelGrid carving."},
             {"camera_parameters",
              "PinholeCameraParameters used to record the given depth_map."},
             {"keep_voxels_outside_image",
              "retain voxels that don't project"
              " to pixels in the image"}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "to_octree",
            {{"max_depth", "int: Maximum depth of the octree."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_octree",
            {{"octree", "geometry.Octree: The source octree."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "get_voxel_center_coordinate",
            {{"idx", "The grid index of the query voxel."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "get_voxel_bounding_points",
            {{"index", "The grid index of the query voxel."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_dense",
            {{"origin", "Coordinate center of the VoxelGrid"},
             {"color", "Voxel color for all voxels if the VoxelGrid."},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."},
             {"width", "Spatial width extend of the VoxelGrid."},
             {"height", "Spatial height extend of the VoxelGrid."},
             {"depth", "Spatial depth extend of the VoxelGrid."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_point_cloud",
            {{"input", "The input PointCloud"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_point_cloud_within_bounds",
            {{"input", "The input PointCloud"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."},
             {"min_bound",
              "Minimum boundary point for the VoxelGrid to create."},
             {"max_bound",
              "Maximum boundary point for the VoxelGrid to create."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_triangle_mesh",
            {{"input", "The input TriangleMesh"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_triangle_mesh_within_bounds",
            {{"input", "The input TriangleMesh"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."},
             {"min_bound",
              "Minimum boundary point for the VoxelGrid to create."},
             {"max_bound",
              "Maximum boundary point for the VoxelGrid to create."}});
}

void pybind_voxelgrid_methods(py::module &m) {}

}  // namespace geometry
}  // namespace open3d
