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

#include "open3d/t/geometry/Geometry.h"

#include "open3d/core/TensorCheck.h"
#include "open3d/t/geometry/Grid.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_geometry_class(py::module& m) {
    // open3d.t.geometry.Geometry
    py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
            geometry(m, "Geometry", "The base geometry class.");

    geometry.def("clear", &Geometry::Clear,
                 "Clear all elements in the geometry.")
            .def("is_empty", &Geometry::IsEmpty,
                 "Returns ``True`` iff the geometry is empty.");
    docstring::ClassMethodDocInject(m, "Geometry", "clear");
    docstring::ClassMethodDocInject(m, "Geometry", "is_empty");

    py::class_<GridWithWeightCount> grid(m, "GridWithWeightCount",
                                         "Grid with weight and count.");
    grid.def(py::init<int>(), "resolution"_a);
    grid.def("insert_batch", &GridWithWeightCount::InsertBatch,
             "grid_indices"_a, "weights"_a);
    grid.def("get_weights", &GridWithWeightCount::GetWeights);
    grid.def("get_counts", &GridWithWeightCount::GetCounts);
}

void pybind_geometry(py::module& m) {
    py::module m_submodule = m.def_submodule(
            "geometry", "Tensor-based geometry defining module.");

    pybind_geometry_class(m_submodule);
    pybind_drawable_geometry_class(m_submodule);
    pybind_tensormap(m_submodule);
    pybind_pointcloud(m_submodule);
    pybind_lineset(m_submodule);
    pybind_trianglemesh(m_submodule);
    pybind_image(m_submodule);
    pybind_tsdf_voxelgrid(m_submodule);
    pybind_voxel_block_grid(m_submodule);
    pybind_raycasting_scene(m_submodule);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
