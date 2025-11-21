// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/Geometry.h"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <deque>
#include <vector>

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_geometry_class_declarations(py::module& m) {
    py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
            geometry(m, "Geometry", "The base geometry class.");
}

void pybind_geometry_class_definitions(py::module& m) {
    // open3d.t.geometry.Geometry
    auto geometry = static_cast<py::class_<Geometry, PyGeometry<Geometry>,
                                           std::shared_ptr<Geometry>>>(
            m.attr("Geometry"));
    geometry.def("clear", &Geometry::Clear,
                 "Clear all elements in the geometry.");
    geometry.def("is_empty", &Geometry::IsEmpty,
                 "Returns ``True`` iff the geometry is empty.");
    geometry.def_property_readonly("device", &Geometry::GetDevice,
                                   "Returns the device of the geometry.");
    geometry.def_property_readonly("is_cpu", &Geometry::IsCPU,
                                   "Returns true if the geometry is on CPU.");
    geometry.def_property_readonly("is_cuda", &Geometry::IsCUDA,
                                   "Returns true if the geometry is on CUDA.");
    docstring::ClassMethodDocInject(m, "Geometry", "clear");
    docstring::ClassMethodDocInject(m, "Geometry", "is_empty");
}

void pybind_geometry_declarations(py::module& m) {
    py::module m_geometry = m.def_submodule(
            "geometry", "Tensor-based geometry defining module.");

    py::native_enum<Metric>(
            m_geometry, "Metric", "enum.Enum",
            "Enum for metrics for comparing point clouds and triangle meshes.")
            .value("ChamferDistance", Metric::ChamferDistance,
                   "Chamfer Distance")
            .value("HausdorffDistance", Metric::HausdorffDistance,
                   "Hausdorff Distance")
            .value("FScore", Metric::FScore, "F-Score")
            .export_values()
            .finalize();
    py::class_<MetricParameters> metric_parameters(
            m_geometry, "MetricParameters",
            "Holder for various parameters required by metrics.");

    py::native_enum<MethodOBBCreate>(
            m_geometry, "MethodOBBCreate", "enum.Enum",
            "Methods for creating oriented bounding boxes.")
            .value("PCA", MethodOBBCreate::PCA)
            .value("MINIMAL_APPROX", MethodOBBCreate::MINIMAL_APPROX)
            .value("MINIMAL_JYLANKI", MethodOBBCreate::MINIMAL_JYLANKI)
            .export_values()
            .finalize();

    pybind_geometry_class_declarations(m_geometry);
    pybind_drawable_geometry_class_declarations(m_geometry);
    pybind_tensormap_declarations(m_geometry);
    pybind_pointcloud_declarations(m_geometry);
    pybind_lineset_declarations(m_geometry);
    pybind_trianglemesh_declarations(m_geometry);
    pybind_image_declarations(m_geometry);
    pybind_boundingvolume_declarations(m_geometry);
    pybind_voxel_block_grid_declarations(m_geometry);
    pybind_raycasting_scene_declarations(m_geometry);
}

void pybind_geometry_definitions(py::module& m) {
    auto m_geometry = static_cast<py::module>(m.attr("geometry"));
    py::bind_vector<std::vector<Metric>>(m_geometry, "VectorMetric");

    auto metric_parameters = static_cast<py::class_<MetricParameters>>(
            m_geometry.attr("MetricParameters"));
    // Use std::deque instead of std::vector to enable automatic Python list /
    // tuple conversion. FIXME: Ideally this should work for std::vector.
    metric_parameters
            .def(py::init([](const std::deque<float>& fsr, size_t nsp) {
                     std::vector<float> fsrvec{fsr.begin(), fsr.end()};
                     return MetricParameters{fsrvec, nsp};
                 }),
                 "fscore_radius"_a = std::deque<float>{0.01f},
                 "n_sampled_points"_a = 1000)
            .def_property(
                    "fscore_radius",
                    [](const MetricParameters& self) {  // getter
                        return std::deque<float>(self.fscore_radius.begin(),
                                                 self.fscore_radius.end());
                    },
                    [](MetricParameters& self,
                       const std::deque<float>& fsr) {  // setter
                        self.fscore_radius =
                                std::vector<float>(fsr.begin(), fsr.end());
                    },
                    "Radius for computing the F-Score. A match between "
                    "a point and its nearest neighbor is sucessful if "
                    "it is within this radius.")
            .def_readwrite("n_sampled_points",
                           &MetricParameters::n_sampled_points,
                           "Points are sampled uniformly from the surface of "
                           "triangle meshes before distance computation. This "
                           "specifies the number of points sampled. No "
                           "sampling is done for point clouds.")
            .def("__repr__", &MetricParameters::ToString);

    pybind_geometry_class_definitions(m_geometry);
    pybind_drawable_geometry_class_definitions(m_geometry);
    pybind_tensormap_definitions(m_geometry);
    pybind_pointcloud_definitions(m_geometry);
    pybind_lineset_definitions(m_geometry);
    pybind_trianglemesh_definitions(m_geometry);
    pybind_image_definitions(m_geometry);
    pybind_boundingvolume_definitions(m_geometry);
    pybind_voxel_block_grid_definitions(m_geometry);
    pybind_raycasting_scene_definitions(m_geometry);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
