// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4789)
#endif

#include "open3d/visualization/rendering/Material.h"

#include "open3d/visualization/rendering/MaterialRecord.h"
#include "pybind/open3d_pybind.h"

PYBIND11_MAKE_OPAQUE(
        std::unordered_map<std::string, open3d::t::geometry::Image>)
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, float>)
// NOTE: The following line triggers buffer overflow warnings on Windows which
// is why warning 4789 is disabled when building on Windows
PYBIND11_MAKE_OPAQUE(
        open3d::visualization::rendering::Material::VectorPropertyMap)

namespace open3d {
namespace visualization {
namespace rendering {

void pybind_material_declarations(py::module& m) {
    py::bind_map<std::unordered_map<std::string, t::geometry::Image>>(
            m, "TextureMaps");
    py::bind_map<std::unordered_map<std::string, float>>(m, "ScalarProperties");
    py::bind_map<Material::VectorPropertyMap>(m, "VectorProperties");
    py::class_<Material, std::shared_ptr<Material>> mat(
            m, "Material",
            "Properties (texture maps, scalar and vector) related to "
            "visualization. Materials are optionally set for 3D geometries "
            "such as TriangleMesh, LineSets, and PointClouds");
}
void pybind_material_definitions(py::module& m) {
    auto mat = static_cast<py::class_<Material, std::shared_ptr<Material>>>(
            m.attr("Material"));
    mat.def(py::init<>())
            .def(py::init<Material>(), "", "mat"_a)
            .def(py::init<const std::string&>(), "", "material_name"_a)
            .def(py::init(&Material::FromMaterialRecord), "material_record"_a,
                 "Convert from MaterialRecord.")
            .def("__repr__", &Material::ToString)
            .def("set_default_properties", &Material::SetDefaultProperties,
                 "Fills material with defaults for common PBR material "
                 "properties used by Open3D")
            .def("is_valid", &Material::IsValid,
                 "Returns false if material is an empty material")
            .def_property("material_name", &Material::GetMaterialName,
                          &Material::SetMaterialName)
            .def_property_readonly("texture_maps", &Material::GetTextureMaps)
            .def_property_readonly("scalar_properties",
                                   &Material::GetScalarProperties)
            .def_property_readonly("vector_properties",
                                   &Material::GetVectorProperties);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
