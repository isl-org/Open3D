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

#include "open3d/visualization/rendering/Material.h"

#include <string>
#include <unordered_map>

#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace visualization {
namespace rendering {

void pybind_material(py::module& m) {
    py::class_<Material, std::shared_ptr<Material>> mat(
            m, "Material",
            "Properties (texture maps, scalar and vector) related to "
            "visualization. Materials are optionally set for 3D geometries "
            "such as TriangleMesh, LineSets, and PointClouds");

    mat.def(py::init<>())
            .def(py::init<Material>())
            .def(py::init<const std::string&>())
            .def("set_default_properties", &Material::SetDefaultProperties,
                 "Fills material with defaults for common PBR material "
                 "properties used by Open3D")
            .def("is_valid", &Material::IsValid,
                 "Returns false if material is an empty material")
            .def_property("shader_name", &Material::GetShaderName,
                          &Material::SetShaderName)
            .def("get_texture_map", &Material::GetTextureMap,
                 "Return the image associated with key")
            .def("set_texture_map", &Material::SetTextureMap,
                 "Associates image with key")
            .def("has_texture_map", &Material::HasTextureMap,
                 "Returns true if the material has the specified map")
            .def("get_scalar_property", &Material::GetScalarProperty,
                 "Return the scalar property associated with key")
            .def("set_scalar_property", &Material::SetScalarProperty,
                 "Set value of a scalar property of the Material")
            .def("has_scalar_property", &Material::HasScalarProperty,
                 "Returns true if the material has the scalar property")
            .def("get_vector_property", &Material::GetVectorProperty,
                 "Return the vector property associated with key")
            .def("set_vector_property", &Material::SetVectorProperty,
                 "Set value of a vector property of the Material")
            .def("has_vector_property", &Material::HasVectorProperty,
                 "Returns true if the material has the vector property");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
