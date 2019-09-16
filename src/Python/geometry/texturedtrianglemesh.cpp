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

#include "Open3D/Geometry/TexturedTriangleMesh.h"
#include "Python/docstring.h"
#include "Python/geometry/geometry.h"
#include "Python/geometry/geometry_trampoline.h"

using namespace open3d;

void pybind_texturedtrianglemesh(py::module &m) {
    py::class_<geometry::TexturedTriangleMesh,
               PyGeometry3D<geometry::TexturedTriangleMesh>,
               std::shared_ptr<geometry::TexturedTriangleMesh>,
               geometry::TriangleMesh>
            texturedtrianglemesh(
                    m, "TexturedTriangleMesh",
                    "TexturedTriangleMesh class. Triangle mesh contains "
                    "vertices "
                    "and triangles represented by the indices to the "
                    "vertices. Optionally, the mesh may also contain "
                    "triangle normals, vertex normals and vertex colors, uv "
                    "coordinates and a "
                    "texture image.");
    py::detail::bind_default_constructor<geometry::TexturedTriangleMesh>(
            texturedtrianglemesh);
    py::detail::bind_copy_functions<geometry::TexturedTriangleMesh>(
            texturedtrianglemesh);
    texturedtrianglemesh
            .def("__repr__",
                 [](const geometry::TexturedTriangleMesh &mesh) {
                     return std::string(
                                    "geometry::TexturedTriangleMesh with ") +
                            std::to_string(mesh.vertices_.size()) +
                            " points and " +
                            std::to_string(mesh.triangles_.size()) +
                            " triangles.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_uvs", &geometry::TexturedTriangleMesh::HasUvs,
                 "Returns ``True`` if the mesh contains uv coordinates.")
            .def("has_texture", &geometry::TexturedTriangleMesh::HasTexture,
                 "Returns ``True`` if the mesh contains a texture image.")
            .def_readwrite("uvs", &geometry::TexturedTriangleMesh::uvs_,
                           "``float64`` array of shape ``(3 * num_triangles, "
                           "2)``, use "
                           "``numpy.asarray()`` to access data: List of "
                           "uvs denoted by the index of points forming "
                           "the triangle.")
            .def_readwrite("texture", &geometry::TexturedTriangleMesh::texture_,
                           "open3d.geometry.Image: The texture image.");
    docstring::ClassMethodDocInject(m, "TexturedTriangleMesh", "has_uvs");
    docstring::ClassMethodDocInject(m, "TexturedTriangleMesh", "has_texture");
}

void pybind_texturedtrianglemesh_methods(py::module &m) {}
