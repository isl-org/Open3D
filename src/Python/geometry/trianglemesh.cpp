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

#include "Python/geometry/geometry_trampoline.h"
#include "Python/geometry/geometry.h"
#include "Python/docstring.h"

#include <Open3D/Geometry/TriangleMesh.h>

using namespace open3d;

void pybind_trianglemesh(py::module &m) {
    py::class_<geometry::TriangleMesh, PyGeometry3D<geometry::TriangleMesh>,
               std::shared_ptr<geometry::TriangleMesh>, geometry::Geometry3D>
            trianglemesh(m, "TriangleMesh", "TriangleMesh");
    py::detail::bind_default_constructor<geometry::TriangleMesh>(trianglemesh);
    py::detail::bind_copy_functions<geometry::TriangleMesh>(trianglemesh);
    trianglemesh
            .def("__repr__",
                 [](const geometry::TriangleMesh &mesh) {
                     return std::string("geometry::TriangleMesh with ") +
                            std::to_string(mesh.vertices_.size()) +
                            " points and " +
                            std::to_string(mesh.triangles_.size()) +
                            " triangles.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("compute_triangle_normals",
                 &geometry::TriangleMesh::ComputeTriangleNormals,
                 "Function to compute triangle normals, usually called before "
                 "rendering",
                 "normalized"_a = true)
            .def("compute_vertex_normals",
                 &geometry::TriangleMesh::ComputeVertexNormals,
                 "Function to compute vertex normals, usually called before "
                 "rendering",
                 "normalized"_a = true)
            .def("compute_adjacency_list",
                 &geometry::TriangleMesh::ComputeAdjacencyList,
                 "Function to compute adjacency list, call before adjacency "
                 "list is needed")
            .def("purge", &geometry::TriangleMesh::Purge,
                 "Function to remove duplicated and non-manifold "
                 "vertices/triangles")
            .def("has_vertices", &geometry::TriangleMesh::HasVertices)
            .def("has_triangles", &geometry::TriangleMesh::HasTriangles)
            .def("has_vertex_normals",
                 &geometry::TriangleMesh::HasVertexNormals)
            .def("has_vertex_colors", &geometry::TriangleMesh::HasVertexColors)
            .def("has_triangle_normals",
                 &geometry::TriangleMesh::HasTriangleNormals)
            .def("has_adjacency_list",
                 &geometry::TriangleMesh::HasAdjacencyList)
            .def("normalize_normals", &geometry::TriangleMesh::NormalizeNormals)
            .def("paint_uniform_color",
                 &geometry::TriangleMesh::PaintUniformColor)
            .def_readwrite("vertices", &geometry::TriangleMesh::vertices_)
            .def_readwrite("vertex_normals",
                           &geometry::TriangleMesh::vertex_normals_)
            .def_readwrite("vertex_colors",
                           &geometry::TriangleMesh::vertex_colors_)
            .def_readwrite("triangles", &geometry::TriangleMesh::triangles_)
            .def_readwrite("triangle_normals",
                           &geometry::TriangleMesh::triangle_normals_)
            .def_readwrite("adjacency_list",
                           &geometry::TriangleMesh::adjacency_list_);
}

void pybind_trianglemesh_methods(py::module &m) {
    // Overloaded function, do not inject docs. Keep commented out for future.
    m.def("select_down_sample",
          (std::shared_ptr<geometry::TriangleMesh>(*)(
                  const geometry::TriangleMesh &,
                  const std::vector<size_t> &)) &
                  geometry::SelectDownSample,
          "Function to select mesh from input triangle mesh into output "
          "triangle mesh. ``input``: The input triangle mesh. ``indices``: "
          "Indices of vertices to be selected.",
          "input"_a, "indices"_a);
    // docstring::FunctionDocInject(
    //         m, "select_down_sample",
    //         {{"input", "The input triangle mesh."},
    //          {"indices", "Indices of vertices to be selected."}});

    m.def("crop_triangle_mesh", &geometry::CropTriangleMesh,
          "Function to crop input triangle mesh into output triangle mesh",
          "input"_a, "min_bound"_a, "max_bound"_a);
    docstring::FunctionDocInject(
            m, "crop_triangle_mesh",
            {{"input", "The input triangle mesh."},
             {"min_bound", "Minimum bound for vertex coordinate."},
             {"max_bound", "Maximum bound for vertex coordinate."}});

    m.def("create_mesh_box", &geometry::CreateMeshBox,
          "Factory function to create a box. The left bottom corner on the "
          "front will be placed at (0, 0, 0).",
          "width"_a = 1.0, "height"_a = 1.0, "depth"_a = 1.0);
    docstring::FunctionDocInject(m, "create_mesh_box",
                                 {{"width", "x-directional length."},
                                  {"height", "y-directional length."},
                                  {"depth", "z-directional length."}});

    m.def("create_mesh_sphere", &geometry::CreateMeshSphere,
          "Factory function to create a sphere mesh centered at (0, 0, 0).",
          "radius"_a = 1.0, "resolution"_a = 20);
    docstring::FunctionDocInject(
            m, "create_mesh_sphere",
            {{"radius", "The radius of the sphere."},
             {"resolution",
              "The resolution of the sphere. The longitues will be split into "
              "``resolution`` segments (i.e. there are ``resolution + 1`` "
              "latitude lines including the north and south pole). The "
              "latitudes will be split into ```2 * resolution`` segments (i.e. "
              "there are ``2 * resolution`` longitude lines.)"}});

    m.def("create_mesh_cylinder", &geometry::CreateMeshCylinder,
          "Factory function to create a cylinder mesh", "radius"_a = 1.0,
          "height"_a = 2.0, "resolution"_a = 20, "split"_a = 4);
    docstring::FunctionDocInject(
            m, "create_mesh_cylinder",
            {{"radius", "The radius of the cylinder."},
             {"height",
              "The height of the cylinder. The axis of the cylinder will be "
              "from (0, 0, -height/2) to (0, 0, height/2)."},
             {"resolution",
              " The circle will be split into ``resolution`` segments"},
             {"split",
              "The ``height`` will be split into ``split`` segments."}});

    m.def("create_mesh_cone", &geometry::CreateMeshCone,
          "Factory function to create a cone mesh", "radius"_a = 1.0,
          "height"_a = 2.0, "resolution"_a = 20, "split"_a = 1);
    docstring::FunctionDocInject(
            m, "create_mesh_cone",
            {{"radius", "The radius of the cone."},
             {"height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, 0) to (0, 0, height)."},
             {"resolution",
              "The circle will be split into ``resolution`` segments"},
             {"split",
              "The ``height`` will be split into ``split`` segments."}});

    m.def("create_mesh_arrow", &geometry::CreateMeshArrow,
          "Factory function to create an arrow mesh", "cylinder_radius"_a = 1.0,
          "cone_radius"_a = 1.5, "cylinder_height"_a = 5.0,
          "cone_height"_a = 4.0, "resolution"_a = 20, "cylinder_split"_a = 4,
          "cone_split"_a = 1);
    docstring::FunctionDocInject(
            m, "create_mesh_arrow",
            {{"cylinder_radius", "The radius of the cylinder."},
             {"cone_radius", "The radius of the cone."},
             {"cylinder_height",
              "The height of the cylinder. The cylinder is from (0, 0, 0) to "
              "(0, 0, cylinder_height)"},
             {"cone_height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, cylinder_height) to (0, 0, cylinder_height + cone_height)"},
             {"resolution",
              "The cone will be split into ``resolution`` segments."},
             {"cylinder_split",
              "The ``cylinder_height`` will be split into ``cylinder_split`` "
              "segments."},
             {"cone_split",
              "The ``cone_height`` will be split into ``cone_split`` "
              "segments."}});

    m.def("create_mesh_coordinate_frame", &geometry::CreateMeshCoordinateFrame,
          "Factory function to create a coordinate frame mesh. The coordinate "
          "frame will be centered at ``origin``. The x, y, z axis will be "
          "rendered as red, green, and blue arrows respectively.",
          "size"_a = 1.0, "origin"_a = Eigen::Vector3d(0.0, 0.0, 0.0));
    docstring::FunctionDocInject(
            m, "create_mesh_coordinate_frame",
            {{"size", "The size of the coordinate frame."},
             {"origin", "The origin of the cooridnate frame."}});
}
