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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Geometry/TriangleMesh.h>
#include <IO/ClassIO/TriangleMeshIO.h>
using namespace three;

void pybind_trianglemesh(py::module &m)
{
	py::class_<TriangleMesh, PyGeometry3D<TriangleMesh>,
			std::shared_ptr<TriangleMesh>, Geometry3D> trianglemesh(m,
			"TriangleMesh");
	py::detail::bind_default_constructor<TriangleMesh>(trianglemesh);
	py::detail::bind_copy_functions<TriangleMesh>(trianglemesh);
	trianglemesh
		.def("__repr__", [](const TriangleMesh &mesh) {
			return std::string("TriangleMesh with ") +
					std::to_string(mesh.vertices_.size()) + " points and " +
					std::to_string(mesh.triangles_.size()) + " triangles.";
		})
		.def(py::self + py::self)
		.def(py::self += py::self)
		.def("compute_triangle_normals", &TriangleMesh::ComputeTriangleNormals,
				"Function to compute triangle normals, usually called before rendering",
				"normalized"_a = true)
		.def("compute_vertex_normals", &TriangleMesh::ComputeVertexNormals,
				"Function to compute vertex normals, usually called before rendering",
				"normalized"_a = true)
		.def("purge", &TriangleMesh::Purge,
				"Function to remove duplicated and non-manifold vertices/triangles")
		.def("has_vertices", &TriangleMesh::HasVertices)
		.def("has_triangles", &TriangleMesh::HasTriangles)
		.def("has_vertex_normals", &TriangleMesh::HasVertexNormals)
		.def("has_vertex_colors", &TriangleMesh::HasVertexColors)
		.def("has_triangle_normals", &TriangleMesh::HasTriangleNormals)
		.def("normalize_normals", &TriangleMesh::NormalizeNormals)
		.def("paint_uniform_color", &TriangleMesh::PaintUniformColor)
		.def_readwrite("vertices", &TriangleMesh::vertices_)
		.def_readwrite("vertex_normals", &TriangleMesh::vertex_normals_)
		.def_readwrite("vertex_colors", &TriangleMesh::vertex_colors_)
		.def_readwrite("triangles", &TriangleMesh::triangles_)
		.def_readwrite("triangle_normals", &TriangleMesh::triangle_normals_);
}

void pybind_trianglemesh_methods(py::module &m)
{
	m.def("read_triangle_mesh", [](const std::string &filename) {
		TriangleMesh mesh;
		ReadTriangleMesh(filename, mesh);
		return mesh;
	}, "Function to read TriangleMesh from file", "filename"_a);
	m.def("write_triangle_mesh", [](const std::string &filename,
			const TriangleMesh &mesh, bool write_ascii, bool compressed) {
		return WriteTriangleMesh(filename, mesh, write_ascii, compressed);
	}, "Function to write TriangleMesh to file", "filename"_a, "mesh"_a,
			"write_ascii"_a = false, "compressed"_a = false);
	m.def("create_mesh_sphere", &CreateMeshSphere,
			"Factory function to create a sphere mesh",
			"radius"_a = 1.0, "resolution"_a = 20);
	m.def("create_mesh_cylinder", &CreateMeshCylinder,
			"Factory function to create a cylinder mesh",
			"radius"_a = 1.0, "height"_a = 2.0, "resolution"_a = 20,
			"split"_a = 4);
	m.def("create_mesh_cone", &CreateMeshCone,
			"Factory function to create a cone mesh",
			"radius"_a = 1.0, "height"_a = 2.0, "resolution"_a = 20,
			"split"_a = 1);
	m.def("create_mesh_arrow", &CreateMeshArrow,
			"Factory function to create an arrow mesh",
			"cylinder_radius"_a = 1.0, "cone_radius"_a = 1.5,
			"cylinder_height"_a = 5.0, "cone_height"_a = 4.0,
			"resolution"_a = 20, "cylinder_split"_a = 4, "cone_split"_a = 1);
	m.def("create_mesh_coordinate_frame", &CreateMeshCoordinateFrame,
			"Factory function to create a coordinate frame mesh",
			"size"_a = 1.0, "origin"_a = Eigen::Vector3d(0.0, 0.0, 0.0));
}
