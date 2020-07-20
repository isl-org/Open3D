// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/utility/RemoteFunctions.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {

void pybind_remote_functions(py::module& m) {
#ifdef BUILD_RPC_INTERFACE
    // py::class_<utility::Connection, std::shared_ptr<utility::Connection>>(
    // m, "_Connection");

    m.def("set_point_cloud", &utility::SetPointCloud, "pcd"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "connection"_a = std::shared_ptr<utility::Connection>(),
          "Sends a point cloud message to a viewer.");
    docstring::FunctionDocInject(
            m, "set_point_cloud",
            {
                    {"pcd", "Point cloud object."},
                    {"path", "A path descriptor, e.g., 'mygroup/points'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_mesh_data", &utility::SetMeshData, "vertices"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "vertex_attributes"_a = std::map<std::string, core::Tensor>(),
          "faces"_a = core::Tensor({0}, core::Dtype::Int32),
          "face_attributes"_a = std::map<std::string, core::Tensor>(),
          "lines"_a = core::Tensor({0}, core::Dtype::Int32),
          "line_attributes"_a = std::map<std::string, core::Tensor>(),
          "textures"_a = std::map<std::string, core::Tensor>(),
          "connection"_a = std::shared_ptr<utility::Connection>(),
          "Sends a set_mesh_data message.");

    docstring::FunctionDocInject(
            m, "set_mesh_data",
            {
                    {"vertices", "Tensor defining the vertices."},
                    {"path", "A path descriptor, e.g., 'mygroup/points'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"vertex_attributes",
                     "dict of Tensors with vertex attributes."},
                    {"faces", "Tensor defining the faces with vertex indices."},
                    {"face_attributes",
                     "dict of Tensors with face attributes."},
                    {"lines", "Tensor defining lines with vertex indices."},
                    {"line_attributes",
                     "dict of Tensors with line attributes."},
                    {"textures", "dict of Tensors with textures."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

#endif
}
}  // namespace open3d
