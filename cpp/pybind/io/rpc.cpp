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

#include "open3d/io/rpc/BufferConnection.h"
#include "open3d/io/rpc/Connection.h"
#include "open3d/io/rpc/DummyReceiver.h"
#include "open3d/io/rpc/RemoteFunctions.h"
#include "open3d/io/rpc/ZMQContext.h"
#include "pybind/core/tensor_type_caster.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace io {

void pybind_rpc(py::module& m_io) {
    py::module m = m_io.def_submodule("rpc");

    // this is to cleanly shutdown the zeromq context on windows.
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(
            py::cpp_function([]() { rpc::DestroyZMQContext(); }));

    py::class_<rpc::ConnectionBase, std::shared_ptr<rpc::ConnectionBase>>(
            m, "_ConnectionBase");

    py::class_<rpc::Connection, std::shared_ptr<rpc::Connection>,
               rpc::ConnectionBase>(m, "Connection")
            .def(py::init([](std::string address, int connect_timeout,
                             int timeout) {
                     return std::shared_ptr<rpc::Connection>(
                             new rpc::Connection(address, connect_timeout,
                                                 timeout));
                 }),
                 "Creates a connection object",
                 "address"_a = "tcp://127.0.0.1:51454",
                 "connect_timeout"_a = 5000, "timeout"_a = 10000);

    py::class_<rpc::BufferConnection, std::shared_ptr<rpc::BufferConnection>,
               rpc::ConnectionBase>(m, "BufferConnection")
            .def(py::init<>())
            .def(
                    "get_buffer",
                    [](const rpc::BufferConnection& self) {
                        return py::bytes(self.buffer().str());
                    },
                    "Returns a copy of the buffer.");

    py::class_<rpc::DummyReceiver, std::shared_ptr<rpc::DummyReceiver>>(
            m, "_DummyReceiver",
            "Dummy receiver for the server side receiving requests from a "
            "client.")
            .def(py::init([](std::string address, int timeout) {
                     return std::shared_ptr<rpc::DummyReceiver>(
                             new rpc::DummyReceiver(address, timeout));
                 }),
                 "Creates the receiver object which can be used for testing "
                 "connections.",
                 "address"_a = "tcp://127.0.0.1:51454", "timeout"_a = 10000)
            .def("start", &rpc::DummyReceiver::Start,
                 "Starts the receiver mainloop in a new thread.")
            .def("stop", &rpc::DummyReceiver::Stop,
                 "Stops the receiver mainloop and joins the thread. This "
                 "function blocks until the mainloop is done with processing "
                 "messages that have already been received.");

    m.def("destroy_zmq_context", &rpc::DestroyZMQContext,
          "Destroys the ZMQ context.");

    m.def("set_point_cloud", &rpc::SetPointCloud, "pcd"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "connection"_a = std::shared_ptr<rpc::Connection>(),
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

    m.def("set_triangle_mesh", &rpc::SetTriangleMesh, "mesh"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sends a point cloud message to a viewer.");
    docstring::FunctionDocInject(
            m, "set_triangle_mesh",
            {
                    {"mesh", "The TriangleMesh object."},
                    {"path", "A path descriptor, e.g., 'mygroup/mesh'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_mesh_data", &rpc::SetMeshData, "vertices"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "vertex_attributes"_a = std::map<std::string, core::Tensor>(),
          "faces"_a = core::Tensor({0}, core::Dtype::Int32),
          "face_attributes"_a = std::map<std::string, core::Tensor>(),
          "lines"_a = core::Tensor({0}, core::Dtype::Int32),
          "line_attributes"_a = std::map<std::string, core::Tensor>(),
          "textures"_a = std::map<std::string, core::Tensor>(),
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
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

    m.def("set_legacy_camera", &rpc::SetLegacyCamera, "camera"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sends a PinholeCameraParameters object.");
    docstring::FunctionDocInject(
            m, "set_legacy_camera",
            {
                    {"path", "A path descriptor, e.g., 'mygroup/camera'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_time", &rpc::SetTime, "time"_a,
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sets the time in the external visualizer.");
    docstring::FunctionDocInject(
            m, "set_time",
            {
                    {"time", "The time value to set."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_active_camera", &rpc::SetActiveCamera, "path"_a,
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sets the object with the specified path as the active camera.");
    docstring::FunctionDocInject(
            m, "set_active_camera",
            {
                    {"path", "A path descriptor, e.g., 'mygroup/camera'."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });
}

}  // namespace io
}  // namespace open3d
