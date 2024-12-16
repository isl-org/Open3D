// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/BufferConnection.h"
#include "open3d/io/rpc/Connection.h"
#include "open3d/io/rpc/DummyReceiver.h"
#include "open3d/io/rpc/MessageUtils.h"
#include "open3d/io/rpc/RemoteFunctions.h"
#include "open3d/io/rpc/ZMQContext.h"
#include "pybind/core/tensor_type_caster.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace io {

void pybind_rpc_declarations(py::module& m_io) {
    py::module m_rpc = m_io.def_submodule("rpc");
    py::class_<rpc::ConnectionBase, std::shared_ptr<rpc::ConnectionBase>>(
            m_rpc, "_ConnectionBase");
    py::class_<rpc::Connection, std::shared_ptr<rpc::Connection>,
               rpc::ConnectionBase>(m_rpc, "Connection", R"doc(
The default connection class which uses a ZeroMQ socket.
)doc");
    py::class_<rpc::BufferConnection, std::shared_ptr<rpc::BufferConnection>,
               rpc::ConnectionBase>(m_rpc, "BufferConnection", R"doc(
A connection writing to a memory buffer.
)doc");
    py::class_<rpc::DummyReceiver, std::shared_ptr<rpc::DummyReceiver>>(
            m_rpc, "_DummyReceiver",
            "Dummy receiver for the server side receiving requests from a "
            "client.");
}
void pybind_rpc_definitions(py::module& m_io) {
    auto m_rpc = static_cast<py::module>(m_io.attr("rpc"));
    // this is to cleanly shutdown the zeromq context on windows.
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(
            py::cpp_function([]() { rpc::DestroyZMQContext(); }));
    auto connection = static_cast<
            py::class_<rpc::Connection, std::shared_ptr<rpc::Connection>,
                       rpc::ConnectionBase>>(m_rpc.attr("Connection"));
    connection.def(
            py::init([](std::string address, int connect_timeout, int timeout) {
                return std::shared_ptr<rpc::Connection>(
                        new rpc::Connection(address, connect_timeout, timeout));
            }),
            "Creates a connection object",
            "address"_a = "tcp://127.0.0.1:51454", "connect_timeout"_a = 5000,
            "timeout"_a = 10000);

    auto buffer_connection =
            static_cast<py::class_<rpc::BufferConnection,
                                   std::shared_ptr<rpc::BufferConnection>,
                                   rpc::ConnectionBase>>(
                    m_rpc.attr("BufferConnection"));
    buffer_connection.def(py::init<>())
            .def(
                    "get_buffer",
                    [](const rpc::BufferConnection& self) {
                        return py::bytes(self.buffer().str());
                    },
                    "Returns a copy of the buffer.");
    auto dummy_receiver =
            static_cast<py::class_<rpc::DummyReceiver,
                                   std::shared_ptr<rpc::DummyReceiver>>>(
                    m_rpc.attr("_DummyReceiver"));
    dummy_receiver
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

    m_rpc.def("destroy_zmq_context", &rpc::DestroyZMQContext,
              "Destroys the ZMQ context.");

    m_rpc.def("set_point_cloud", &rpc::SetPointCloud, "pcd"_a, "path"_a = "",
              "time"_a = 0, "layer"_a = "",
              "connection"_a = std::shared_ptr<rpc::Connection>(),
              "Sends a point cloud message to a viewer.");
    docstring::FunctionDocInject(
            m_rpc, "set_point_cloud",
            {
                    {"pcd", "Point cloud object."},
                    {"path", "A path descriptor, e.g., 'mygroup/points'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m_rpc.def("set_triangle_mesh",
              py::overload_cast<const geometry::TriangleMesh&,
                                const std::string&, int, const std::string&,
                                std::shared_ptr<rpc::ConnectionBase>>(
                      &rpc::SetTriangleMesh),
              "mesh"_a, "path"_a = "", "time"_a = 0, "layer"_a = "",
              "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
              R"doc(Sends a triangle mesh to a viewer.
Args:
    mesh (o3d.geometry.TriangleMesh): The triangle mesh.
    path (str): The path in the scene graph.
    time (int): The time associated with the data.
    layer (str): A layer name that can be used by receivers that support layers.
    connection (o3d.io.rpc.Connection): A connection object that will be used for sending the data.

Returns:
    Returns True if the data was successfully received.
)doc");

    m_rpc.def("set_triangle_mesh",
              py::overload_cast<const t::geometry::TriangleMesh&,
                                const std::string&, int, const std::string&,
                                std::shared_ptr<rpc::ConnectionBase>>(
                      &rpc::SetTriangleMesh),
              "mesh"_a, "path"_a = "", "time"_a = 0, "layer"_a = "",
              "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
              R"doc(Sends a triangle mesh to a viewer.
Args:
    mesh (o3d.t.geometry.TriangleMesh): The triangle mesh.
    path (str): The path in the scene graph.
    time (int): The time associated with the data.
    layer (str): A layer name that can be used by receivers that support layers.
    connection (o3d.io.rpc.Connection): A connection object that will be used for sending the data.

Returns:
    Returns True if the data was successfully received.
)doc");

    m_rpc.def("set_mesh_data", &rpc::SetMeshData, "path"_a = "", "time"_a = 0,
              "layer"_a = "", "vertices"_a = core::Tensor({0}, core::Float32),
              "vertex_attributes"_a = std::map<std::string, core::Tensor>(),
              "faces"_a = core::Tensor({0}, core::Int32),
              "face_attributes"_a = std::map<std::string, core::Tensor>(),
              "lines"_a = core::Tensor({0}, core::Int32),
              "line_attributes"_a = std::map<std::string, core::Tensor>(),
              "material"_a = "",
              "material_scalar_attributes"_a = std::map<std::string, float>(),
              "material_vector_attributes"_a =
                      std::map<std::string, Eigen::Vector4f>(),
              "texture_maps"_a = std::map<std::string, t::geometry::Image>(),
              "o3d_type"_a = "",
              "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
              "Sends a set_mesh_data message.");
    docstring::FunctionDocInject(
            m_rpc, "set_mesh_data",
            {
                    {"path", "A path descriptor, e.g., 'mygroup/points'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"vertices", "Tensor defining the vertices."},
                    {"vertex_attributes",
                     "dict of Tensors with vertex attributes."},
                    {"faces", "Tensor defining the faces with vertex indices."},
                    {"face_attributes",
                     "dict of Tensors with face attributes."},
                    {"lines", "Tensor defining lines with vertex indices."},
                    {"line_attributes",
                     "dict of Tensors with line attributes."},
                    {"material",
                     "Basic Material for geometry drawing.  Must be non-empty "
                     "if any material attributes or texture maps are "
                     "provided."},
                    {"material_scalar_attributes",
                     "dict of material scalar attributes for geometry drawing "
                     "(e.g. ``point_size``, ``line_width`` or "
                     "``base_reflectance``)."},
                    {"material_vector_attributes",
                     "dict of material Vector4f attributes for geometry "
                     "drawing (e.g. ``base_color`` or ``absorption_color``)"},
                    {"texture_maps", "dict of Images with textures."},
                    {"o3d_type", R"doc(The type of the geometry. This is one of
        ``PointCloud``, ``LineSet``, ``TriangleMesh``.  This argument should be
        specified for partial data that has no primary key data, e.g., a
        triangle mesh without vertices but with other attribute tensors.)doc"},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m_rpc.def("set_legacy_camera", &rpc::SetLegacyCamera, "camera"_a,
              "path"_a = "", "time"_a = 0, "layer"_a = "",
              "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
              "Sends a PinholeCameraParameters object.");
    docstring::FunctionDocInject(
            m_rpc, "set_legacy_camera",
            {
                    {"path", "A path descriptor, e.g., 'mygroup/camera'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m_rpc.def("set_time", &rpc::SetTime, "time"_a,
              "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
              "Sets the time in the external visualizer.");
    docstring::FunctionDocInject(
            m_rpc, "set_time",
            {
                    {"time", "The time value to set."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m_rpc.def("set_active_camera", &rpc::SetActiveCamera, "path"_a,
              "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
              "Sets the object with the specified path as the active camera.");
    docstring::FunctionDocInject(
            m_rpc, "set_active_camera",
            {
                    {"path", "A path descriptor, e.g., 'mygroup/camera'."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m_rpc.def("data_buffer_to_meta_geometry", &rpc::DataBufferToMetaGeometry,
              "data"_a, R"doc(
This function returns the geometry, the path and the time stored in a
SetMeshData message. data must contain the Request header message followed
by the SetMeshData message. The function returns None for the geometry if not
successful.
)doc");
}

}  // namespace io
}  // namespace open3d
