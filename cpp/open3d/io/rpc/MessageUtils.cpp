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

#include "open3d/io/rpc/MessageUtils.h"

#include <zmq.hpp>

#include "open3d/io/rpc/Messages.h"
#include "open3d/utility/Logging.h"

using namespace open3d::utility;

namespace open3d {
namespace io {
namespace rpc {

std::shared_ptr<messages::Status> UnpackStatusFromReply(
        const zmq::message_t& msg, size_t& offset, bool& ok) {
    ok = false;
    if (msg.size() <= offset) {
        return std::shared_ptr<messages::Status>();
    };

    messages::Reply reply;
    messages::Status status;
    try {
        auto obj_handle =
                msgpack::unpack((char*)msg.data(), msg.size(), offset);
        obj_handle.get().convert(reply);
        if (reply.msg_id != status.MsgId()) {
            LogDebug("Expected msg with id {} but got {}", status.MsgId(),
                     reply.msg_id);
        } else {
            auto status_obj_handle =
                    msgpack::unpack((char*)msg.data(), msg.size(), offset);
            status_obj_handle.get().convert(status);
            ok = true;
        }
    } catch (std::exception& e) {
        LogDebug("Failed to parse message: {}", e.what());
        offset = msg.size();
    }
    return std::make_shared<messages::Status>(status);
}

bool ReplyIsOKStatus(const zmq::message_t& msg) {
    size_t offset = 0;
    return ReplyIsOKStatus(msg, offset);
}

bool ReplyIsOKStatus(const zmq::message_t& msg, size_t& offset) {
    bool ok;
    auto status = UnpackStatusFromReply(msg, offset, ok);
    if (ok && status && 0 == status->code) {
        return true;
    }
    return false;
}

std::string CreateSerializedRequestMessage(const std::string& msg_id) {
    messages::Request request{msg_id};
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, request);
    return std::string(sbuf.data(), sbuf.size());
}

std::tuple<const void*, size_t> GetZMQMessageDataAndSize(
        const zmq::message_t& msg) {
    return std::make_tuple(msg.data(), msg.size());
}

std::tuple<int32_t, std::string> GetStatusCodeAndStr(
        const messages::Status& status) {
    return std::make_tuple(status.code, status.str);
}

std::shared_ptr<zmq::message_t> CreateStatusOKMsg() {
    auto OK = messages::Status::OK();
    msgpack::sbuffer sbuf;
    messages::Reply reply{OK.MsgId()};
    msgpack::pack(sbuf, reply);
    msgpack::pack(sbuf, OK);
    return std::shared_ptr<zmq::message_t>(
            new zmq::message_t(sbuf.data(), sbuf.size()));
}

core::Tensor ArrayToTensor(const messages::Array& array) {
    auto TypeStrToDtype = [](const std::string& ts) {
        if ("<f4" == ts) {
            return core::Float32;
        } else if ("<f8" == ts) {
            return core::Float64;
        } else if ("|i1" == ts) {
            return core::Int8;
        } else if ("<i2" == ts) {
            return core::Int16;
        } else if ("<i4" == ts) {
            return core::Int32;
        } else if ("<i8" == ts) {
            return core::Int64;
        } else if ("|u1" == ts) {
            return core::UInt8;
        } else if ("<u2" == ts) {
            return core::UInt16;
        } else if ("<u4" == ts) {
            return core::UInt32;
        } else if ("<u8" == ts) {
            return core::UInt64;
        }
        LogError("Unsupported type {}. Cannot convert to Tensor.", ts);
        return core::Undefined;
    };

    core::Tensor result(array.shape, TypeStrToDtype(array.type));
    memcpy(result.GetDataPtr(), array.data.ptr, array.data.size);

    return result;
}

std::tuple<messages::Array, core::Tensor> TensorToArray(
        const core::Tensor& tensor) {
    // We require the tensor to be contiguous and to use the CPU.
    auto t = tensor.To(core::Device("CPU:0")).Contiguous();
    auto a = DISPATCH_DTYPE_TO_TEMPLATE(t.GetDtype(), [&]() {
        return messages::Array::FromPtr(
                (scalar_t*)t.GetDataPtr(),
                static_cast<std::vector<int64_t>>(t.GetShape()));
    });
    // We do not know if t is a shallow or deep copy. Therefore, we need to
    // return t as well to keep the memory alive.
    return std::tie(a, t);
}

std::shared_ptr<t::geometry::Geometry> MeshDataToGeometry(
        const messages::MeshData& mesh_data) {
    std::string errstr;
    if (mesh_data.CheckMessage(errstr)) {
        if (mesh_data.o3d_type == "TriangleMesh" ||
            mesh_data.faces.CheckNonEmpty()) {
            if (mesh_data.faces.CheckShape({-1, 3}, errstr)) {
                std::shared_ptr<t::geometry::TriangleMesh> mesh(
                        new t::geometry::TriangleMesh());
                mesh->SetVertices(ArrayToTensor(mesh_data.vertices));
                for (auto item : mesh_data.vertex_attributes) {
                    mesh->SetVertexAttr(item.first, ArrayToTensor(item.second));
                }
                mesh->SetTriangles(ArrayToTensor(mesh_data.faces));
                for (auto item : mesh_data.face_attributes) {
                    mesh->SetTriangleAttr(item.first,
                                          ArrayToTensor(item.second));
                }
                return mesh;
            } else {
                LogWarning("MeshDataToGeometry: Invalid shape for faces, {}",
                           errstr);
            }
        } else if (mesh_data.o3d_type == "LineSet" ||
                   mesh_data.lines.CheckNonEmpty()) {
            // TODO
        } else if (mesh_data.o3d_type == "PointCloud" ||
                   mesh_data.vertices.CheckNonEmpty()) {
            std::shared_ptr<t::geometry::PointCloud> pcd(
                    new t::geometry::PointCloud());
            pcd->SetPoints(ArrayToTensor(mesh_data.vertices));
            for (auto item : mesh_data.vertex_attributes) {
                pcd->SetPointAttr(item.first, ArrayToTensor(item.second));
            }
            return pcd;
        } else {
            LogWarning(
                    "MeshDataToGeometry: MeshData has no triangles, lines, or "
                    "vertices");
        }

    } else {
        LogWarning("MeshDataToGeometry: {}", errstr);
    }

    return std::shared_ptr<t::geometry::Geometry>();
}

std::tuple<messages::MeshData, std::vector<core::Tensor>>
TriangleMeshToMeshData(const t::geometry::TriangleMesh& trimesh) {
    std::vector<core::Tensor> keep_alive_tensors;
    messages::MeshData mesh_data;

    // vertices
    auto vertex_attributes = trimesh.GetVertexAttr();
    // TODO switch to "positions" after the primary key has changed.
    if (vertex_attributes.Contains("vertices")) {
        core::Tensor tensor;
        std::tie(mesh_data.vertices, tensor) =
                TensorToArray(trimesh.GetVertices());
        keep_alive_tensors.push_back(tensor);
    } else {
        LogError("TriangleMeshToMeshData: TriangleMesh has no vertices!");
    }

    for (auto item : vertex_attributes) {
        // TODO switch to "positions" after the primary key has changed.
        if (item.first != "vertices") {
            core::Tensor tensor;
            std::tie(mesh_data.vertex_attributes[item.first], tensor) =
                    TensorToArray(item.second);
            keep_alive_tensors.push_back(tensor);
        }
    }

    // triangles
    auto triangle_attributes = trimesh.GetTriangleAttr();
    for (auto item : triangle_attributes) {
        // TODO switch to "indices" after the primary key has changed.
        if (item.first != "triangles") {
            core::Tensor tensor;
            std::tie(mesh_data.face_attributes[item.first], tensor) =
                    TensorToArray(item.second);
            keep_alive_tensors.push_back(tensor);
        } else {
            core::Tensor tensor;
            std::tie(mesh_data.faces, tensor) = TensorToArray(item.second);
            keep_alive_tensors.push_back(tensor);
        }
    }

    mesh_data.o3d_type = "TriangleMesh";

    return std::tie(mesh_data, keep_alive_tensors);
}

std::tuple<messages::MeshData, std::vector<core::Tensor>> PointCloudToMeshData(
        const t::geometry::PointCloud& pcd) {
    std::vector<core::Tensor> keep_alive_tensors;
    messages::MeshData mesh_data;

    // points
    auto point_attributes = pcd.GetPointAttr();
    // TODO switch to "positions" after the primary key has changed.
    if (point_attributes.Contains("points")) {
        core::Tensor tensor;
        std::tie(mesh_data.vertices, tensor) = TensorToArray(pcd.GetPoints());
        keep_alive_tensors.push_back(tensor);
    } else {
        LogError("PointCloudToMeshData: PointCloud has no points!");
    }

    for (auto item : point_attributes) {
        // TODO switch to "positions" after the primary key has changed.
        if (item.first != "points") {
            core::Tensor tensor;
            std::tie(mesh_data.vertex_attributes[item.first], tensor) =
                    TensorToArray(item.second);
            keep_alive_tensors.push_back(tensor);
        }
    }

    mesh_data.o3d_type = "PointCloud";

    return std::tie(mesh_data, keep_alive_tensors);
}

std::tuple<std::string, double, std::shared_ptr<t::geometry::Geometry>>
GetDataFromSetMeshDataBuffer(std::string& data) {
    const char* buffer = data.data();
    size_t buffer_size = data.size();

    auto limits = msgpack::unpack_limit(0xffffffff,  // array
                                        0xffffffff,  // map
                                        65536,       // str
                                        0xffffffff,  // bin
                                        0xffffffff,  // ext
                                        100          // depth
    );

    size_t offset = 0;
    messages::Request req;
    try {
        auto obj_handle = msgpack::unpack(buffer, buffer_size, offset, nullptr,
                                          nullptr, limits);
        auto obj = obj_handle.get();
        req = obj.as<messages::Request>();

        if (messages::SetMeshData::MsgId() == req.msg_id) {
            auto oh = msgpack::unpack(buffer, buffer_size, offset, nullptr,
                                      nullptr, limits);
            auto obj = oh.get();
            messages::SetMeshData msg;
            msg = obj.as<messages::SetMeshData>();
            auto result = MeshDataToGeometry(msg.data);
            double time = msg.time;
            return std::tie(msg.path, time, result);
        } else {
            LogWarning(
                    "GetDataFromSetMeshDataBuffer: Wrong message id. Expected "
                    "'{}' but got '{}'.",
                    messages::SetMeshData::MsgId(), req.msg_id);
        }
    } catch (std::exception& err) {
        LogWarning("GetDataFromSetMeshDataBuffer: {}", err.what());
    }
    return std::forward_as_tuple(std::string(), 0.,
                                 std::shared_ptr<t::geometry::Geometry>());
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
