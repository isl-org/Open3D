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

std::map<std::string, messages::Array> TensorMapToArrayMap(
        const t::geometry::TensorMap& tensor_map) {
    std::map<std::string, messages::Array> result;
    for (auto item : tensor_map) {
        result[item.first] = messages::Array::FromTensor(item.second);
    }
    return result;
}

std::shared_ptr<t::geometry::Geometry> MeshDataToGeometry(
        const messages::MeshData& mesh_data) {
    std::string errstr;
    if (mesh_data.CheckMessage(errstr)) {
        if (mesh_data.O3DTypeIsTriangleMesh() ||
            mesh_data.faces.CheckNonEmpty()) {
            if (mesh_data.faces.CheckShape({-1, 3}, errstr)) {
                auto mesh = std::make_shared<t::geometry::TriangleMesh>();
                mesh->SetVertexPositions(ArrayToTensor(mesh_data.vertices));
                for (auto item : mesh_data.vertex_attributes) {
                    mesh->SetVertexAttr(item.first, ArrayToTensor(item.second));
                }
                mesh->SetTriangleIndices(ArrayToTensor(mesh_data.faces));
                for (auto item : mesh_data.face_attributes) {
                    mesh->SetTriangleAttr(item.first,
                                          ArrayToTensor(item.second));
                }
                return mesh;
            } else {
                LogWarning("MeshDataToGeometry: Invalid shape for faces, {}",
                           errstr);
            }
        } else if (mesh_data.O3DTypeIsLineSet() ||
                   mesh_data.lines.CheckNonEmpty()) {
            if (mesh_data.lines.CheckShape({-1, 2}, errstr)) {
                auto ls = std::make_shared<t::geometry::LineSet>();
                ls->SetPointPositions(ArrayToTensor(mesh_data.vertices));
                for (auto item : mesh_data.vertex_attributes) {
                    ls->SetPointAttr(item.first, ArrayToTensor(item.second));
                }
                ls->SetLineIndices(ArrayToTensor(mesh_data.lines));
                for (auto item : mesh_data.line_attributes) {
                    ls->SetLineAttr(item.first, ArrayToTensor(item.second));
                }
                return ls;
            } else {
                LogWarning("MeshDataToGeometry: Invalid shape for lines, {}",
                           errstr);
            }
        } else if (mesh_data.O3DTypeIsPointCloud() ||
                   mesh_data.vertices.CheckNonEmpty()) {
            auto pcd = std::make_shared<t::geometry::PointCloud>();
            pcd->SetPointPositions(ArrayToTensor(mesh_data.vertices));
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

messages::MeshData GeometryToMeshData(
        const t::geometry::TriangleMesh& trimesh) {
    messages::MeshData mesh_data;

    // vertices
    auto vertex_attributes = trimesh.GetVertexAttr();
    mesh_data.vertex_attributes = TensorMapToArrayMap(vertex_attributes);
    if (mesh_data.vertex_attributes.count("positions")) {
        mesh_data.vertices = mesh_data.vertex_attributes["positions"];
        mesh_data.vertex_attributes.erase("positions");
    } else {
        LogError("GeometryToMeshData: TriangleMesh has no vertices!");
    }

    // triangles
    auto face_attributes = trimesh.GetTriangleAttr();
    mesh_data.face_attributes = TensorMapToArrayMap(face_attributes);
    if (mesh_data.face_attributes.count("indices")) {
        mesh_data.faces = mesh_data.face_attributes["indices"];
        mesh_data.face_attributes.erase("indices");
    }

    mesh_data.SetO3DTypeToTriangleMesh();

    return mesh_data;
}

messages::MeshData GeometryToMeshData(const t::geometry::PointCloud& pcd) {
    messages::MeshData mesh_data;

    // points
    auto point_attributes = pcd.GetPointAttr();
    mesh_data.vertex_attributes = TensorMapToArrayMap(point_attributes);
    if (mesh_data.vertex_attributes.count("positions")) {
        mesh_data.vertices = mesh_data.vertex_attributes["positions"];
        mesh_data.vertex_attributes.erase("positions");
    } else {
        LogError("GeometryToMeshData: PointCloud has no points!");
    }

    mesh_data.SetO3DTypeToPointCloud();

    return mesh_data;
}

messages::MeshData GeometryToMeshData(const t::geometry::LineSet& ls) {
    messages::MeshData mesh_data;

    // points
    auto point_attributes = ls.GetPointAttr();
    mesh_data.vertex_attributes = TensorMapToArrayMap(point_attributes);
    if (mesh_data.vertex_attributes.count("positions")) {
        mesh_data.vertices = mesh_data.vertex_attributes["positions"];
        mesh_data.vertex_attributes.erase("positions");
    } else {
        LogError("GeometryToMeshData: LineSet has no points!");
    }

    // triangles
    auto line_attributes = ls.GetLineAttr();
    mesh_data.line_attributes = TensorMapToArrayMap(line_attributes);
    if (mesh_data.line_attributes.count("indices")) {
        mesh_data.lines = mesh_data.line_attributes["indices"];
        mesh_data.line_attributes.erase("indices");
    }

    mesh_data.SetO3DTypeToLineSet();

    return mesh_data;
}

std::tuple<std::string, double, std::shared_ptr<t::geometry::Geometry>>
DataBufferToMetaGeometry(std::string& data) {
    const char* buffer = data.data();
    size_t buffer_size = data.size();

    auto limits = msgpack::unpack_limit(0xffffffff,  // array
                                        0xffffffff,  // map
                                        65536,       // str
                                        0xffffffff,  // bin
                                        0xffffffff,  // ext
                                        100          // depth
    );

    messages::Request req;
    try {
        size_t offset = 0;
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
