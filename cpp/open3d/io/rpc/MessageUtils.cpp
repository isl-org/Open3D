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
#include "open3d/visualization/rendering/Material.h"

using namespace open3d::utility;
using open3d::visualization::rendering::Material;

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

/// Creates a Tensor from an Array. This function also returns a contiguous CPU
/// Tensor. Note that the msgpack object backing the memory for \p array must be
/// alive for calling this function.
static core::Tensor ArrayToTensor(const messages::Array& array) {
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

/// Converts a TensorMap to an Array map.
static std::map<std::string, messages::Array> TensorMapToArrayMap(
        const t::geometry::TensorMap& tensor_map) {
    std::map<std::string, messages::Array> result;
    for (auto item : tensor_map) {
        result[item.first] = messages::Array::FromTensor(item.second);
    }
    return result;
}

static Material GetMaterialFromMeshData(const messages::MeshData& mesh_data,
                                        std::string& errstr) {
    Material material(mesh_data.material);
    if (mesh_data.material.empty()) return material;
    for (const auto& scalar : mesh_data.material_scalar_attributes)
        material.SetScalarProperty(scalar.first, scalar.second);
    for (const auto& vec : mesh_data.material_vector_attributes)
        material.SetVectorProperty(vec.first,
                                   Eigen::Vector4f(vec.second.data()));
    // Allow 2, 3 dim images. Don't restrict n_channels to allow channel packing
    const std::vector<int64_t> expected_shapes[] = {{-1, -1}, {-1, -1, -1}};
    for (const auto& texture : mesh_data.texture_maps) {
        std::string es(texture.first + ": ");
        bool is_right_shape = false;
        for (const auto& shape : expected_shapes) {
            is_right_shape = texture.second.CheckShape(shape, es);
            if (is_right_shape) break;
        }
        if (!is_right_shape) {
            errstr = errstr.empty() ? es : errstr + '\n' + es;
        } else {
            material.SetTextureMap(
                    texture.first,
                    t::geometry::Image(ArrayToTensor(texture.second)));
        }
    }
    return material;
}

std::shared_ptr<t::geometry::Geometry> MeshDataToGeometry(
        const messages::MeshData& mesh_data) {
    std::string errstr;
    if (mesh_data.CheckMessage(errstr)) {
        if (mesh_data.O3DTypeIsTriangleMesh() ||
            mesh_data.faces.CheckNonEmpty()) {
            auto mesh = std::make_shared<t::geometry::TriangleMesh>();
            if (mesh_data.vertices.CheckNonEmpty()) {
                mesh->SetVertexPositions(ArrayToTensor(mesh_data.vertices));
            }
            for (auto item : mesh_data.vertex_attributes) {
                mesh->SetVertexAttr(item.first, ArrayToTensor(item.second));
            }
            if (mesh_data.faces.CheckNonEmpty()) {
                if (mesh_data.faces.CheckShape({-1, 3}, errstr)) {
                    mesh->SetTriangleIndices(ArrayToTensor(mesh_data.faces));
                } else {
                    errstr = "Invalid shape for faces, " + errstr;
                }
            }
            for (auto item : mesh_data.face_attributes) {
                mesh->SetTriangleAttr(item.first, ArrayToTensor(item.second));
            }
            mesh->SetMaterial(GetMaterialFromMeshData(mesh_data, errstr));
            if (!errstr.empty()) {
                LogWarning("MeshDataToGeometry: {}", errstr);
            }
            return mesh;
        } else if (mesh_data.O3DTypeIsLineSet() ||
                   mesh_data.lines.CheckNonEmpty()) {
            auto ls = std::make_shared<t::geometry::LineSet>();
            if (mesh_data.vertices.CheckNonEmpty()) {
                ls->SetPointPositions(ArrayToTensor(mesh_data.vertices));
            }
            for (auto item : mesh_data.vertex_attributes) {
                ls->SetPointAttr(item.first, ArrayToTensor(item.second));
            }
            if (mesh_data.lines.CheckNonEmpty()) {
                if (mesh_data.lines.CheckShape({-1, 2}, errstr)) {
                    ls->SetLineIndices(ArrayToTensor(mesh_data.lines));
                } else {
                    errstr = "Invalid shape for lines, " + errstr;
                }
            }
            for (auto item : mesh_data.line_attributes) {
                ls->SetLineAttr(item.first, ArrayToTensor(item.second));
            }
            ls->SetMaterial(GetMaterialFromMeshData(mesh_data, errstr));
            if (!errstr.empty()) {
                LogWarning("MeshDataToGeometry: {}", errstr);
            }
            return ls;
        } else if (mesh_data.O3DTypeIsPointCloud() ||
                   mesh_data.vertices.CheckNonEmpty()) {
            auto pcd = std::make_shared<t::geometry::PointCloud>();
            if (mesh_data.vertices.CheckNonEmpty()) {
                pcd->SetPointPositions(ArrayToTensor(mesh_data.vertices));
            }
            for (auto item : mesh_data.vertex_attributes) {
                pcd->SetPointAttr(item.first, ArrayToTensor(item.second));
            }
            pcd->SetMaterial(GetMaterialFromMeshData(mesh_data, errstr));
            if (!errstr.empty()) {
                LogWarning("MeshDataToGeometry: {}", errstr);
            }
            return pcd;
        } else {
            errstr += "MeshData has no triangles, lines, or vertices";
        }
    }
    LogWarning("MeshDataToGeometry: {}", errstr);
    return std::shared_ptr<t::geometry::Geometry>();
}

static void AddMaterialToMeshData(messages::MeshData& mesh_data,
                                  const Material& material) {
    if (!material.IsValid()) return;
    mesh_data.material = material.GetMaterialName();
    auto scalars = material.GetScalarProperties();
    mesh_data.material_scalar_attributes =
            std::map<std::string, float>(scalars.begin(), scalars.end());
    for (const auto& item : material.GetVectorProperties()) {
        mesh_data.material_vector_attributes[item.first] = {
                item.second[0], item.second[1], item.second[2], item.second[3]};
    }
    for (const auto& item : material.GetTextureMaps()) {
        mesh_data.texture_maps[item.first] =
                messages::Array::FromTensor(item.second.AsTensor());
    }
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

    // material
    AddMaterialToMeshData(mesh_data, trimesh.GetMaterial());

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

    // material
    AddMaterialToMeshData(mesh_data, pcd.GetMaterial());

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

    // lines
    auto line_attributes = ls.GetLineAttr();
    mesh_data.line_attributes = TensorMapToArrayMap(line_attributes);
    if (mesh_data.line_attributes.count("indices")) {
        mesh_data.lines = mesh_data.line_attributes["indices"];
        mesh_data.line_attributes.erase("indices");
    }

    mesh_data.SetO3DTypeToLineSet();

    // material
    AddMaterialToMeshData(mesh_data, ls.GetMaterial());

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
            auto mesh_obj = oh.get();
            messages::SetMeshData msg;
            msg = mesh_obj.as<messages::SetMeshData>();
            auto result = MeshDataToGeometry(msg.data);
            double time = msg.time;
            return std::tie(msg.path, time, result);
        } else {
            LogWarning(
                    "DataBufferToMetaGeometry: Wrong message id. Expected "
                    "'{}' but got '{}'.",
                    messages::SetMeshData::MsgId(), req.msg_id);
        }
    } catch (std::exception& err) {
        LogWarning("DataBufferToMetaGeometry: {}", err.what());
    }
    return std::forward_as_tuple(std::string(), 0.,
                                 std::shared_ptr<t::geometry::Geometry>());
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
