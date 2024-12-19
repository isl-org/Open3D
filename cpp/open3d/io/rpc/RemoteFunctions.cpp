// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/RemoteFunctions.h"

#include <Eigen/Geometry>
#include <zmq.hpp>

#include "open3d/core/Dispatch.h"
#include "open3d/io/rpc/Connection.h"
#include "open3d/io/rpc/MessageUtils.h"
#include "open3d/io/rpc/Messages.h"
#include "open3d/utility/Logging.h"

using namespace open3d::utility;

namespace open3d {
namespace io {
namespace rpc {

bool SetPointCloud(const geometry::PointCloud& pcd,
                   const std::string& path,
                   int time,
                   const std::string& layer,
                   std::shared_ptr<ConnectionBase> connection) {
    // TODO use SetMeshData here after switching to the new PointCloud class.
    if (!pcd.HasPoints()) {
        LogInfo("SetMeshData: point cloud is empty");
        return false;
    }

    messages::SetMeshData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    msg.data.vertices = messages::Array::FromPtr(
            (double*)pcd.points_.data(), {int64_t(pcd.points_.size()), 3});
    if (pcd.HasNormals()) {
        msg.data.vertex_attributes["normals"] =
                messages::Array::FromPtr((double*)pcd.normals_.data(),
                                         {int64_t(pcd.normals_.size()), 3});
    }
    if (pcd.HasColors()) {
        msg.data.vertex_attributes["colors"] = messages::Array::FromPtr(
                (double*)pcd.colors_.data(), {int64_t(pcd.colors_.size()), 3});
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetTriangleMesh(const geometry::TriangleMesh& mesh,
                     const std::string& path,
                     int time,
                     const std::string& layer,
                     std::shared_ptr<ConnectionBase> connection) {
    // TODO use SetMeshData here after switching to the new TriangleMesh class.
    if (!mesh.HasTriangles()) {
        LogInfo("SetMeshData: triangle mesh is empty");
        return false;
    }

    messages::SetMeshData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    msg.data.SetO3DTypeToTriangleMesh();

    msg.data.vertices =
            messages::Array::FromPtr((double*)mesh.vertices_.data(),
                                     {int64_t(mesh.vertices_.size()), 3});
    msg.data.faces = messages::Array::FromPtr(
            (int*)mesh.triangles_.data(), {int64_t(mesh.triangles_.size()), 3});
    if (mesh.HasVertexNormals()) {
        msg.data.vertex_attributes["normals"] = messages::Array::FromPtr(
                (double*)mesh.vertex_normals_.data(),
                {int64_t(mesh.vertex_normals_.size()), 3});
    }
    if (mesh.HasVertexColors()) {
        msg.data.vertex_attributes["colors"] = messages::Array::FromPtr(
                (double*)mesh.vertex_colors_.data(),
                {int64_t(mesh.vertex_colors_.size()), 3});
    }
    if (mesh.HasTriangleNormals()) {
        msg.data.face_attributes["normals"] = messages::Array::FromPtr(
                (double*)mesh.triangle_normals_.data(),
                {int64_t(mesh.triangle_normals_.size()), 3});
    }
    if (mesh.HasTriangleUvs()) {
        msg.data.face_attributes["uvs"] = messages::Array::FromPtr(
                (double*)mesh.triangle_uvs_.data(),
                {int64_t(mesh.triangle_uvs_.size()), 2});
    }
    if (mesh.HasTextures()) {
        int tex_id = 0;
        for (const auto& image : mesh.textures_) {
            if (!image.IsEmpty()) {
                std::vector<int64_t> shape(
                        {image.height_, image.width_, image.num_of_channels_});
                if (image.bytes_per_channel_ == sizeof(uint8_t)) {
                    msg.data.texture_maps[std::to_string(tex_id)] =
                            messages::Array::FromPtr(
                                    (uint8_t*)image.data_.data(), shape);
                } else if (image.bytes_per_channel_ == sizeof(float)) {
                    msg.data.texture_maps[std::to_string(tex_id)] =
                            messages::Array::FromPtr((float*)image.data_.data(),
                                                     shape);
                } else if (image.bytes_per_channel_ == sizeof(double)) {
                    msg.data.texture_maps[std::to_string(tex_id)] =
                            messages::Array::FromPtr(
                                    (double*)image.data_.data(), shape);
                }
            }
            ++tex_id;
        }
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetTriangleMesh(const t::geometry::TriangleMesh& mesh,
                     const std::string& path,
                     int time,
                     const std::string& layer,
                     std::shared_ptr<ConnectionBase> connection) {
    std::map<std::string, core::Tensor> vertex_attributes(
            mesh.GetVertexAttr().begin(), mesh.GetVertexAttr().end());
    std::map<std::string, core::Tensor> face_attributes(
            mesh.GetTriangleAttr().begin(), mesh.GetTriangleAttr().end());

    std::string material_name;
    std::map<std::string, float> material_scalar_attributes;
    std::map<std::string, std::array<float, 4>> material_vector_attributes;
    std::map<std::string, t::geometry::Image> texture_maps;
    if (mesh.HasMaterial()) {
        const auto& material = mesh.GetMaterial();
        material_name = material.GetMaterialName();
        material_scalar_attributes = std::map<std::string, float>(
                material.GetScalarProperties().begin(),
                material.GetScalarProperties().end());
        for (const auto& it : material.GetVectorProperties()) {
            std::array<float, 4> vec = {it.second(0), it.second(1),
                                        it.second(2), it.second(3)};
            material_vector_attributes[it.first] = vec;
        }
        texture_maps = std::map<std::string, t::geometry::Image>(
                material.GetTextureMaps().begin(),
                material.GetTextureMaps().end());
    }

    messages::MeshData o3d_type;
    o3d_type.SetO3DTypeToTriangleMesh();

    return SetMeshData(path, time, layer, mesh.GetVertexPositions(),
                       vertex_attributes, mesh.GetTriangleIndices(),
                       face_attributes, core::Tensor(),
                       std::map<std::string, core::Tensor>(), material_name,
                       material_scalar_attributes, material_vector_attributes,
                       texture_maps, o3d_type.o3d_type, connection);
}

bool SetMeshData(const std::string& path,
                 int time,
                 const std::string& layer,
                 const core::Tensor& vertices,
                 const std::map<std::string, core::Tensor>& vertex_attributes,
                 const core::Tensor& faces,
                 const std::map<std::string, core::Tensor>& face_attributes,
                 const core::Tensor& lines,
                 const std::map<std::string, core::Tensor>& line_attributes,
                 const std::string& material,
                 const std::map<std::string, float>& material_scalar_attributes,
                 const std::map<std::string, std::array<float, 4>>&
                         material_vector_attributes,
                 const std::map<std::string, t::geometry::Image>& texture_maps,
                 const std::string& o3d_type,
                 std::shared_ptr<ConnectionBase> connection) {
    messages::SetMeshData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;
    msg.data.o3d_type = o3d_type;

    if (vertices.NumElements()) {
        if (vertices.NumDims() != 2) {
            LogError("SetMeshData: vertices ndim must be 2 but is {}",
                     vertices.NumDims());
        }
        if (vertices.GetDtype() != core::Float32 &&
            vertices.GetDtype() != core::Float64) {
            LogError(
                    "SetMeshData: vertices must have dtype Float32 or Float64 "
                    "but "
                    "is {}",
                    vertices.GetDtype().ToString());
        }
        msg.data.vertices = messages::Array::FromTensor(vertices);
    }

    for (const auto& item : vertex_attributes) {
        if (item.second.NumDims() >= 1) {
            msg.data.vertex_attributes[item.first] =
                    messages::Array::FromTensor(item.second);
        } else {
            LogError("SetMeshData: Attribute {} has incompatible shape {}",
                     item.first, item.second.GetShape().ToString());
        }
    }

    if (faces.NumElements()) {
        if (faces.GetDtype() != core::Int32 &&
            faces.GetDtype() != core::Int64) {
            LogError(
                    "SetMeshData: faces must have dtype Int32 or Int64 but "
                    "is {}",
                    faces.GetDtype().ToString());
        } else if (faces.NumDims() != 2) {
            LogError("SetMeshData: faces must have rank 2 but is {}",
                     faces.NumDims());
        } else if (faces.GetShape()[1] < 3) {
            LogError("SetMeshData: last dim of faces must be >=3 but is {}",
                     faces.GetShape()[1]);
        } else {
            msg.data.faces = messages::Array::FromTensor(faces);
        }
    }

    for (const auto& item : face_attributes) {
        if (item.second.NumDims() >= 1) {
            msg.data.face_attributes[item.first] =
                    messages::Array::FromTensor(item.second);
        } else {
            LogError(
                    "SetMeshData: Attribute {} has incompatible shape "
                    "{}",
                    item.first, item.second.GetShape().ToString());
        }
    }

    if (lines.NumElements()) {
        if (lines.GetDtype() != core::Int32 &&
            lines.GetDtype() != core::Int64) {
            LogError(
                    "SetMeshData: lines must have dtype Int32 or Int64 but "
                    "is {}",
                    vertices.GetDtype().ToString());
        } else if (lines.NumDims() != 2) {
            LogError("SetMeshData: lines must have rank 2 but is {}",
                     lines.NumDims());
        } else if (lines.GetShape()[1] < 2) {
            LogError("SetMeshData: last dim of lines must be >=2 but is {}",
                     lines.GetShape()[1]);
        } else {
            msg.data.lines = messages::Array::FromTensor(lines);
        }
    }

    for (const auto& item : line_attributes) {
        if (item.second.NumDims() >= 1) {
            msg.data.line_attributes[item.first] =
                    messages::Array::FromTensor(item.second);
        } else {
            LogError(
                    "SetMeshData: Attribute {} has incompatible shape "
                    "{}",
                    item.first, item.second.GetShape().ToString());
        }
    }

    if (!material.empty()) {
        msg.data.material = material;
        msg.data.material_scalar_attributes = material_scalar_attributes;
        for (const auto& item : material_vector_attributes) {
            msg.data.material_vector_attributes[item.first] = item.second;
        }
        for (const auto& texture_map : texture_maps) {
            if (texture_map.second.IsEmpty()) {
                LogError("SetMeshData: Texture map {} is empty",
                         texture_map.first);
            } else {
                msg.data.texture_maps[texture_map.first] =
                        messages::Array::FromTensor(
                                texture_map.second.AsTensor());
            }
        }

    } else if (!material_scalar_attributes.empty() ||
               !material_vector_attributes.empty() || !texture_maps.empty()) {
        LogError("{}",
                 "SetMeshData: Please provide a material for the texture maps");
    }

    {
        std::string errstr;
        if (!msg.data.CheckMessage(errstr)) {
            LogError("SetMeshData: {}", errstr);
        }
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetLegacyCamera(const camera::PinholeCameraParameters& camera,
                     const std::string& path,
                     int time,
                     const std::string& layer,
                     std::shared_ptr<ConnectionBase> connection) {
    messages::SetCameraData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    // convert extrinsics
    Eigen::Matrix3d R = camera.extrinsic_.block<3, 3>(0, 0);
    Eigen::Vector3d t = camera.extrinsic_.block<3, 1>(0, 3);
    Eigen::Quaterniond q(R);
    msg.data.R[0] = q.x();
    msg.data.R[1] = q.y();
    msg.data.R[2] = q.z();
    msg.data.R[3] = q.w();

    msg.data.t[0] = t[0];
    msg.data.t[1] = t[1];
    msg.data.t[2] = t[2];

    // convert intrinsics
    if (camera.intrinsic_.IsValid()) {
        msg.data.width = camera.intrinsic_.width_;
        msg.data.height = camera.intrinsic_.height_;
        msg.data.intrinsic_model = "PINHOLE";
        msg.data.intrinsic_parameters.resize(4);
        msg.data.intrinsic_parameters[0] =
                camera.intrinsic_.intrinsic_matrix_(0, 0);
        msg.data.intrinsic_parameters[1] =
                camera.intrinsic_.intrinsic_matrix_(1, 1);
        msg.data.intrinsic_parameters[2] =
                camera.intrinsic_.intrinsic_matrix_(0, 2);
        msg.data.intrinsic_parameters[3] =
                camera.intrinsic_.intrinsic_matrix_(1, 2);
        if (camera.intrinsic_.GetSkew() != 0.0) {
            LogWarning(
                    "SetLegacyCamera: Nonzero skew parameteer in "
                    "PinholeCameraParameters will be ignored!");
        }
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetTime(int time, std::shared_ptr<ConnectionBase> connection) {
    messages::SetTime msg;
    msg.time = time;

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetActiveCamera(const std::string& path,
                     std::shared_ptr<ConnectionBase> connection) {
    messages::SetActiveCamera msg;
    msg.path = path;

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
