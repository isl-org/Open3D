// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <tuple>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/LineSet.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace zmq {
class message_t;
}

namespace open3d {
namespace io {
namespace rpc {

namespace messages {
struct Array;
struct Status;
struct MeshData;
}  // namespace messages

/// Helper function for unpacking the Status message from a reply.
/// \param msg     The message that contains the Reply and the Status messages.
///
/// \param offset  Byte offset into the message. Defines where to start parsing
/// the message. The offset will be updated and will point to the first byte
/// after the parse messages. If unpacking fails offset will be set to the end
/// of the message.
///
/// \param ok      Output variable which will be set to true if the unpacking
/// was successful.
///
/// \return The extracted Status message object. Check \p ok to see if the
/// returned object is valid.
std::shared_ptr<messages::Status> UnpackStatusFromReply(
        const zmq::message_t& msg, size_t& offset, bool& ok);

/// Convenience function for checking if the message is an OK.
bool ReplyIsOKStatus(const zmq::message_t& msg);

/// Convenience function for checking if the message is an OK.
/// \param offset \see UnpackStatusFromReply
bool ReplyIsOKStatus(const zmq::message_t& msg, size_t& offset);

/// Creates a serialized Request message for testing purposes.
std::string CreateSerializedRequestMessage(const std::string& msg_id);

std::tuple<const void*, size_t> GetZMQMessageDataAndSize(
        const zmq::message_t& msg);

std::tuple<int32_t, std::string> GetStatusCodeAndStr(
        const messages::Status& status);

std::shared_ptr<zmq::message_t> CreateStatusOKMsg();

/// Converts MeshData to a geometry type. MeshData can store TriangleMesh,
/// PointCloud, and LineSet. The function returns a pointer to the base class
/// Geometry. The pointer is null if the conversion is not successful. Note that
/// the msgpack object backing the memory for \p mesh_data must be alive for
/// calling this function.
std::shared_ptr<t::geometry::Geometry> MeshDataToGeometry(
        const messages::MeshData& mesh_data);

/// Creates MeshData from a TriangleMesh. This function returns the MeshData
/// object for serialization.
messages::MeshData GeometryToMeshData(const t::geometry::TriangleMesh& trimesh);

/// Creates MeshData from a TriangleMesh. This function returns the MeshData
/// object for serialization.
messages::MeshData GeometryToMeshData(const t::geometry::PointCloud& pcd);

/// Creates MeshData from a LineSet. This function returns the MeshData
/// object for serialization.
messages::MeshData GeometryToMeshData(const t::geometry::LineSet& ls);

/// This function returns the geometry, the path and the time stored in a
/// SetMeshData message. \p data must contain the Request header message
/// followed by the SetMeshData message. The function returns a null pointer for
/// the geometry if not successful.
std::tuple<std::string, double, std::shared_ptr<t::geometry::Geometry>>
DataBufferToMetaGeometry(std::string& data);

}  // namespace rpc
}  // namespace io
}  // namespace open3d
