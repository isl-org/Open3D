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

#pragma once

#include "open3d/camera/PinholeCameraParameters.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Connection.h"
#include "open3d/utility/Messages.h"

#include <map>

namespace open3d {
namespace utility {

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
/// \return The extracted Status message object. Check \param ok to see if the
/// returned object is valid.
messages::Status UnpackStatusFromReply(const zmq::message_t& msg,
                                       size_t& offset,
                                       bool& ok);

/// Convenience function for checking if the message is an OK.
bool ReplyIsOKStatus(const zmq::message_t& msg);

/// Convenience function for checking if the message is an OK.
/// \param offset \see UnpackStatusFromReply
bool ReplyIsOKStatus(const zmq::message_t& msg, size_t& offset);

/// Function for sending a PointCloud.
/// \param pcd         The PointCloud object.
///
/// \param path        Path descriptor defining a location in the scene tree.
/// E.g., 'mygroup/mypoints'.
///
/// \param time        The time point associated with the object.
///
/// \param layer       The layer for this object.
///
/// \param connection  The connection object used for sending the data.
///
bool SetPointCloud(
        const open3d::geometry::PointCloud& pcd,
        const std::string& path = std::string(),
        int time = 0,
        const std::string& layer = std::string(),
        std::shared_ptr<Connection> connection = std::shared_ptr<Connection>());

/// Function for sending general mesh data.
/// \param vertices    Tensor with vertices of shape [N,3]
///
/// \param path               Path descriptor defining a location in the scene
/// tree. E.g., 'mygroup/mypoints'.
///
/// \param time               The time point associated with the object.
///
/// \param layer              The layer for this object.
///
/// \param vertex_attributes  Map with Tensors storing vertex attributes. The
/// first dim of each attribute must match the number of vertices.
///
/// \param faces              Tensor with vertex indices defining the faces. The
/// Tensor is of rank 1 or 2. A rank 2 Tensor with shape [num_faces,n] defines
/// num_faces n-gons. If the rank is 1 then polys of different lengths are
/// stored sequentially. Each polygon is stored as a sequence 'n i1 i2 ... in'
/// with n>=3. The type of the array must be int32_t or int64_t
///
/// \param face_attributes    Map with Tensors storing face attributes. The
/// first dim of each attribute must match the number of faces.
///
/// \param lines              Tensor with vertex indices defining the lines. The
/// Tensor is of rank 1 or 2. A rank 2 Tensor with shape [num_lines,n] defines
/// num_lines linestrips. If the rank is 1 then linestrips of different lengths
/// are stored sequentially. Each linestrips is stored as a sequence 'n i1 i2
/// ... in' with n>=2. The type of the array must be int32_t or int64_t
///
/// \param line_attributes    Map with Tensors storing line attributes. The
/// first dim of each attribute must match the number of lines.
///
/// \param textures           Map of Tensors for storing textures.
///
/// \param connection  The connection object used for sending the data.
///
bool SetMeshData(
        const open3d::core::Tensor& vertices,
        const std::string& path = "",
        int time = 0,
        const std::string& layer = "",
        const std::map<std::string, open3d::core::Tensor>& vertex_attributes =
                std::map<std::string, open3d::core::Tensor>(),
        const open3d::core::Tensor& faces =
                open3d::core::Tensor({0}, open3d::core::Dtype::Int32),
        const std::map<std::string, open3d::core::Tensor>& face_attributes =
                std::map<std::string, open3d::core::Tensor>(),
        const open3d::core::Tensor& lines =
                open3d::core::Tensor({0}, open3d::core::Dtype::Int32),
        const std::map<std::string, open3d::core::Tensor>& line_attributes =
                std::map<std::string, open3d::core::Tensor>(),
        const std::map<std::string, open3d::core::Tensor>& textures =
                std::map<std::string, open3d::core::Tensor>(),
        std::shared_ptr<Connection> connection = std::shared_ptr<Connection>());

/// Function for sending Camera data.
/// \param camera      The PinholeCameraParameters object.
///
/// \param path        Path descriptor defining a location in the scene tree.
/// E.g., 'mygroup/mycam'.
///
/// \param time        The time point associated with the object.
///
/// \param layer       The layer for this object.
///
/// \param connection  The connection object used for sending the data.
///
bool SetLegacyCamera(
        const open3d::camera::PinholeCameraParameters& camera,
        const std::string& path = std::string(),
        int time = 0,
        const std::string& layer = std::string(),
        std::shared_ptr<Connection> connection = std::shared_ptr<Connection>());

/// Sets the time in the external visualizer.
/// \param time        The time value
///
/// \param connection  The connection object used for sending the data.
///
bool SetTime(
        int time,
        std::shared_ptr<Connection> connection = std::shared_ptr<Connection>());

/// Sets the object with the specified path as the active camera.
/// \param path        Path descriptor defining a location in the scene tree.
/// E.g., 'mygroup/mycam'.
///
/// \param connection  The connection object used for sending the data.
///
bool SetActiveCamera(
        const std::string& path,
        std::shared_ptr<Connection> connection = std::shared_ptr<Connection>());

}  // namespace utility
}  // namespace open3d
