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

#pragma once

#include <array>
#include <map>

#include "open3d/camera/PinholeCameraParameters.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/rpc/ConnectionBase.h"
#include "open3d/t/geometry/Image.h"

namespace zmq {
class message_t;
}

namespace open3d {
namespace io {
namespace rpc {

namespace messages {
struct Status;
}

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
///                    If nullptr a default connection object will be used.
///
bool SetPointCloud(const geometry::PointCloud& pcd,
                   const std::string& path = "",
                   int time = 0,
                   const std::string& layer = "",
                   std::shared_ptr<ConnectionBase> connection =
                           std::shared_ptr<ConnectionBase>());

/// Function for sending a TriangleMesh.
/// \param mesh        The TriangleMesh object.
///
/// \param path        Path descriptor defining a location in the scene tree.
/// E.g., 'mygroup/mypoints'.
///
/// \param time        The time point associated with the object.
///
/// \param layer       The layer for this object.
///
/// \param connection  The connection object used for sending the data.
///                    If nullptr a default connection object will be used.
///
bool SetTriangleMesh(const geometry::TriangleMesh& mesh,
                     const std::string& path = "",
                     int time = 0,
                     const std::string& layer = "",
                     std::shared_ptr<ConnectionBase> connection =
                             std::shared_ptr<ConnectionBase>());

/// Function for sending general mesh data.
///
/// \param path               Path descriptor defining a location in the scene
/// tree. E.g., 'mygroup/mypoints'.
///
/// \param time               The time point associated with the object.
///
/// \param layer              The layer for this object.
///
/// \param vertices           Tensor with vertices of shape [N,3]
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
/// \param material           Basic material model for rendering a
///                           DrawableGeometry (e.g. defaultLit or
///                           defaultUnlit). Must be non-empty if any material
///                           attributes or texture maps are provided.
/// \param material_scalar_attributes Map of material scalar attributes for a
///                           DrawableGeometry  Material (e.g. "point_size",
///                           "line_width" or "base_reflectance")
/// \param material_vector_attributes  Map of material 4-vector attributes for a
///                           DrawableGeometry Material (e.g. "base_color" or
///                           "absorption_color")
/// \param texture_maps       Map of t::geometry::Image for storing textures.
///
/// \param o3d_type           The type of the geometry. This is one of
/// "PointCloud", "LineSet", "TriangleMesh". This argument should be specified
/// for partial data that has no primary key data, e.g., a triangle mesh without
/// vertices but with other attribute tensors.
///
/// \param connection         The connection object used for sending the data.
///                           If nullptr a default connection object will be
///                           used.
///
bool SetMeshData(
        const std::string& path = "",
        int time = 0,
        const std::string& layer = "",
        const core::Tensor& vertices = core::Tensor({0}, core::Float32),
        const std::map<std::string, core::Tensor>& vertex_attributes =
                std::map<std::string, core::Tensor>(),
        const core::Tensor& faces = core::Tensor({0}, core::Int32),
        const std::map<std::string, core::Tensor>& face_attributes =
                std::map<std::string, core::Tensor>(),
        const core::Tensor& lines = core::Tensor({0}, core::Int32),
        const std::map<std::string, core::Tensor>& line_attributes =
                std::map<std::string, core::Tensor>(),
        const std::string& material = "",
        const std::map<std::string, float>& material_scalar_attributes =
                std::map<std::string, float>(),
        const std::map<std::string, std::array<float, 4>>&
                material_vector_attributes =
                        std::map<std::string, std::array<float, 4>>(),
        const std::map<std::string, t::geometry::Image>& texture_maps =
                std::map<std::string, t::geometry::Image>(),
        const std::string& o3d_type = "",
        std::shared_ptr<ConnectionBase> connection =
                std::shared_ptr<ConnectionBase>());

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
///                    If nullptr a default connection object will be used.
///
bool SetLegacyCamera(const camera::PinholeCameraParameters& camera,
                     const std::string& path = "",
                     int time = 0,
                     const std::string& layer = "",
                     std::shared_ptr<ConnectionBase> connection =
                             std::shared_ptr<ConnectionBase>());

/// Sets the time in the external visualizer.
/// \param time        The time value
///
/// \param connection  The connection object used for sending the data.
///                    If nullptr a default connection object will be used.
///
bool SetTime(int time,
             std::shared_ptr<ConnectionBase> connection =
                     std::shared_ptr<ConnectionBase>());

/// Sets the object with the specified path as the active camera.
/// \param path        Path descriptor defining a location in the scene tree.
/// E.g., 'mygroup/mycam'.
///
/// \param connection  The connection object used for sending the data.
///                    If nullptr a default connection object will be used.
///
bool SetActiveCamera(const std::string& path,
                     std::shared_ptr<ConnectionBase> connection =
                             std::shared_ptr<ConnectionBase>());

}  // namespace rpc
}  // namespace io
}  // namespace open3d
