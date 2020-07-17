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

#include "open3d/utility/Console.h"
#include "open3d/utility/Messages.h"
#include "open3d/utility/RemoteFunctions.h"

namespace open3d {
namespace utility {
void SetPointCloud(const open3d::geometry::PointCloud& pcd,
                   const std::string& path,
                   int time,
                   const std::string& layer,
                   std::shared_ptr<Connection> connection) {
    if (pcd.HasPoints() == 0) {
        LogInfo("SetMeshData: point cloud is empty");
        return;
    }

    messages::SetMeshData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    msg.data.vertices = messages::Array::fromPtr((double*)pcd.points_.data(),
                                                 {pcd.points_.size(), 3});
    if (pcd.HasNormals()) {
        msg.data.vertex_attributes["normals"] = messages::Array::fromPtr(
                (double*)pcd.normals_.data(), {pcd.normals_.size(), 3});
    }
    if (pcd.HasColors()) {
        msg.data.vertex_attributes["colors"] = messages::Array::fromPtr(
                (double*)pcd.colors_.data(), {pcd.colors_.size(), 3});
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.msg_id()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    connection->send(sbuf.data(), sbuf.size());
}

}  // namespace utility
}  // namespace open3d
