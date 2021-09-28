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

#include "open3d/io/rpc/MessageProcessorBase.h"

namespace open3d {

namespace geometry {
class Geometry3D;
}  // namespace geometry

namespace visualization {

namespace gui {
class Window;
}  // namespace gui

/// MessageProcessor implementation which interfaces with the Open3DScene and a
/// Window.
class MessageProcessor : public io::rpc::MessageProcessorBase {
public:
    using OnGeometryFunc = std::function<void(
            std::shared_ptr<geometry::Geometry3D>,  // geometry
            const std::string&,                     // path
            int,                                    // time
            const std::string&)>;                   // layer
    MessageProcessor(gui::Window* window, OnGeometryFunc on_geometry)
        : MessageProcessorBase(), window_(window), on_geometry_(on_geometry) {}

    std::shared_ptr<zmq::message_t> ProcessMessage(
            const io::rpc::messages::Request& req,
            const io::rpc::messages::SetMeshData& msg,
            const msgpack::object_handle& obj) override;

private:
    gui::Window* window_;
    OnGeometryFunc on_geometry_;

    void SetGeometry(std::shared_ptr<geometry::Geometry3D> geom,
                     const std::string& path,
                     int time,
                     const std::string& layer);
};

}  // namespace visualization
}  // namespace open3d
