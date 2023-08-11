// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
