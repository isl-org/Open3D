// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

namespace zmq {
class message_t;
class socket_t;
}  // namespace zmq

namespace open3d {
namespace io {
namespace rpc {

/// Base class for all connections
class ConnectionBase {
public:
    ConnectionBase(){};
    virtual ~ConnectionBase(){};

    /// Function for sending data wrapped in a zmq message object.
    virtual std::shared_ptr<zmq::message_t> Send(zmq::message_t& send_msg) = 0;
    virtual std::shared_ptr<zmq::message_t> Send(const void* data,
                                                 size_t size) = 0;
};
}  // namespace rpc
}  // namespace io
}  // namespace open3d
