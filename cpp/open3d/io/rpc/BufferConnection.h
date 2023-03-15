// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <sstream>

#include "open3d/io/rpc/ConnectionBase.h"

namespace open3d {
namespace io {
namespace rpc {

/// Implements a connection writing to a buffer
class BufferConnection : public ConnectionBase {
public:
    BufferConnection() {}

    /// Function for sending data wrapped in a zmq message object.
    std::shared_ptr<zmq::message_t> Send(zmq::message_t& send_msg);

    /// Function for sending raw data. Meant for testing purposes
    std::shared_ptr<zmq::message_t> Send(const void* data, size_t size);

    std::stringstream& buffer() { return buffer_; }
    const std::stringstream& buffer() const { return buffer_; }

private:
    std::stringstream buffer_;
};
}  // namespace rpc
}  // namespace io
}  // namespace open3d
