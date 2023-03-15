// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>

#include "open3d/io/rpc/ConnectionBase.h"
#include "open3d/io/rpc/ZMQContext.h"

namespace open3d {
namespace io {
namespace rpc {

/// This class implements the Connection which is used as default in all
/// functions.
class Connection : public ConnectionBase {
public:
    /// Creates a connection with the default parameters
    Connection();

    /// Creates a Connection object used for sending data.
    /// \param address          The address of the receiving end.
    ///
    /// \param connect_timeout  The timeout for the connect operation of the
    /// socket.
    ///
    /// \param timeout          The timeout for sending data.
    ///
    Connection(const std::string& address, int connect_timeout, int timeout);
    ~Connection();

    /// Function for sending data wrapped in a zmq message object.
    std::shared_ptr<zmq::message_t> Send(zmq::message_t& send_msg);

    /// Function for sending raw data. Meant for testing purposes
    std::shared_ptr<zmq::message_t> Send(const void* data, size_t size);

    static std::string DefaultAddress();

private:
    std::shared_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
    const std::string address_;
    const int connect_timeout_;
    const int timeout_;
};
}  // namespace rpc
}  // namespace io
}  // namespace open3d
