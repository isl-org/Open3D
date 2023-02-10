// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/Connection.h"

#include <zmq.hpp>

#include "open3d/io/rpc/ZMQContext.h"
#include "open3d/utility/Logging.h"

using namespace open3d::utility;

namespace {

struct ConnectionDefaults {
    std::string address = "tcp://127.0.0.1:51454";
    int connect_timeout = 5000;
    int timeout = 10000;
} defaults;

}  // namespace

namespace open3d {
namespace io {
namespace rpc {

Connection::Connection()
    : Connection(defaults.address, defaults.connect_timeout, defaults.timeout) {
}

Connection::Connection(const std::string& address,
                       int connect_timeout,
                       int timeout)
    : context_(GetZMQContext()),
      socket_(new zmq::socket_t(*GetZMQContext(), ZMQ_REQ)),
      address_(address),
      connect_timeout_(connect_timeout),
      timeout_(timeout) {
    socket_->set(zmq::sockopt::linger, timeout_);
    socket_->set(zmq::sockopt::connect_timeout, connect_timeout_);
    socket_->set(zmq::sockopt::rcvtimeo, timeout_);
    socket_->set(zmq::sockopt::sndtimeo, timeout_);
    socket_->connect(address_.c_str());
}

Connection::~Connection() { socket_->close(); }

std::shared_ptr<zmq::message_t> Connection::Send(zmq::message_t& send_msg) {
    if (!socket_->send(send_msg, zmq::send_flags::none)) {
        zmq::error_t err;
        if (err.num()) {
            LogInfo("Connection::send() send failed with: {}", err.what());
        }
    }

    std::shared_ptr<zmq::message_t> msg(new zmq::message_t());
    if (socket_->recv(*msg)) {
        LogDebug("Connection::send() received answer with {} bytes",
                 msg->size());
    } else {
        zmq::error_t err;
        if (err.num()) {
            LogInfo("Connection::send() recv failed with: {}", err.what());
        }
    }
    return msg;
}

std::shared_ptr<zmq::message_t> Connection::Send(const void* data,
                                                 size_t size) {
    zmq::message_t send_msg(data, size);
    return Send(send_msg);
}

std::string Connection::DefaultAddress() { return defaults.address; }

}  // namespace rpc
}  // namespace io
}  // namespace open3d
