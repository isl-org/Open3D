// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/BufferConnection.h"

#include <zmq.hpp>

#include "open3d/io/rpc/Messages.h"
#include "open3d/utility/Logging.h"

using namespace open3d::utility;

namespace open3d {
namespace io {
namespace rpc {

std::shared_ptr<zmq::message_t> BufferConnection::Send(
        zmq::message_t& send_msg) {
    buffer_.write((char*)send_msg.data(), send_msg.size());

    auto OK = messages::Status::OK();
    msgpack::sbuffer sbuf;
    messages::Reply reply{OK.MsgId()};
    msgpack::pack(sbuf, reply);
    msgpack::pack(sbuf, OK);
    return std::shared_ptr<zmq::message_t>(
            new zmq::message_t(sbuf.data(), sbuf.size()));
}

std::shared_ptr<zmq::message_t> BufferConnection::Send(const void* data,
                                                       size_t size) {
    zmq::message_t send_msg(data, size);
    return Send(send_msg);
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
