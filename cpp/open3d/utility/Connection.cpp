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

#include "open3d/utility/Connection.h"
#include "open3d/utility/Console.h"
namespace {
zmq::context_t context;

struct ConnectionDefaults {
    std::string address = "tcp://localhost:51454";
    int connect_timeout = 1000;
    int timeout = 10000;
} defaults;
}  // namespace

namespace open3d {
namespace utility {

Connection::Connection()
    : socket(new zmq::socket_t(context, ZMQ_REQ)),
      address(defaults.address),
      connect_timeout(defaults.connect_timeout),
      timeout(defaults.timeout) {
    init();
}

Connection::Connection(const std::string& address,
                       int connect_timeout,
                       int timeout)
    : socket(new zmq::socket_t(context, ZMQ_REQ)),
      address(address),
      connect_timeout(connect_timeout),
      timeout(timeout) {
    init();
}

Connection::~Connection() { socket->close(); }

void Connection::init() {
    socket->setsockopt(ZMQ_LINGER, timeout);
    socket->setsockopt(ZMQ_CONNECT_TIMEOUT, connect_timeout);
    socket->setsockopt(ZMQ_RCVTIMEO, timeout);
    socket->setsockopt(ZMQ_SNDTIMEO, timeout);
    socket->connect(address.c_str());
}

std::shared_ptr<zmq::message_t> Connection::send(const void* buffer,
                                                 size_t len) {
    zmq::message_t send_msg(buffer, len);
    if (socket->send(send_msg)) {
#ifndef NDEBUG
        LogDebug("Connection::send {} bytes", len);
#endif
    } else {
        // TODO print warning or info
    }

    std::shared_ptr<zmq::message_t> msg(new zmq::message_t());
    if (socket->recv(*msg)) {
#ifndef NDEBUG
        LogDebug("Connection::send received answer with {} bytes", msg->size());
#endif
    } else {
        // TODO print warning or info
    }
    return msg;
}

}  // namespace utility
}  // namespace open3d
