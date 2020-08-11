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

#include "open3d/io/rpc/MessageUtils.h"

#include <zmq.hpp>

#include "open3d/io/rpc/Messages.h"
#include "open3d/utility/Console.h"

using namespace open3d::utility;

namespace open3d {
namespace io {
namespace rpc {

std::shared_ptr<messages::Status> UnpackStatusFromReply(
        const zmq::message_t& msg, size_t& offset, bool& ok) {
    ok = false;
    if (msg.size() <= offset) {
        return std::shared_ptr<messages::Status>();
    };

    messages::Reply reply;
    messages::Status status;
    try {
        auto obj_handle =
                msgpack::unpack((char*)msg.data(), msg.size(), offset);
        obj_handle.get().convert(reply);
        if (reply.msg_id != status.MsgId()) {
            LogDebug("Expected msg with id {} but got {}", status.MsgId(),
                     reply.msg_id);
        } else {
            auto status_obj_handle =
                    msgpack::unpack((char*)msg.data(), msg.size(), offset);
            status_obj_handle.get().convert(status);
            ok = true;
        }
    } catch (std::exception& e) {
        LogDebug("Failed to parse message: {}", e.what());
        offset = msg.size();
    }
    return std::make_shared<messages::Status>(status);
}

bool ReplyIsOKStatus(const zmq::message_t& msg) {
    size_t offset = 0;
    return ReplyIsOKStatus(msg, offset);
}

bool ReplyIsOKStatus(const zmq::message_t& msg, size_t& offset) {
    bool ok;
    auto status = UnpackStatusFromReply(msg, offset, ok);
    if (ok && status && 0 == status->code) {
        return true;
    }
    return false;
}

std::string CreateSerializedRequestMessage(const std::string& msg_id) {
    messages::Request request{msg_id};
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, request);
    return std::string(sbuf.data(), sbuf.size());
}

std::tuple<const void*, size_t> GetZMQMessageDataAndSize(
        const zmq::message_t& msg) {
    return std::make_tuple(msg.data(), msg.size());
}

std::tuple<int32_t, std::string> GetStatusCodeAndStr(
        const messages::Status& status) {
    return std::make_tuple(status.code, status.str);
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
