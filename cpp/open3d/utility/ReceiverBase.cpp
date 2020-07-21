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

#include "open3d/utility/ReceiverBase.h"
#include "open3d/utility/ZMQContext.h"

namespace {
std::shared_ptr<zmq::message_t> CreateStatusMessage(
        const open3d::utility::messages::Status& status) {
    msgpack::sbuffer sbuf;
    open3d::utility::messages::Reply reply{status.MsgId()};
    msgpack::pack(sbuf, reply);
    msgpack::pack(sbuf, status);
    std::shared_ptr<zmq::message_t> msg = std::shared_ptr<zmq::message_t>(
            new zmq::message_t(sbuf.data(), sbuf.size()));

    return msg;
}
}  // namespace

namespace open3d {
namespace utility {

ReceiverBase::ReceiverBase(const std::string& bind_address, int timeout)
    : bind_address(bind_address), timeout(timeout), keep_running(false) {}

ReceiverBase::~ReceiverBase() { Stop(); }

void ReceiverBase::Start() {
    const std::lock_guard<std::mutex> lock(mutex);
    if (!keep_running) {
        keep_running = true;
        thread = std::thread(&ReceiverBase::Mainloop, this);
    }
}

void ReceiverBase::Stop() {
    bool keep_running_old;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        keep_running_old = keep_running;
        if (keep_running_old) {
            keep_running = false;
        }
    }
    if (keep_running_old) {
        thread.join();
    }
}

void ReceiverBase::Mainloop() {
    socket = zmq::socket_t(GetZMQContext(), ZMQ_REP);

    socket.setsockopt(ZMQ_LINGER, 1000);
    socket.setsockopt(ZMQ_RCVTIMEO, 1000);
    socket.setsockopt(ZMQ_SNDTIMEO, timeout);

    auto limits = msgpack::unpack_limit(0xffffffff,  // array
                                        0xffffffff,  // map
                                        65536,       // str
                                        0xffffffff,  // bin
                                        0xffffffff,  // ext
                                        100          // depth
    );
    try {
        socket.bind(bind_address.c_str());
    } catch (const zmq::error_t& err) {
        LogError("ReceiverBase::Mainloop: Failed to bind address, {}",
                 err.what());
    }

    while (true) {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            if (!keep_running) break;
        }
        try {
            zmq::message_t message;
            if (!socket.recv(message)) {
                continue;
            }

            const char* buffer = (char*)message.data();
            size_t buffer_size = message.size();

            std::vector<std::shared_ptr<zmq::message_t>> replies;

            size_t offset = 0;
            while (offset < buffer_size) {
                messages::Request req;
                try {
                    auto obj_handle =
                            msgpack::unpack(buffer, buffer_size, offset,
                                            nullptr, nullptr, limits);
                    auto obj = obj_handle.get();
                    req = obj.as<messages::Request>();

                    if (false) {
                    }
#define PROCESS_MESSAGE(MSGTYPE)                                        \
    else if (MSGTYPE::MsgId() == req.msg_id) {                          \
        auto oh = msgpack::unpack(buffer, buffer_size, offset, nullptr, \
                                  nullptr, limits);                     \
        auto obj = oh.get();                                            \
        MSGTYPE msg;                                                    \
        msg = obj.as<MSGTYPE>();                                        \
        auto reply = ProcessMessage(req, msg, obj);                     \
        replies.push_back(reply);                                       \
    }
                    PROCESS_MESSAGE(messages::SetMeshData)
                    PROCESS_MESSAGE(messages::GetMeshData)
                    PROCESS_MESSAGE(messages::SetCameraData)
                    PROCESS_MESSAGE(messages::SetProperties)
                    PROCESS_MESSAGE(messages::SetActiveCamera)
                    PROCESS_MESSAGE(messages::SetTime)
                    else {
                        LogInfo("ReceiverBase::Mainloop: unsupported msg "
                                "id '{}'",
                                req.msg_id);
                        auto status = messages::Status::ErrorUnsupportedMsgId();
                        replies.push_back(CreateStatusMessage(status));
                        break;
                    }
                } catch (std::exception& err) {
                    LogInfo("ReceiverBase::Mainloop: {}", err.what());
                    auto status = messages::Status::ErrorUnpackingFailed();
                    status.str += std::string(" with ") + err.what();
                    replies.push_back(CreateStatusMessage(status));
                    break;
                }
            }
            if (replies.size() == 1) {
                socket.send(*replies[0]);
            } else {
                size_t size = 0;
                for (auto r : replies) {
                    size += r->size();
                }
                zmq::message_t reply(size);
                size_t offset = 0;
                for (auto r : replies) {
                    memcpy((char*)reply.data() + offset, r->data(), r->size());
                    offset += r->size();
                }
                socket.send(reply);
            }
        } catch (const zmq::error_t& err) {
            LogInfo("ReceiverBase::Mainloop: {}", err.what());
        }
    }
    socket.close();
}  // namespace utility

}  // namespace utility
}  // namespace open3d
