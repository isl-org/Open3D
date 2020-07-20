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

#pragma once

#include <mutex>
#include <thread>
#include <zmq.hpp>
#include "open3d/utility/Console.h"
#include "open3d/utility/Messages.h"

namespace open3d {
namespace utility {

class ReceiverBase {
public:
    ReceiverBase(const std::string& bind_address = "tcp://127.0.0.1:51454",
                 int timeout = 10000);

    ReceiverBase(const ReceiverBase&) = delete;
    ReceiverBase& operator=(const ReceiverBase&) = delete;

    ~ReceiverBase();

    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetMeshData& msg,
            const msgpack::object& obj) {
        LogInfo("ReceiverBase::ProcessMessage: messages with id {} will be "
                "ignored",
                msg.MsgId());
        return std::shared_ptr<zmq::message_t>();
    }
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::GetMeshData& msg,
            const msgpack::object& obj) {
        LogInfo("ReceiverBase::ProcessMessage: messages with id {} will be "
                "ignored",
                msg.MsgId());
        return std::shared_ptr<zmq::message_t>();
    }
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetCameraData& msg,
            const msgpack::object& obj) {
        LogInfo("ReceiverBase::ProcessMessage: messages with id {} will be "
                "ignored",
                msg.MsgId());
        return std::shared_ptr<zmq::message_t>();
    }
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetProperties& msg,
            const msgpack::object& obj) {
        LogInfo("ReceiverBase::ProcessMessage: messages with id {} will be "
                "ignored",
                msg.MsgId());
        return std::shared_ptr<zmq::message_t>();
    }
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetActiveCamera& msg,
            const msgpack::object& obj) {
        LogInfo("ReceiverBase::ProcessMessage: messages with id {} will be "
                "ignored",
                msg.MsgId());
        return std::shared_ptr<zmq::message_t>();
    }
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetTime& msg,
            const msgpack::object& obj) {
        LogInfo("ReceiverBase::ProcessMessage: messages with id {} will be "
                "ignored",
                msg.MsgId());
        return std::shared_ptr<zmq::message_t>();
    }

    void Run();
    void Stop();

private:
    void Mainloop();

    const std::string bind_address;
    const int timeout;
    zmq::socket_t socket;
    std::thread thread;
    std::mutex mutex;
    bool keep_running;
};

}  // namespace utility
}  // namespace open3d
