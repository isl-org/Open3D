// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/io/rpc/MessageProcessorBase.h"

#include <zmq.hpp>

#include "open3d/io/rpc/Messages.h"
#include "open3d/io/rpc/ZMQContext.h"

using namespace open3d::utility;

namespace {
std::shared_ptr<zmq::message_t> CreateStatusMessage(
        const open3d::io::rpc::messages::Status& status) {
    msgpack::sbuffer sbuf;
    open3d::io::rpc::messages::Reply reply{status.MsgId()};
    msgpack::pack(sbuf, reply);
    msgpack::pack(sbuf, status);
    std::shared_ptr<zmq::message_t> msg =
            std::make_shared<zmq::message_t>(sbuf.data(), sbuf.size());

    return msg;
}

template <class T>
std::shared_ptr<zmq::message_t> IgnoreMessage(
        const open3d::io::rpc::messages::Request& req,
        const T& msg,
        const msgpack::object_handle& obj) {
    LogInfo("MessageProcessorBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status = open3d::io::rpc::messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}

}  // namespace

namespace open3d {
namespace io {
namespace rpc {

MessageProcessorBase::MessageProcessorBase() {}

MessageProcessorBase::~MessageProcessorBase() {}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetMeshData& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::GetMeshData& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetCameraData& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetProperties& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetActiveCamera& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetTime& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
