// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
